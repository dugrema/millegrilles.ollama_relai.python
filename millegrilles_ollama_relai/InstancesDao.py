import logging
import httpx
import re

from typing import Any, TypedDict, Optional, AsyncIterator, Union

from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration
from millegrilles_ollama_relai.OllamaContext import OllamaContext

from ollama import AsyncClient as OllamaAsyncClient, ResponseError as OllamaResponseError, ListResponse, ShowResponse, \
    ChatResponse, GenerateResponse
from openai import AsyncClient as OpenaiAsyncClient, APIConnectionError as OpenaiAPIConnectionError

from millegrilles_ollama_relai.Util import model_name_to_id


class ModelsUpdate(TypedDict):
    models: dict[str, Any]
    added: list[str]
    removed: list[str]


class ClientDaoError(Exception):
    pass


class MessageContent(TypedDict):
    role: str
    content: str
    thinking: Optional[str]
    tool_calls: Optional[dict]


class MessageWrapper:
    message: MessageContent
    done: bool

    def get_content(self) -> str:
        pass


class OllamaChatResponseWrapper(MessageWrapper):

    def __init__(self, value: ChatResponse):
        self.__value = value
        self.message = MessageContent(**dict(value.message))
        self.done = value.done

    def to_dict(self):
        return {'message': dict(self.message), 'done': self.done}


class OllamaGenerateResponseWrapper(MessageWrapper):

    def __init__(self, value: GenerateResponse):
        self.__value = value
        self.message = MessageContent(role='assistant', content=value.response, thinking=value.thinking, tool_calls=None)
        self.done = value.done

    def to_dict(self):
        return {'message': dict(self.message), 'done': self.done}


class OllamaModelParams:

    def __init__(self, id: str, model: ListResponse.Model, show_response: ShowResponse):
        self.id = id
        self.model = model
        self.show_response = show_response
        self.__context_length = 4096

        self.__load_parameters()

    def __load_parameters(self):
        # Note: Unless overridden by parameters, the ollama num_ctx defaults to 4096 regardless of model capacity
        # info = self.show_response.modelinfo
        # try:
        #     architecture = info['general.architecture']
        #     self.__context_length = info[f'{architecture}.context_length']
        # except KeyError:
        #     pass

        # Overrides
        try:
            for param in self.show_response.parameters.splitlines():
                group = re.search(r'(\w+)\s+(\S+)', param)
                key = group[1]
                value = group[2]
                if key == 'num_ctx':
                    self.__context_length = int(value)
        except AttributeError:
            pass  # No custom parameters

    @property
    def capabilities(self):
        return self.show_response.capabilities

    @property
    def context_length(self):
        return self.__context_length



class InstanceDao:

    def __init__(self, configuration: OllamaConfiguration, url: str):
        self._configuration = configuration
        self._url = url

        self._known_models: dict[str, Any] = dict()

    async def status(self) -> bool:
        """
        :return: True if the connection is working properly, False otherwise.
        """
        raise NotImplementedError('must implement')

    async def models(self) -> ModelsUpdate:
        raise NotImplementedError('must implement')

    async def chat(self, model: str, messages: Optional[list] = None, stream=False, think=False,
                   tools: Optional[list] = None, format: Optional[dict] = None, max_len: Optional[int] = None, temperature=0.1):
        raise NotImplementedError('must implement')

    async def generate(self, model: str, prompt: str, system: Optional[str] = None, images: Optional[list[Union[str, bytes]]] = None,
                       stream=False, think: Optional[bool] = None, format: Optional[dict] = None, max_len: Optional[int] = None,
                       temperature=0.1) -> Union[MessageWrapper, AsyncIterator[OllamaChatResponseWrapper]]:
        raise NotImplementedError('must implement')

    def get_client_options(self) -> dict:
        configuration = self._configuration
        connection_url = self._url
        if connection_url.lower().startswith('https://'):
            # Use a millegrille certificate authentication
            cert = (configuration.cert_path, configuration.key_path)
            params = {'verify':configuration.ca_path, 'cert':cert}
        else:
            params = {}
        return params


class OllamaInstanceDao(InstanceDao):

    def __init__(self, configuration: OllamaConfiguration, url: str):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        super().__init__(configuration, url)

    def get_async_client(self, timeout=None) -> OllamaAsyncClient:
        options = self.get_client_options()
        return OllamaAsyncClient(host=self._url, timeout=timeout, **options)

    async def status(self) -> bool:
        self.__logger.debug(f"Checking with {self._url}")
        client = self.get_async_client(timeout=5)
        try:
            # Test connection by getting currently loaded model information
            await client.ps()
            return True
        except (ConnectionError, httpx.ConnectError, OllamaResponseError) as e:
            # Failed to connect
            if self.__logger.isEnabledFor(logging.DEBUG):
                self.__logger.exception(f"Connection error on {self._url}")
            else:
                self.__logger.info(f"Connection error on {self._url}: %s" % str(e))

            # Reset status, avoids picking this instance up
            return False

    async def models(self):
        client = self.get_async_client(timeout=3)
        models_response = await client.list()
        known_models = self._known_models.copy()
        updated_models = dict()

        added_models = list()
        for model in models_response.models:
            model_id = model_name_to_id(model.model)
            try:
                updated_models[model_id] = known_models[model_id]
                del known_models[model_id]
            except KeyError:
                # New model, fetch information
                show_model = await client.show(model.model)
                model_params = OllamaModelParams(model_id, model, show_model)
                updated_models[model_id] = model_params
                added_models.append(model_id)

        # Any remaining model has been removed from the list
        removed_models = list(known_models.keys())

        # Keep for reference
        self._known_models = updated_models

        return ModelsUpdate(models=updated_models, added=added_models, removed=removed_models)

    async def chat(self, model: str, messages: Optional[list] = None, stream=False, think=False,
                   tools: Optional[list] = None, format: Optional[dict] = None, max_len: Optional[int] = None,
                   temperature=0.1) -> Union[MessageWrapper, AsyncIterator[OllamaChatResponseWrapper]]:

        options = dict()
        if max_len:
            options['num_ctx'] = max_len
        if temperature:
            options['temperature'] = temperature
        if len(options) == 0:
            options = None

        response = await self.get_async_client().chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            think=think,
            options=options,
        )

        if isinstance(response, AsyncIterator):
            return self.__wrap_stream(response)

        wrapper = OllamaChatResponseWrapper(response)
        return wrapper

    async def generate(self, model: str, prompt: str, system: Optional[str] = None, images: Optional[list[Union[str, bytes]]] = None,
                       stream=False, think: Optional[bool] = None, format: Optional[dict] = None, max_len: Optional[int] = None,
                       temperature=0.1) -> Union[MessageWrapper, AsyncIterator[OllamaChatResponseWrapper]]:

        options = dict()
        if max_len:
            options['num_ctx'] = max_len
        if temperature:
            options['temperature'] = temperature
        if len(options) == 0:
            options = None

        response = await self.get_async_client().generate(
            model=model,
            prompt=prompt,
            system=system,
            images=images,
            stream=stream,
            think=think,
            format=format,
            options=options,
        )

        if isinstance(response, AsyncIterator):
            return self.__wrap_stream_generate(response)

        wrapper = OllamaGenerateResponseWrapper(response)
        return wrapper

    async def __wrap_stream(self, response: AsyncIterator) -> AsyncIterator:
        async for chunk in response:
            yield OllamaChatResponseWrapper(chunk)

    async def __wrap_stream_generate(self, response: AsyncIterator) -> AsyncIterator:
        async for chunk in response:
            yield OllamaGenerateResponseWrapper(chunk)


class OpenAiInstanceDao(InstanceDao):

    def __init__(self, configuration: OllamaConfiguration, url: str):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        super().__init__(configuration, url)

    def get_async_client(self, timeout=None) -> OpenaiAsyncClient:
        options = self.get_client_options()
        return OpenaiAsyncClient(api_key='DUMMY', base_url=self._url, timeout=timeout, **options)

    async def status(self) -> bool:
        client = self.get_async_client(timeout=1)
        try:
            models = await client.models.list()
        except OpenaiAPIConnectionError:
            return False
        return models is not None
