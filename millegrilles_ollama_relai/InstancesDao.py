import logging
import httpx
import re
import base64

from typing import Any, TypedDict, Optional, AsyncIterator, Union

from openai.types import Model as OpenaiModel
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam, ChatCompletion, ChatCompletionChunk, ChatCompletionContentPartTextParam, \
    ChatCompletionContentPartImageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from pydantic import BaseModel

from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration

from ollama import AsyncClient as OllamaAsyncClient, ResponseError as OllamaResponseError, ListResponse, ShowResponse, \
    ChatResponse, GenerateResponse
from openai import AsyncClient as OpenaiAsyncClient, APIConnectionError as OpenaiAPIConnectionError, OpenAIError

from millegrilles_ollama_relai.OllamaContext import OllamaContext
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

class OpenaiChatResponseWrapper(MessageWrapper):

    def __init__(self, value: ChatCompletion):
        self.__value = value
        choice = value.choices[0]
        self.message = MessageContent(role='assistant', content=choice.message.content, thinking=None, tool_calls=None)
        self.done = choice.finish_reason is not None

    def to_dict(self):
        return {'message': dict(self.message), 'done': self.done}

class OpenaiChatCompletionStreamWrapper(MessageWrapper):

    def __init__(self, value: ChatCompletionChunk):
        self.__value = value
        choice = value.choices[0]
        self.message = MessageContent(role='assistant', content=choice.delta.content, thinking=None, tool_calls=None)
        self.done = choice.finish_reason is not None

    def to_dict(self):
        return {'message': dict(self.message), 'done': self.done}


class ModelParams:

    def __init__(self, id: str):
        self.id = id

    @property
    def name(self) -> str:
        raise NotImplementedError('must implement')

    @property
    def capabilities(self):
        raise NotImplementedError('must implement')

    @property
    def context_length(self):
        raise NotImplementedError('must implement')


class OllamaModelParams(ModelParams):

    def __init__(self, id: str, model: ListResponse.Model, show_response: ShowResponse):
        super().__init__(id)
        self.__model = model
        self.__show_response = show_response
        self.__context_length = 4096

        self.__load_parameters()

    def __load_parameters(self):
        # Overrides
        try:
            for param in self.__show_response.parameters.splitlines():
                group = re.search(r'(\w+)\s+(\S+)', param)
                key = group[1]
                value = group[2]
                if key == 'num_ctx':
                    self.__context_length = int(value)
        except AttributeError:
            pass  # No custom parameters

    @property
    def name(self):
        return self.__model.model

    @property
    def capabilities(self):
        return self.__show_response.capabilities

    @property
    def context_length(self):
        return self.__context_length


class OpenaiModelParams(ModelParams):

    def __init__(self, id: str, model: OpenaiModel):
        super().__init__(id)
        self.__model = model
        self.__context_length = 4096
        self.__capabilities = ['completion', 'vision']
        self.__load_parameters()

    def __load_parameters(self):
        # Overrides
        try:
            self.__context_length = self.__model.model_extra['max_model_len']
        except AttributeError:
            pass  # No custom parameters

    @property
    def name(self):
        return self.__model.id

    @property
    def capabilities(self):
        return self.__capabilities

    @property
    def context_length(self):
        return self.__context_length


class InstanceDao:

    def __init__(self, context: OllamaContext, url: str):
        self._context = context
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
                   tools: Optional[list] = None, response_format: Optional[BaseModel] = None, max_len: Optional[int] = None, temperature=0.1):
        raise NotImplementedError('must implement')

    async def generate(self, model: str, prompt: str, system: Optional[str] = None, images: Optional[list[Union[str, bytes]]] = None,
                       stream=False, think: Optional[bool] = None, response_format: Optional[BaseModel] = None, max_len: Optional[int] = None,
                       temperature=0.1) -> Union[MessageWrapper, AsyncIterator[OllamaChatResponseWrapper]]:
        raise NotImplementedError('must implement')

    def get_client_options(self) -> dict:
        configuration = self._context.configuration
        connection_url = self._url
        if connection_url.lower().startswith('https://'):
            # Use a millegrille certificate authentication
            cert = (configuration.cert_path, configuration.key_path)
            params = {'verify':configuration.ca_path, 'cert':cert}
        else:
            params = {}
        return params


class OllamaInstanceDao(InstanceDao):

    def __init__(self, context: OllamaContext, url: str):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        super().__init__(context, url)

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
        try:
            models_response = await client.list()
        except (ConnectionError, httpx.ConnectError, OllamaResponseError) as e:
            raise ClientDaoError(e)
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
                try:
                    show_model = await client.show(model.model)
                except (ConnectionError, httpx.ConnectError, OllamaResponseError) as e:
                    raise ClientDaoError(e)
                model_params = OllamaModelParams(model_id, model, show_model)
                updated_models[model_id] = model_params
                added_models.append(model_id)

        # Any remaining model has been removed from the list
        removed_models = list(known_models.keys())

        # Keep for reference
        self._known_models = updated_models

        return ModelsUpdate(models=updated_models, added=added_models, removed=removed_models)

    async def chat(self, model: str, messages: Optional[list] = None, stream=False, think=False,
                   tools: Optional[list] = None, response_format: Optional[BaseModel] = None, max_len: Optional[int] = None,
                   temperature=0.1) -> Union[MessageWrapper, AsyncIterator[OllamaChatResponseWrapper]]:

        options = dict()
        if max_len:
            options['num_ctx'] = max_len
        if temperature:
            options['temperature'] = temperature
        if len(options) == 0:
            options = None

        try:
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
        except (ConnectionError, httpx.ConnectError, OllamaResponseError) as e:
            raise ClientDaoError(e)

    async def generate(self, model: str, prompt: str, system: Optional[str] = None, images: Optional[list[Union[str, bytes]]] = None,
                       stream=False, think: Optional[bool] = None, response_format: Optional[BaseModel] = None, max_len: Optional[int] = None,
                       temperature=0.1) -> Union[MessageWrapper, AsyncIterator[OllamaChatResponseWrapper]]:

        options = dict()
        if max_len:
            options['num_ctx'] = max_len
        if temperature:
            options['temperature'] = temperature
        if len(options) == 0:
            options = None

        try:
            response = await self.get_async_client().generate(
                model=model,
                prompt=prompt,
                system=system,
                images=images,
                stream=stream,
                think=think,
                format=response_format,
                options=options,
            )

            if isinstance(response, AsyncIterator):
                return self.__wrap_stream_generate(response)

            wrapper = OllamaGenerateResponseWrapper(response)
            return wrapper
        except (ConnectionError, httpx.ConnectError, OllamaResponseError) as e:
            raise ClientDaoError(e)

    async def __wrap_stream(self, response: AsyncIterator) -> AsyncIterator:
        async for chunk in response:
            yield OllamaChatResponseWrapper(chunk)

    async def __wrap_stream_generate(self, response: AsyncIterator) -> AsyncIterator:
        async for chunk in response:
            yield OllamaGenerateResponseWrapper(chunk)


class OpenAiInstanceDao(InstanceDao):

    def __init__(self, context: OllamaContext, url: str):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        super().__init__(context, url)

    def get_async_client(self, timeout=None) -> OpenaiAsyncClient:
        options = self.get_client_options()
        if options.get('verify'):
            ssl_context = self._context.ssl_context
        else:
            ssl_context = None
        httpx_client = httpx.AsyncClient(verify=ssl_context, cert=options.get('cert'))
        return OpenaiAsyncClient(http_client=httpx_client, base_url=self._url, api_key="DUMMY")

    async def status(self) -> bool:
        client = self.get_async_client(timeout=1)
        try:
            models = await client.models.list()
        except OpenaiAPIConnectionError:
            return False
        return models is not None

    async def models(self):
        client = self.get_async_client(timeout=3)
        try:
            model_data, metadata = await client.models.list()
            models = model_data[1]
        except (ConnectionError, httpx.ConnectError, OpenAIError) as e:
            raise ClientDaoError(e)

        known_models = self._known_models.copy()
        updated_models = dict()

        added_models = list()
        for model in models:
            model_name = model.id
            model_id = model_name_to_id(model_name)
            try:
                updated_models[model_id] = known_models[model_id]
                del known_models[model_id]
            except KeyError:
                # New model, fetch information
                model_params = OpenaiModelParams(model_id, model)
                updated_models[model_id] = model_params
                added_models.append(model_id)
                pass

        # Any remaining model has been removed from the list
        removed_models = list(known_models.keys())

        # Keep for reference
        self._known_models = updated_models

        return ModelsUpdate(models=updated_models, added=added_models, removed=removed_models)

    def __map_messages(self, messages: list):
        mapped_messages = list()
        last_role = None
        for message in messages:
            role = message['role']
            if role == 'system':
                mapped_messages.append(ChatCompletionSystemMessageParam(content=message['content'], role="system"))
            elif role == 'user':
                if last_role == 'user':
                    mapped_messages[-1]['content'] += '\n\n' + message['content']  # Concatenate to last message
                else:
                    mapped_messages.append(ChatCompletionUserMessageParam(content=message['content'], role="user"))
            elif role == 'assistant':
                if last_role == 'assistant':
                    mapped_messages[-1]['content'] += '\n\n' + message['content']  # Concatenate to last message
                else:
                    mapped_messages.append(ChatCompletionAssistantMessageParam(content=message['content'], role="assistant"))
            last_role = role

        return mapped_messages

    def __map_format(self, format_object: BaseModel) -> ResponseFormatJSONSchema:
        json_schema_mapped = format_object.model_json_schema()
        json_schema = JSONSchema(name='json_response', schema=json_schema_mapped)
        return ResponseFormatJSONSchema(
            json_schema=json_schema,
            type="json_schema"
        )

    async def chat(self, model: str, messages: Optional[list] = None, stream=False, think=False,
                   tools: Optional[list] = None, response_format: Optional[BaseModel] = None, max_len=1024, temperature=0.1):

        if messages:
            mapped_messages = self.__map_messages(messages)
        else:
            mapped_messages = None

        if response_format:
            mapped_format = self.__map_format(response_format)
        else:
            mapped_format = None

        response = await self.get_async_client().chat.completions.create(
            messages=mapped_messages,
            model=model,
            response_format=mapped_format,
            max_tokens=max_len,
            stream=stream,
            # think=think,
            temperature=temperature
        )

        if isinstance(response, AsyncIterator):
            return self.__wrap_stream(response)

        return OpenaiChatResponseWrapper(response)

    async def generate(self, model: str, prompt: str, system: Optional[str] = None, images: Optional[list[Union[str, bytes]]] = None,
                       stream=False, think: Optional[bool] = None, response_format: Optional[BaseModel] = None, max_len=1024,
                       temperature=0.1) -> Union[MessageWrapper, AsyncIterator[OllamaChatResponseWrapper]]:

        messages = list()
        if system:
            messages.append(ChatCompletionSystemMessageParam(content=system, role="system"))
        if prompt:
            if images:
                image = images[0]
                if len(images) > 1:
                    raise ValueError("Only 1 image supported")

                if isinstance(image, str):
                    b64_image = image
                elif isinstance(image, bytes):
                    b64_image = base64.b64encode(image).decode('utf-8')
                else:
                    raise ValueError('Image in wrong format')

                messages.append(ChatCompletionUserMessageParam(role="user", content=[
                    ChatCompletionContentPartTextParam(type="text", text=prompt),
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=f"data:image/png;base64,{b64_image}", detail="high")
                    )
                ]))
            else:
                messages.append(ChatCompletionUserMessageParam(content=prompt, role="user"))

        return await self.chat(model, messages, stream, think, None, response_format, max_len, temperature)

    async def __wrap_stream(self, response: AsyncIterator):
        async for chunk in response:
            yield OpenaiChatCompletionStreamWrapper(chunk)
