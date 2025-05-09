import logging
import httpx

from asyncio import TaskGroup
from ollama import AsyncClient
from typing import Union, Mapping, Any, Optional

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.DocumentIndexHandler import DocumentIndexHandler
from millegrilles_ollama_relai.OllamaChatHandler import OllamaChatHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext


class MessageHandler:

    def __init__(self, context: OllamaContext, chat_handler: OllamaChatHandler, document_handler: DocumentIndexHandler):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__chat_handler = chat_handler
        self.__document_handler = document_handler
        # self.__attachment_handler = attachment_handler
        # self.__waiting_ids: dict[str, dict] = dict()  # Key = chat_id, value = {reply_to, correlation_id}

        self.__ollama_status: Union[bool, dict] = False
        self.__ollama_models: Optional[Mapping[str, Any]] = None

    def get_async_client(self) -> AsyncClient:
        return self.__context.get_async_client()
        # configuration = self.__context.configuration
        # connection_url = self.__context.configuration.ollama_url
        # if connection_url.lower().startswith('https://'):
        #     # Use a millegrille certificate authentication
        #     cert = (configuration.cert_path, configuration.key_path)
        #     client = AsyncClient(host=self.__context.configuration.ollama_url, verify=configuration.ca_path, cert=cert)
        # else:
        #     client = AsyncClient(host=self.__context.configuration.ollama_url)
        # return client

    async def run(self):
        async with TaskGroup() as group:
            # group.create_task(self.emit_event_thread(self.__waiting_ids, 'attente'))
            group.create_task(self.ollama_watchdog_thread())

    async def ollama_watchdog_thread(self):
        """ Regularly checks status of ollama connection. """
        while self.__context.stopping is False:
            available = self.__ollama_status is not False
            await self.check_ollama_status()
            now_available = self.__ollama_status is not False

            if available != now_available:
                self.__logger.info("ollama Status now %s" % now_available)
                producer = await self.__context.get_producer()
                status_event = {'event_type': 'availability', 'available': now_available}
                await producer.event(
                    status_event, 'ollama_relai', 'status',
                    exchange=ConstantesMilleGrilles.SECURITE_PRIVE)

            await self.__context.wait(10)

    # async def emit_event_thread(self, correlations: dict[str, dict], event_name: str):
    #     # Wait for the initialization of the producer
    #     await self.__context.wait(5)
    #
    #     while self.__context.stopping is False:
    #         producer = await self.__context.get_producer()
    #
    #         for correlation_info in correlations.values():
    #             await producer.reply({'ok': True, 'evenement': event_name},
    #                                  correlation_id=correlation_info['correlation_id'], reply_to=correlation_info['reply_to'],
    #                                  attachments={'streaming': True})
    #
    #         await self.__context.wait(15)

    async def handle_requests(self, message: MessageWrapper):
        # Wait for the producer to be ready
        _producer = await self.__context.get_producer()

        # Confirm routing key
        rk_split = message.routing_key.split('.')
        type_message = rk_split[0]
        domaine = rk_split[1]
        action = rk_split.pop()

        if type_message != 'requete':
            raise Exception('Wrong message kind, must be request')

        if domaine != 'ollama_relai':
            raise Exception('Domaine must be ollama_relai')

        if action == 'ping':
            available, message = await self.ollama_ping()
            return {'ok': available, 'err': message}
        elif action == 'getModels':
            models = await self.check_ollama_list_models()
            return {'ok': True, 'models': models}
        elif action == 'queryDocuments':
            return self.__document_handler.query_documents(message)

        return {'ok': False, 'code': 404, 'err': 'Unknown action'}

    async def handle_commands(self, message: MessageWrapper):
        # Confirm routing key
        rk_split = message.routing_key.split('.')
        type_message = rk_split[0]
        domaine = rk_split[1]
        action = rk_split.pop()

        if type_message != 'commande':
            raise Exception('Wrong message kind, must be command')

        if domaine != 'ollama_relai':
            raise Exception('Domaine must be ollama_relai')

        if action == 'chat':
            await self.__chat_handler.register_query(message)
            return False  # We'll stream the messages, no automatic response here
        elif action == 'indexDocuments':
            return await self.__document_handler.index_documents(message)

        return {'ok': False, 'code': 404, 'err': 'Unknown action'}

    async def check_ollama_status(self):
        client = self.__context.get_async_client()
        try:
            # Test connection by getting currently loaded model information
            async with self.__context.ollama_http_semaphore:
                self.__ollama_status = await client.ps()
                self.__ollama_models = await client.list()
            # return dict(status), models
        except httpx.ConnectError:
            # Failed to connect
            self.__ollama_status = False

    async def check_ollama_list_models(self) -> Union[bool, list]:
        # client = self.__context.get_async_client()
        try:
            # Test connection by getting currently loaded model information
            # async with self.__context.ollama_http_semaphore:
            #     models = await client.list()
            models = self.__ollama_models
            model_list = list()
            for model in models['models']:
                name = model['name']
                model_list.append({'name': name})
            return model_list
        except httpx.ConnectError:
            # Failed to connect
            return False

    async def ollama_ping(self) -> (bool, str):
        available = self.__ollama_status is not False
        return available, ''
