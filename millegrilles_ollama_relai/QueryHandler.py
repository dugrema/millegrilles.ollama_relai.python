import logging
import httpx

from asyncio import TaskGroup
from ollama import AsyncClient
from typing import Union

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.OllamaChatHandler import OllamaChatHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext


class QueryHandler:

    def __init__(self, context: OllamaContext, chat_handler: OllamaChatHandler):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__chat_handler = chat_handler
        # self.__attachment_handler = attachment_handler
        # self.__waiting_ids: dict[str, dict] = dict()  # Key = chat_id, value = {reply_to, correlation_id}

        self.__ollama_status: Union[bool, dict] = False

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
            self.__ollama_status = await self.check_ollama_status()
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
            raise Exception('Wrong message kind, must be command')

        if domaine != 'ollama_relai':
            raise Exception('Domaine must be ollama_relai')

        if action == 'ping':
            available, message = await self.ollama_ping()
            return {'ok': available, 'err': message}
        elif action == 'getModels':
            models = await self.check_ollama_list_models()
            return {'ok': True, 'models': models}

        raise Exception('Unknown request: %s' % action)

    async def handle_query(self, message: MessageWrapper):
        producer = await self.__context.get_producer()

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
        elif action == 'indexDocument':
            await self.__chat_handler.register_query(message)
            return False  # We'll stream the messages, no automatic response here
        else:
            # if action not in ['chat', 'generate']:
            raise Exception('Actions can only be one of: chat')

        # # Start emitting "waiting" events to tell the client it is still queued-up (keep-alive)
        # chat_id = message.id
        # self.__waiting_ids[chat_id] = {'reply_to': message.reply_to, 'correlation_id': message.correlation_id}
        #
        # # Re-emit the message on the traitement Q
        # attachements = {'correlation_id': message.correlation_id, 'reply_to': message.reply_to}
        # attachements.update(message.original['attachements'])
        # await producer.command(
        #     message.original, domain='ollama_relai', action='traitement', exchange=ConstantesMilleGrilles.SECURITE_PROTEGE,
        #     noformat=True, nowait=True, attachments=attachements)
        #
        # await producer.reply(
        #     {'ok': True, 'partition': chat_id, 'stream': True, 'reponse': 1},
        #     reply_to=message.reply_to, correlation_id=message.correlation_id,
        #     attachments={'streaming': True}
        # )
        #
        # return False  # We'll stream the messages, no automatic response here

    async def check_ollama_status(self) -> Union[bool, dict]:
        client = self.__context.get_async_client()
        try:
            # Test connection by getting currently loaded model information
            status = await client.ps()
            return dict(status)
        except httpx.ConnectError:
            # Failed to connect
            return False

    async def check_ollama_list_models(self) -> Union[bool, list]:
        client = self.__context.get_async_client()
        try:
            # Test connection by getting currently loaded model information
            models = await client.list()
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
