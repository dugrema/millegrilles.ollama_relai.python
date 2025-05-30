import logging

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

    async def run(self):
        await self.__context.wait()

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
            available = self.__context.ollama_status
            return {'ok': available}
        elif action == 'getModels':
            models = self.__context.ollama_models
            return {'ok': True, 'models': models}
        elif action == 'queryRag':
            return await self.__document_handler.query_rag(message)

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

        return {'ok': False, 'code': 404, 'err': 'Unknown action'}
