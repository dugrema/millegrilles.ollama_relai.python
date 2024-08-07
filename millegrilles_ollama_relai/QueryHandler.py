import asyncio
import requests

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.Context import OllamaRelaiContext


class QueryHandler:

    def __init__(self, context: OllamaRelaiContext):
        self.__context = context

    async def handle_query(self, message: MessageWrapper):
        producer = self.__context.producer
        await producer.producer_pret().wait()

        # Confirm routing key
        rk_split = message.routing_key.split('.')
        type_message = rk_split[0]
        domaine = rk_split[1]
        action = rk_split.pop()
        enveloppe = message.certificat

        if type_message != 'commande':
            raise Exception('Wrong message kind, must be command')

        if domaine != 'ollama_relai':
            raise Exception('Domaine must be ollama_relai')

        if action not in ['chat', 'generate']:
            raise Exception('Actions can only be one of: chat, generate')

        # TODO: Start emitting "queued up" events for the client

        # Re-emit the message on the traitement Q
        await producer.executer_commande(message.original, domaine='ollama_relai', action='traitement',
                                         exchange=ConstantesMilleGrilles.SECURITE_PROTEGE, noformat=True, nowait=True)

        return {'ok': True}

    async def process_query(self, message: MessageWrapper):
        producer = self.__context.producer
        await producer.producer_pret().wait()

        action = message.routage['action']
        url_query = f'{self.__context.configuration.ollama_url}/api/{action}'
        content = message.parsed.copy()
        del content['__original']
        response = await asyncio.to_thread(requests.post, url_query, json=content)
        response_content = response.json()
        print("Response: %s" % response_content['response'])

        # Send response as result event to client

        return None
