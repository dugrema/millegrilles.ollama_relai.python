import asyncio
import logging

import requests

from typing import Optional

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.Context import OllamaRelaiContext


class QueryHandler:

    def __init__(self, context: OllamaRelaiContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__waiting_ids: set[str] = set()
        self.__processing_ids: set[str] = set()
        self.__stop_event: Optional[asyncio.Event] = None

    async def run(self, stop_event: asyncio.Event):
        self.__stop_event = stop_event
        tasks = [
            asyncio.create_task(self.emit_event_thread(self.__waiting_ids, 'attente')),
            asyncio.create_task(self.emit_event_thread(self.__processing_ids, 'encours')),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if self.__stop_event.is_set() is False:
            self.__logger.warning("Threads shutting down")
            self.__stop_event.set()

    async def emit_event_thread(self, partition_ids: set[str], event_name: str):
        # Wait for the initialization of the producer
        try:
            await asyncio.wait_for(self.__stop_event.wait(), 5)
        except asyncio.TimeoutError:
            pass  # Ok, this is supposed to time out

        while self.__stop_event.is_set() is False:
            producer = self.__context.producer
            await producer.producer_pret().wait()

            for partition_id in partition_ids:
                await producer.emettre_evenement({}, domaine='ollama_relai', action=event_name, partition=partition_id,
                                                 exchanges=ConstantesMilleGrilles.SECURITE_PRIVE)

            try:
                await asyncio.wait_for(self.__stop_event.wait(), 15)
            except asyncio.TimeoutError:
                pass  # This is supposed to time out

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

        # Start emitting "waiting" events to tell the client it is still queued-up
        try:
            chat_id = message.parsed['chat_id']
        except KeyError:
            chat_id = message.id
        self.__waiting_ids.add(chat_id)

        # Re-emit the message on the traitement Q
        await producer.executer_commande(message.original, domaine='ollama_relai', action='traitement',
                                         exchange=ConstantesMilleGrilles.SECURITE_PROTEGE, noformat=True, nowait=True)

        return {'ok': True}

    async def process_query(self, message: MessageWrapper):
        content = message.parsed.copy()
        del content['__original']
        try:
            chat_id = content['chat_id']
            del content['chat_id']
        except KeyError:
            chat_id = message.id

        try:
            self.__waiting_ids.remove(chat_id)
        except KeyError:
            pass  # Ok, could be on another instance

        producer = self.__context.producer
        await producer.producer_pret().wait()

        try:
            action = message.routage['action']

            if action not in ['chat', 'generate']:
                # Raising an exception sends the cancel event to the client
                raise Exception('Unsupported action: %s' % action)

            self.__processing_ids.add(chat_id)

            await producer.emettre_evenement(dict(), domaine='ollama_relai', action='debutTraitement',
                                             partition=chat_id, exchanges=ConstantesMilleGrilles.SECURITE_PRIVE)

            # Run the query. Emit
            done_event = asyncio.Event()
            response_content = await self.query_ollama(action, content, done_event)
            # response_content = {'role': 'dummy', 'message': 'NANANA'}

            # if action == 'generate':
            #     print("Response: %s" % response_content['response'])
            # elif action == 'chat':
            #     print("Response: %s" % response_content['message'])

            # Send the encrypted response as result event to client
            enveloppe = message.certificat
            reponse_tuple = await producer.chiffrer([enveloppe], ConstantesMilleGrilles.KIND_REPONSE_CHIFFREE,
                                                    response_content, domaine='ollama_relai', action='resultat',
                                                    partition=chat_id)
            encrypted_response = reponse_tuple[0]
            await producer.emettre_evenement(encrypted_response, domaine='ollama_relai', action='resultat',
                                             partition=chat_id, exchanges=ConstantesMilleGrilles.SECURITE_PRIVE, noformat=True)

            # Ensure all instances of relai stop issuing events for this chat_id
            await producer.emettre_evenement({}, domaine='ollama_relai', action='termine', partition=chat_id,
                                             exchanges=ConstantesMilleGrilles.SECURITE_PRIVE)

            return None
        except Exception as e:
            await producer.emettre_evenement({}, domaine='ollama_relai', action='annuler', partition=chat_id,
                                             exchanges=ConstantesMilleGrilles.SECURITE_PRIVE)
            raise e
        finally:
            try:
                self.__processing_ids.remove(chat_id)
            except KeyError:
                pass  # Ok, could have been cancelled

    async def query_ollama(self, action: str, content: dict, done: asyncio.Event) -> dict:
        try:
            url_query = f'{self.__context.configuration.ollama_url}/api/{action}'
            response = await asyncio.to_thread(requests.post, url_query, json=content)
            return response.json()
        finally:
            done.set()
