import asyncio
import datetime
import logging

import httpx
import multibase
from ollama import AsyncClient

from typing import Optional

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.Context import OllamaRelaiContext
from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_bytes_secrete, dechiffrer_document_secrete


class QueryHandler:

    def __init__(self, context: OllamaRelaiContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__waiting_ids: dict[str, dict] = dict()  # Key = chat_id, value = {reply_to, correlation_id}
        # self.__processing_ids: set[str] = set()
        self.__stop_event: Optional[asyncio.Event] = None

    async def run(self, stop_event: asyncio.Event):
        self.__stop_event = stop_event
        tasks = [
            asyncio.create_task(self.emit_event_thread(self.__waiting_ids, 'attente')),
            # asyncio.create_task(self.emit_event_thread(self.__processing_ids, 'encours')),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if self.__stop_event.is_set() is False:
            self.__logger.warning("Threads shutting down")
            self.__stop_event.set()

    async def emit_event_thread(self, correlations: dict[str, dict], event_name: str):
        # Wait for the initialization of the producer
        try:
            await asyncio.wait_for(self.__stop_event.wait(), 5)
        except asyncio.TimeoutError:
            pass  # Ok, this is supposed to time out

        while self.__stop_event.is_set() is False:
            producer = self.__context.producer
            await producer.producer_pret().wait()

            for correlation_info in correlations.values():
                await producer.repondre({'ok': True, 'evenement': event_name},
                                        correlation_id=correlation_info['correlation_id'], reply_to=correlation_info['reply_to'],
                                        attachements={'streaming': True})

            try:
                await asyncio.wait_for(self.__stop_event.wait(), 15)
            except asyncio.TimeoutError:
                pass  # This is supposed to time out

    async def handle_requests(self, message: MessageWrapper):
        producer = self.__context.producer
        await producer.producer_pret().wait()

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

        raise Exception('Unknown request: %s' % action)


    async def handle_query(self, message: MessageWrapper):
        producer = self.__context.producer
        await producer.producer_pret().wait()

        # Confirm routing key
        rk_split = message.routing_key.split('.')
        type_message = rk_split[0]
        domaine = rk_split[1]
        action = rk_split.pop()

        if type_message != 'commande':
            raise Exception('Wrong message kind, must be command')

        if domaine != 'ollama_relai':
            raise Exception('Domaine must be ollama_relai')

        if action not in ['chat', 'generate']:
            raise Exception('Actions can only be one of: chat, generate')

        # Start emitting "waiting" events to tell the client it is still queued-up (keep-alive)
        chat_id = message.id
        self.__waiting_ids[chat_id] = {'reply_to': message.reply_to, 'correlation_id': message.correlation_id}

        # Re-emit the message on the traitement Q
        attachements = {'correlation_id': message.correlation_id, 'reply_to': message.reply_to}
        attachements.update(message.original['attachements'])
        await producer.executer_commande(message.original, domaine='ollama_relai', action='traitement',
                                         exchange=ConstantesMilleGrilles.SECURITE_PROTEGE, noformat=True, nowait=True,
                                         attachements=attachements)

        await producer.repondre(
            {'ok': True, 'partition': chat_id, 'stream': True, 'reponse': 1},
            reply_to=message.reply_to, correlation_id=message.correlation_id,
            attachements={'streaming': True}
        )

        return False  # We'll stream the messages, no automatic response here

    async def process_query(self, message: MessageWrapper):
        chat_id = message.id
        # original_message: MessageWrapper = self.__waiting_ids[chat_id]['original']

        # Extract encryption and routing information for the response
        original_message = message.original
        # enveloppe = EnveloppeCertificat.from_pem(original_message['certificat'])
        enveloppe = message.certificat
        attachments: dict = original_message['attachements']
        correlation_id = attachments['correlation_id']
        reply_to = attachments['reply_to']
        chat_history: Optional[dict] = attachments['history']
        decryption_keys = attachments['keys']
        domain_signature = attachments['signature']

        producer = self.__context.producer
        await producer.producer_pret().wait()

        try:
            # Message for other instances, indicates we're taking ownership of the processing for this request
            await producer.emettre_evenement({'evenement': 'debutTraitement'},
                                             domaine='ollama_relai', action='evenement', partition=chat_id,
                                             exchanges=ConstantesMilleGrilles.SECURITE_PROTEGE)

            # decryption_key: Optional[bytes] = None
            # chiffrage: Optional[dict] = None
            content = message.parsed.copy()
            del content['__original']

            chat_model = content['model']
            chat_role = content['role']

            # Recover the keys, send to MaitreDesCles to get the decrypted value
            dechiffrage = {'signature': domain_signature, 'cles': decryption_keys}
            decryption_key_response = await producer.executer_requete(dechiffrage, 'MaitreDesCles', 'dechiffrageMessage',
                                                                      exchange=ConstantesMilleGrilles.SECURITE_PROTEGE)

            decryption_key = decryption_key_response.parsed['cle_secrete_base64']
            decryption_key: bytes = multibase.decode('m' + decryption_key)

            chat_message = await asyncio.to_thread(dechiffrer_bytes_secrete, decryption_key, content['encrypted_content'])
            if chat_history:
                chat_messages = await asyncio.to_thread(dechiffrer_document_secrete, decryption_key, chat_history)
            else:
                chat_messages = list()

            # Add the new user message to the history of chat messages
            chat_messages.append({'role': chat_role, 'content': chat_message.decode('utf-8')})

            action = message.routage['action']

            if action not in ['chat', 'generate']:
                # Raising an exception sends the cancel event to the client
                raise Exception('Unsupported action: %s' % action)

            # Run the query. Emit
            done_event = asyncio.Event()
            # response_content = await self.query_ollama(action, content, done_event)
            if action == 'chat':
                chat_stream = self.ollama_chat(chat_messages, chat_model, done_event)
            elif action == 'generate':
                chat_messages = [{'role': 'user', 'content': content['prompt']}]
                chat_stream = self.ollama_chat(chat_messages, chat_model, done_event)
            else:
                raise Exception('action %s not supported' % action)
            # response_content = {'message': {'role': 'dummy', 'content': 'NANANA2'}}

            emit_interval = datetime.timedelta(milliseconds=750)
            next_emit = datetime.datetime.now() + emit_interval
            buffer = ''
            complete_response = []
            async for chunk in chat_stream:
                try:
                    # Stop emitting keep-alive messages
                    del self.__waiting_ids[chat_id]
                except KeyError:
                    pass  # Ok, already removed or on another instance

                attachements = None
                done = True
                try:
                    if chunk['done'] is not True:
                        attachements = {'streaming': True}
                        done = False
                except KeyError:
                    pass

                buffer += chunk['message']['content']

                now = datetime.datetime.now()
                if attachements is None or now > next_emit:
                    chunk['message']['content'] = buffer
                    complete_response.append(buffer)  # Keep for response transaction
                    buffer = ''

                    if done:
                        # Prepare a command to save the complete response, send id as part of last streaming message
                        encrypted_command, command_id = await producer.chiffrer(
                            [enveloppe], ConstantesMilleGrilles.KIND_COMMANDE_INTER_MILLEGRILLE, chunk, cle_secrete=decryption_key)
                        # Transfer keys and signature for secret key to command
                        encrypted_command['dechiffrage']['signature'] = dechiffrage['signature']
                        encrypted_command['dechiffrage']['cles'] = dechiffrage['cles']

                        chunk['message_id'] = command_id

                        attachements_echange = {
                            'query': original_message,
                        }

                        try:
                            await producer.executer_commande(encrypted_command, 'AiLanguage', 'chatExchange',
                                                             exchange=ConstantesMilleGrilles.SECURITE_PROTEGE,
                                                             attachements=attachements_echange, noformat=True, timeout=2)
                            chunk['chat_exchange_persisted'] = True
                        except asyncio.TimeoutError:
                            # Failure to save command, relay to user
                            chunk['chat_exchange_persisted'] = False

                    reponse_tuple = await producer.chiffrer([enveloppe], ConstantesMilleGrilles.KIND_REPONSE_CHIFFREE,
                                                            chunk, partition=chat_id)
                    encrypted_response = reponse_tuple[0]
                    await producer.repondre(encrypted_response,
                                            correlation_id=correlation_id, reply_to=reply_to,
                                            noformat=True, attachements=attachements)

                    next_emit = now + emit_interval

            return None
        except Exception as e:
            await producer.repondre({'ok': False, 'err': str(e)},
                                    correlation_id=correlation_id, reply_to=reply_to)
            raise e
        finally:
            try:
                del self.__waiting_ids[chat_id]
            except KeyError:
                pass  # Ok, could have been cancelled

    async def ollama_chat(self, messages: list[dict], model: str, done: asyncio.Event):
        client = AsyncClient(host=self.__context.configuration.ollama_url)
        stream = await client.chat(
            model=model,
            messages=messages,
            stream=True,
        )

        async for part in stream:
            yield part

    async def ollama_ping(self) -> (bool, str):
        client = AsyncClient(host=self.__context.configuration.ollama_url)
        try:
            # Test connection by getting currently loaded model information
            await client.ps()
            return True, ''
        except httpx.ConnectError:
            # Failed to connect
            pass

        return False, 'ollama connection error'
