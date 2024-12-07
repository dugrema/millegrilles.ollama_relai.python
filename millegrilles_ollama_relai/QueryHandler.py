import asyncio
import datetime
import logging
from asyncio import TaskGroup

import httpx
import multibase

from ollama import AsyncClient

from typing import Optional, Union

from millegrilles_messages.chiffrage.Mgs4 import chiffrer_mgs4_bytes_secrete
from millegrilles_messages.chiffrage.SignatureDomaines import SignatureDomaines
from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_bytes_secrete, dechiffrer_document_secrete
from millegrilles_ollama_relai.OllamaContext import OllamaContext


class QueryHandler:

    def __init__(self, context: OllamaContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__waiting_ids: dict[str, dict] = dict()  # Key = chat_id, value = {reply_to, correlation_id}

        self.__ollama_status: Union[bool, dict] = False

    async def run(self):
        async with TaskGroup() as group:
            group.create_task(self.emit_event_thread(self.__waiting_ids, 'attente'))
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

    async def emit_event_thread(self, correlations: dict[str, dict], event_name: str):
        # Wait for the initialization of the producer
        await self.__context.wait(5)

        while self.__context.stopping is False:
            producer = await self.__context.get_producer()

            for correlation_info in correlations.values():
                await producer.reply({'ok': True, 'evenement': event_name},
                                     correlation_id=correlation_info['correlation_id'], reply_to=correlation_info['reply_to'],
                                     attachments={'streaming': True})

            await self.__context.wait(15)

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

        if action not in ['chat', 'generate']:
            raise Exception('Actions can only be one of: chat, generate')

        # Start emitting "waiting" events to tell the client it is still queued-up (keep-alive)
        chat_id = message.id
        self.__waiting_ids[chat_id] = {'reply_to': message.reply_to, 'correlation_id': message.correlation_id}

        # Re-emit the message on the traitement Q
        attachements = {'correlation_id': message.correlation_id, 'reply_to': message.reply_to}
        attachements.update(message.original['attachements'])
        await producer.command(
            message.original, domain='ollama_relai', action='traitement', exchange=ConstantesMilleGrilles.SECURITE_PROTEGE,
            noformat=True, nowait=True, attachments=attachements)

        await producer.reply(
            {'ok': True, 'partition': chat_id, 'stream': True, 'reponse': 1},
            reply_to=message.reply_to, correlation_id=message.correlation_id,
            attachments={'streaming': True}
        )

        return False  # We'll stream the messages, no automatic response here

    async def process_query(self, message: MessageWrapper):
        chat_id = message.id
        # original_message: MessageWrapper = self.__waiting_ids[chat_id]['original']

        # Extract encryption and routing information for the response
        original_message = message.original
        # enveloppe = EnveloppeCertificat.from_pem(original_message['certificat'])
        enveloppe = message.certificat
        user_id = enveloppe.get_user_id
        attachments: dict = original_message['attachements']
        correlation_id = attachments['correlation_id']
        reply_to = attachments['reply_to']
        chat_history: Optional[dict] = attachments['history']
        decryption_keys = attachments['keys']
        domain_signature = attachments['signature']

        producer = await self.__context.get_producer()

        try:
            # Message for other instances, indicates we're taking ownership of the processing for this request
            await producer.event(
                {'evenement': 'debutTraitement'},
                domain='ollama_relai', action='evenement', partition=chat_id, exchange=ConstantesMilleGrilles.SECURITE_PROTEGE)

            # decryption_key: Optional[bytes] = None
            # chiffrage: Optional[dict] = None
            content = message.parsed.copy()
            del content['__original']

            chat_model = content['model']
            chat_role = content['role']
            new_conversation = content.get('new') or False
            conversation_id = content['conversation_id']

            # Recover the keys, send to MaitreDesCles to get the decrypted value
            dechiffrage = {'signature': domain_signature, 'cles': decryption_keys}
            try:
                decryption_key_response = await producer.request(
                    dechiffrage, 'MaitreDesCles', 'dechiffrageMessage',
                    exchange=ConstantesMilleGrilles.SECURITE_PROTEGE, timeout=3)
            except asyncio.TimeoutError as e:
                self.__logger.error("Error getting conversation decryption key for AI chat message")
                await producer.reply({'ok': False, 'err': 'Timeout getting decryption key'},
                                     correlation_id=correlation_id, reply_to=reply_to)
                return False

            cle_id = content['encrypted_content']['cle_id']
            try:
                decryption_key = decryption_key_response.parsed['cle_secrete_base64']
                decryption_key: bytes = multibase.decode('m' + decryption_key)
            except KeyError as e:
                self.__logger.error("Error receing key cle_id:%s\nRequest: %s\nResponse: %s" % (cle_id, dechiffrage, decryption_key_response))
                await producer.reply({'ok': False, 'err': 'Error decrypting key'},
                                     correlation_id=correlation_id, reply_to=reply_to)
                return False

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

                attachments = None
                done = True
                try:
                    if chunk['done'] is not True:
                        attachments = {'streaming': True}
                        done = False
                except KeyError:
                    pass

                buffer += chunk['message']['content']

                now = datetime.datetime.now()
                if attachments is None or now > next_emit:
                    chunk['message']['content'] = buffer
                    complete_response.append(buffer)  # Keep for response transaction
                    buffer = ''

                    if done:
                        # Join entire response in single string
                        complete_response = ''.join(complete_response)
                        domaine_signature_obj = SignatureDomaines.from_dict(domain_signature)
                        # cle_id = domaine_signature_obj.get_cle_ref()

                        # Encrypt
                        cipher, encrypted_response = chiffrer_mgs4_bytes_secrete(decryption_key, complete_response)
                        encrypted_response['cle_id'] = cle_id

                        reply_command = {
                            'conversation_id': conversation_id,
                            'user_id': user_id,
                            'encrypted_content': encrypted_response,
                            'new': new_conversation,
                            'model': chunk['model'],
                            'role': 'assistant',
                        }
                        del original_message['attachements']
                        attachements_echange = {'query': original_message}
                        attachements_echange.update(dechiffrage)

                        signed_command, command_id = self.__context.formatteur.signer_message(
                            ConstantesMilleGrilles.KIND_COMMANDE, reply_command, 'AiLanguage', action='chatExchange')
                        chunk['message_id'] = command_id

                        try:
                            await producer.command(signed_command, 'AiLanguage', 'chatExchange',
                                                   exchange=ConstantesMilleGrilles.SECURITE_PROTEGE,
                                                   attachments=attachements_echange, noformat=True, timeout=2)

                            chunk['chat_exchange_persisted'] = True
                        except asyncio.TimeoutError:
                            # Failure to save command, relay to user
                            chunk['chat_exchange_persisted'] = False
                            chunk['exchange_command'] = signed_command

                    await producer.encrypt_reply([enveloppe], chunk, correlation_id=correlation_id, reply_to=reply_to,
                                                 attachments=attachments)

                    next_emit = now + emit_interval

            return None
        except Exception as e:
            self.__logger.exception("Unhandled error during chat")
            await producer.reply({'ok': False, 'err': str(e)},
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

    async def check_ollama_status(self) -> Union[bool, dict]:
        client = AsyncClient(host=self.__context.configuration.ollama_url)
        try:
            # Test connection by getting currently loaded model information
            status = await client.ps()
            return dict(status)
        except httpx.ConnectError:
            # Failed to connect
            return False

    async def check_ollama_list_models(self) -> Union[bool, list]:
        client = AsyncClient(host=self.__context.configuration.ollama_url)
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
