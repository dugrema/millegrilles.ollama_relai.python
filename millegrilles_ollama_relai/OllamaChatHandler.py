import asyncio
import datetime
import logging
from asyncio import TaskGroup

import multibase
import tempfile
import subprocess
import os

from typing import Any, Optional, Coroutine

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_messages.chiffrage.Mgs4 import chiffrer_mgs4_bytes_secrete
from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_bytes_secrete, dechiffrer_document_secrete
from millegrilles_ollama_relai.OllamaTools import OllamaToolHandler


MAX_TOOL_ITERATIONS = 3

class OllamaChatHandler:

    def __init__(self, context: OllamaContext, attachment_handler: AttachmentHandler, tool_handler: OllamaToolHandler):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__attachment_handler = attachment_handler
        self.__tool_handler = tool_handler
        self.__waiting_ids: dict[str, dict] = dict()  # Key = chat_id, value = {reply_to, correlation_id}
        self.__running_chats: dict[str, asyncio.Task] = dict()
        self.__cancelled_chats = set()

    async def run(self):
        async with TaskGroup() as group:
            group.create_task(self.emit_event_thread(self.__waiting_ids, 'attente'))

    async def __maintenance(self):
        while self.__context.stopping is False:
            self.__cancelled_chats.clear()  # Clear all cancelled chats (crude, may clear while chat is in queue)
            await self.__context.wait(900)

    async def register_query(self, message: MessageWrapper):
        producer = await self.__context.get_producer()

        # Start emitting "waiting" events to tell the client it is still queued-up (keep-alive)
        chat_id = message.id
        self.__waiting_ids[chat_id] = {'reply_to': message.reply_to, 'correlation_id': message.correlation_id}

        # Re-emit the message on the traitement Q
        attachements = {'correlation_id': message.correlation_id, 'reply_to': message.reply_to}
        attachements.update(message.original['attachements'])
        await producer.command(
            message.original, domain='ollama_relai', action='traitement',
            exchange=ConstantesMilleGrilles.SECURITE_PROTEGE,
            noformat=True, nowait=True, attachments=attachements)

        await producer.reply(
            {'ok': True, 'partition': chat_id, 'stream': True, 'reponse': 1},
            reply_to=message.reply_to, correlation_id=message.correlation_id,
            attachments={'streaming': True}
        )

        return False  # We'll stream the messages, no automatic response here

    async def cancel_chat(self, message: MessageWrapper):
        chat_id = message.parsed['chat_id']
        self.__logger.info(f"Cancelling chat_id {chat_id}")
        # The chat is not in the waiting queue, try to cancel its task
        try:
            self.__running_chats[chat_id].cancel()
            return {'ok': True}  # Cancelled
        except KeyError:
            # No task running that corresponds to the chat_id
            # Assume it is in the processing Q and any ollama_relai may pick it up
            self.__cancelled_chats.add(chat_id)
            try:
                # Stop sending events on that chat if it is waiting
                del self.__waiting_ids[chat_id]
            except KeyError:
                pass

        return {'ok': True, 'message': 'Will be cancelled'}

    async def process_chat(self, message: MessageWrapper):
        chat_id = message.id
        if chat_id in self.__cancelled_chats:
            # Chat has been cancelled, skip it
            self.__cancelled_chats.remove(chat_id)
            return False

        # original_message: MessageWrapper = self.__waiting_ids[chat_id]['original']

        max_context_length = 512000  # Avoid sending huge documents

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
        attached_files = message.parsed.get('attachments')

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
            user_profile = content.get('user_profile') or dict()

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
                self.__logger.error("Error receving key cle_id:%s\nRequest: %s\nResponse: %s" % (cle_id, dechiffrage, decryption_key_response))
                await producer.reply({'ok': False, 'err': 'Error decrypting key'},
                                     correlation_id=correlation_id, reply_to=reply_to)
                return False

            chat_message = await asyncio.to_thread(dechiffrer_bytes_secrete, decryption_key, content['encrypted_content'])
            if chat_history:
                encrypted_content = await asyncio.to_thread(dechiffrer_document_secrete, decryption_key, chat_history)
                chat_messages = encrypted_content.get('messageHistory') or list()
                attachment_keys = encrypted_content.get('attachmentKeys')
            else:
                chat_messages = list()
                attachment_keys = None

            current_message_content = chat_message.decode('utf-8')
            attached_tuuids: Optional[list[str]] = None

            image_attachments = None
            if attached_files is not None and attachment_keys is not None:
                # Process and download files
                for attached_file in attached_files:

                    # Keep track of attached files for conversation
                    if attached_tuuids is None:
                        attached_tuuids = list()
                    attached_tuuids.append(attached_file['tuuid'])

                    mimetype: str = attached_file['mimetype']
                    key_id = attached_file['keyId']
                    decryption_key_attached_file: str = attachment_keys[key_id]
                    with tempfile.TemporaryFile('wb+') as tmp_file:
                        if mimetype == 'application/pdf':
                            await self.__attachment_handler.download_decrypt_file(decryption_key_attached_file,
                                                                                  attached_file, tmp_file)
                            text_content = await extract_pdf_content(tmp_file)
                            if text_content is not None:
                                if len(current_message_content) > max_context_length:
                                    # Truncate to avoid exceeding context
                                    current_message_content += "\nPDF file (truncated):\n\n" + text_content
                                    current_message_content = current_message_content[0:max_context_length]
                                else:
                                    current_message_content += "\nPDF file full content:\n\n" + text_content

                        elif mimetype.startswith('image/'):
                            file_size = await self.__attachment_handler.download_decrypt_file(decryption_key_attached_file,
                                                                                              attached_file, tmp_file)
                            if image_attachments is None:
                                image_attachments = list()

                            content = await self.__attachment_handler.prepare_image(file_size, tmp_file)
                            image_attachments.append(content)
                        else:
                            raise NotImplementedError('File type not handled')
                pass

            # Add the new user message to the history of chat messages
            current_chat_message = {'role': chat_role, 'content': current_message_content}
            if image_attachments is not None:
                current_chat_message['images'] = image_attachments
            chat_messages.append(current_chat_message)

            action = message.routage['action']

            # Run the query. Emit
            done_event = asyncio.Event()
            if action == 'chat':
                chat_stream = self.ollama_chat(user_profile, chat_messages, chat_model, done_event)
            else:
                raise Exception('action %s not supported' % action)

            # Do another check in case the chat was cancelled.
            if chat_id in self.__cancelled_chats:
                # Chat has been cancelled, skip it
                self.__cancelled_chats.remove(chat_id)
                return False

            # Create the stream task
            output = dict()
            stream_coro = self.stream_chat_response(chat_id, output, enveloppe, correlation_id, reply_to, chat_stream)
            stream_task = asyncio.create_task(stream_coro)

            # Save task to allow cancellation, then wait on completion
            self.__running_chats[chat_id] = stream_task

            try:
                await stream_task
            except* asyncio.CancelledError as e:
                # Response interrupted by user, save
                if self.__context.stopping is True:
                    raise e  # Stopping
            finally:
                # Cleanup task
                try:
                    del self.__running_chats[chat_id]
                except KeyError:
                    pass

            # Join entire response in single string
            complete_response = ''.join(output['response'])

            # Check if the output was completed or interrupted
            complete = output['complete']
            if complete is False:
                complete_response += '\n**INTERRUPTED BY USER**'

            # Encrypt
            cipher, encrypted_response = chiffrer_mgs4_bytes_secrete(decryption_key, complete_response)
            encrypted_response['cle_id'] = cle_id

            reply_command = {
                'conversation_id': conversation_id,
                'user_id': user_id,
                'encrypted_content': encrypted_response,
                'new': new_conversation,
                'model': chat_model,
                'role': 'assistant',
            }
            if attached_tuuids is not None:
                reply_command['tuuids'] = attached_tuuids
            attachements_echange = {'query': original_message}
            attachements_echange.update(dechiffrage)

            signed_command, command_id = self.__context.formatteur.signer_message(
                ConstantesMilleGrilles.KIND_COMMANDE, reply_command, 'AiLanguage', action='chatExchange')

            await producer.command(signed_command, 'AiLanguage', 'chatExchange',
                                   exchange=ConstantesMilleGrilles.SECURITE_PROTEGE,
                                   attachments=attachements_echange, noformat=True, timeout=2)

            # Confirm to streaming client that message is complete
            chunk = {
                'message': {'content': ''},
                'done': True,
                'message_id': command_id
            }
            await producer.encrypt_reply([enveloppe], chunk, correlation_id=correlation_id, reply_to=reply_to, attachments={'stream': False})

        except Exception as e:
            self.__logger.exception("Unhandled error during chat")
            await producer.reply({'ok': False, 'err': str(e)},
                                 correlation_id=correlation_id, reply_to=reply_to)
            raise e
        finally:
            try:
                del self.__running_chats[chat_id]
            except KeyError:
                pass
            try:
                del self.__waiting_ids[chat_id]
            except KeyError:
                pass  # Ok, could have been cancelled

    async def stream_chat_response(self, chat_id: str, output: dict, enveloppe, correlation_id, reply_to, chat_stream):
        producer = await self.__context.get_producer()
        emit_interval = datetime.timedelta(milliseconds=750)
        next_emit = datetime.datetime.now() + emit_interval
        buffer = ''
        complete_response = []
        output['response'] = complete_response
        output['complete'] = False
        async for chunk in chat_stream:
            chunk = dict(chunk)
            chunk['message'] = dict(chunk['message'])

            try:
                # Stop emitting keep-alive messages
                del self.__waiting_ids[chat_id]
            except KeyError:
                pass  # Ok, already removed or on another instance

            done = True
            try:
                if chunk['done'] is not True:
                    done = False
            except KeyError:
                pass

            output['complete'] = done

            buffer += chunk['message']['content']

            now = datetime.datetime.now()
            if done or now > next_emit:
                chunk['message']['content'] = buffer
                complete_response.append(buffer)  # Keep for response transaction
                buffer = ''

                if done:
                    chunk['done'] = False  # Override flag, will bet set after save command

                await producer.encrypt_reply([enveloppe], chunk, correlation_id=correlation_id, reply_to=reply_to, attachments={'streaming': True})

                next_emit = now + emit_interval

        return None

    async def ollama_chat(self, user_profile: dict, messages: list[dict], model: str, done: asyncio.Event):
        instance = self.__context.pick_ollama_instance(model)
        client = instance.get_async_client(self.__context.configuration, timeout=180)
        context_len = self.__context.chat_configuration.get('chat_context_length') or 4096

        # Check if the model supports tools
        try:
            model_info = [m for m in self.__context.ollama_models if m['name'] == model][0]
            if 'tools' in model_info['capabilities']:
                tools = self.__tool_handler.tools()
            else:
                tools = None
        except (TypeError, AttributeError, IndexError, KeyError):
            tools = None

        async with instance.semaphore:
            # Loop to allow tool calls by the model.
            for i in range(0, MAX_TOOL_ITERATIONS):
                if i == MAX_TOOL_ITERATIONS - 1:
                    # Last iteration. Prevent model from making any further calls to tools.
                    self.__logger.warning(f"Stopping excessive tool usage by model {model}")
                    tools = None

                # Chat request to ollama
                stream = await client.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    options={"num_ctx": context_len}
                )

                # Keep the streaming output for tool calls
                cumulative_output = ''
                tools_called = False

                async for part in stream:
                    if part.message.tool_calls:
                        # Tools are being invoked. Keep history of output for assistant.
                        if len(cumulative_output) > 0:
                            messages.append({'role': 'assistant', 'content': cumulative_output})

                        # Add message from assistant with calls to tools
                        messages.append(part.message)

                        # Reset output
                        cumulative_output = ''

                        if len(cumulative_output) > 0:
                            messages.append({'role': 'assistant', 'content': cumulative_output})

                        self.__logger.debug("Calling tools: %s" % part.message.tool_calls)
                        for tool_call in part.message.tool_calls:
                            output = await self.__tool_handler.run_tool(user_profile, tool_call)
                            messages.append({'role': 'tool', 'content': str(output), 'name': tool_call.function.name})
                            tools_called = True

                    cumulative_output += part.message.content
                    if part.message.tool_calls or (part.done and tools_called):
                        pass  # Avoid yielding part with tool calls, need to handle here first
                    else:
                        yield part

                # If no tools were called, the chat is done. If we have tool responses, loop for a new iteration.
                if tools_called is False:
                    break  # Done

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


async def extract_pdf_content(tmp_file: tempfile.TemporaryFile) -> Optional[str]:

    # Create named output file to use with pdftotext - todo, check how to use asyncio.subprocess and pipes
    with tempfile.NamedTemporaryFile('wb') as pdf_in:
        output_text_name = pdf_in.name + '.txt'
        tmp_file.seek(0)
        await asyncio.to_thread(pdf_in.write, tmp_file.read())
        await asyncio.to_thread(pdf_in.flush)
        await asyncio.to_thread(subprocess.run, ['/usr/bin/pdftotext', pdf_in.name, output_text_name])

    # Read the output text into variable
    with open(output_text_name, 'r') as file_in:
        content = await asyncio.to_thread(file_in.read)

    # Cleanup text file
    await asyncio.to_thread(os.unlink, output_text_name)

    return content
