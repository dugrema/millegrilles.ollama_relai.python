import asyncio
import datetime
import logging
import json
import nacl
import tempfile
import httpx
import openai
import pydantic
from typing import Optional

from cryptography.x509 import ExtensionNotFound
from multibase import multibase
from openai import AsyncClient as OpenaiAsyncClient

from millegrilles_messages.Filehost import FilehostConnection
from millegrilles_messages.bus.PikaChannel import MilleGrillesPikaChannel
from millegrilles_messages.bus.PikaQueue import MilleGrillesPikaQueueConsumer, RoutingKey
from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_document, dechiffrer_bytes_secrete
from millegrilles_messages.chiffrage.Mgs4 import chiffrer_document_cles, chiffrer_mgs4_bytes_secrete
from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.DocumentIndexHandler import FileInformation, format_text_prompt
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.Structs import get_property_int, get_property_text, get_property_float, SummaryText
from millegrilles_ollama_relai.Util import conditional_convert_to_png, decode_base64_nopad, cleanup_json_output, \
    encode_image_to_data_uri

CONST_ACTION_SUMMARY = 'leaseForSummary'
CONST_JOB_SUMMARY_TEXT = 'summaryText'
CONST_JOB_SUMMARY_IMAGE = 'summaryImage'
CONST_CHAR_MULTIPLIER = 2.5
CONST_SUMMARY_NUM_PREDICT = 6144

class Processor:

    def __init__(self, context: OllamaContext, attachment_handler: FilehostConnection, task_name: str, task_properties: dict):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context: OllamaContext = context
        self.__attachment_handler: FilehostConnection = attachment_handler
        self.__task_name: str = task_name
        self.__task_properties: dict = task_properties
        self.__params = task_properties['params']
        self.__channels: Optional[list[MilleGrillesPikaChannel]] = None
        self.__stop_event = asyncio.Event()
        self.__fetch_event = asyncio.Event()
        self.__fetch_batch_size = 20
        self.__api_url: Optional[str] = None
        self.__context_length = 16384
        self.__temperature = 1.0
        self.__supports_vision = False
        self.__model = 'NONAME'
        self.__document_prompt = None
        self.__image_prompt = None

    @property
    def routing_action_work(self):
        return f'task_{self.__task_name}_work'

    async def setup(self):
        self.__logger.info(f"Setting up DocumentIndexing task {self.__task_name} with params {self.__params}...")
        await self.__set_up_mq()

        self.__fetch_batch_size = get_property_int(self.__params, 'batchsize') or 20
        self.__api_url = get_property_text(self.__params, 'url')
        if self.__api_url is None:
            raise Exception("API URL not configured")

        self.__context_length = get_property_int(self.__params, 'context')
        self.__temperature = get_property_float(self.__params, 'temperature') or 1.0
        self.__supports_vision = get_property_int(self.__params, 'vision') == 1
        self.__model = get_property_text(self.__params, 'model')
        self.__document_prompt = get_property_text(self.__params, 'prompt_documents')
        self.__image_prompt = get_property_text(self.__params, 'prompt_images')

    async def __stop_thread(self):
        await self.__context.wait()
        self.__stop_event.set()
        # Release all threads
        self.__fetch_event.set()

    async def run(self):
        # Start all tasks
        self.__logger.debug(f"Running task processor {self.__task_name}")
        asyncio.create_task(self.__stop_thread())   # Sets the stop event

        # Work threads
        asyncio.create_task(self.__job_fetch_thread())

        while not self.__stop_event.is_set():
            self.__logger.debug(f"Still running task {self.__task_name}")
            try:
                self.__fetch_event.set()    # Fetches new jobs
                await asyncio.wait_for(self.__stop_event.wait(), 30)
            except asyncio.TimeoutError:
                pass

        # Stop processing
        if self.__channels:
            for channel in self.__channels:
                await channel.stop_consuming()
                await self.__context.bus_connector.remove_channel(channel)
            self.__channels = None

    async def stop(self):
        self.__logger.debug(f"Stopping task processor {self.__task_name}")
        self.__stop_event.set()

    async def __job_fetch_thread(self):
        while not self.__stop_event.is_set():
            # Potential flood of new fuuids, wait a few seconds before starting to process
            await self.__context.wait(5)
            if self.__stop_event.is_set():
                return  # Stopping
            self.__fetch_event.clear()  # Reset flag

            self.__logger.debug(f"Task {self.__task_name} fetching new jobs")
            try:
                await self.__query_batch()
            except:
                self.__logger.exception("Error fetching batch of documents")

            await self.__fetch_event.wait()

    async def __set_up_mq(self):
        # Initialize queue that listens to external events
        channels = list()
        worker_count = get_property_int(self.__params, 'workers') or 1
        self.__logger.debug(f"DocumentIndexing task {self.__task_name} worker count: {worker_count}")

        # Set-up the new fuuid listener, triggers job queries
        q_instance = MilleGrillesPikaQueueConsumer(
            self.__context,
            self.__on_newfuuid_event,
            f'ollama_relai/task_{self.__task_name}_triggers',
            auto_delete=True,
            arguments={'x-message-ttl': 5_000}
        )
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PUBLIC, 'evenement.filecontroler.filehostNewFuuid'))
        q_channel = MilleGrillesPikaChannel(self.__context, prefetch_count=1)
        q_channel.add_queue(q_instance)
        channels.append(q_channel)
        await self.__context.bus_connector.add_channel(q_channel)
        await q_channel.start_consuming()

        # Set-up worker queues
        for _ in range(worker_count):
            q_instance = MilleGrillesPikaQueueConsumer(
                self.__context,
                self.__on_work,
                f'ollama_relai/task_{self.__task_name}_workers',
                auto_delete=True,
                arguments={'x-message-ttl': 120_000}
            )

            q_instance.add_routing_key(RoutingKey(
                Constantes.SECURITE_PRIVE,
                f'commande.ollama_relai.{self.routing_action_work}'
            ))

            q_channel = MilleGrillesPikaChannel(self.__context, prefetch_count=1)
            q_channel.add_queue(q_instance)
            channels.append(q_channel)
            await self.__context.bus_connector.add_channel(q_channel)
            await q_channel.start_consuming()
        self.__channels = channels

    async def __on_newfuuid_event(self, message: MessageWrapper):
        # Authorization check
        enveloppe = message.certificat
        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()

        message_type = message.routing_key.split('.')[0]
        domain = message.routage['domaine']
        action = message.routage['action']
        estampille = message.estampille

        # Fuuid event messages expire after 10 seconds (or 5 seconds on queue)
        expired_timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=10)).timestamp()
        if estampille < expired_timestamp:
            return None  # Ignore

        if message_type == 'evenement':
            if domain == 'filecontroler' and action == 'filehostNewFuuid' and 'filecontroler' in roles:
                self.__logger.debug(f"Task {self.__task_name} new fuuid received: {message.parsed.get('fuuid')}")

                # Trigger a new job fetch from GrosFichiers
                self.__fetch_event.set()

        return None

    async def __on_work(self, message: MessageWrapper):
        # Authorization check
        enveloppe = message.certificat
        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()
        try:
            domains_env = enveloppe.get_domaines
        except ExtensionNotFound:
            domains_env = None

        message_type = message.routing_key.split('.')[0]
        domain = message.routage['domaine']
        action = message.routage['action']
        estampille = message.estampille

        # Volatile messages expire after 90 seconds
        expired_timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=90)).timestamp()
        if estampille < expired_timestamp:
            return None  # Ignore

        if message_type == 'commande':
            if action == self.routing_action_work and 'ollama_relai' in roles:
                self.__logger.debug(f"Task {self.__task_name} new work received")
                try:
                    await self.__process_work_item(message.parsed)
                except:
                    self.__logger.exception("Error processing work item")
                return None
            else:
                self.__logger.debug(f"Task {self.__task_name} unhandled action received: {action}")
        else:
            self.__logger.debug(f"Task {self.__task_name} unhandled message type received: {message_type}")

        return None

    async def __query_batch(self):
        producer = await self.__context.get_producer()

        try:
            filehost_id = self.__context.filehost.filehost_id
        except AttributeError:
            # Filehost not loaded yet, wait and retry
            await self.__context.wait(5)
            filehost_id = self.__context.filehost.filehost_id

        if not filehost_id:
            self.__logger.warning(f"No filehost id provided, skipping document indexing")
            return

        # Get the batch
        command = {"batch_size": self.__fetch_batch_size, "filehost_id": filehost_id}
        try:
            response = await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, CONST_ACTION_SUMMARY,
                                              Constantes.SECURITE_PROTEGE, timeout=60)
        except asyncio.TimeoutError:
            self.__logger.warning(f"Timeout on {CONST_ACTION_SUMMARY}, will retry")
            return

        parsed = response.parsed
        if parsed['ok'] is not True:
            self.__logger.warning("Error retrieving batch of files: %s" % parsed)
        elif parsed.get('code') == 1:
            self.__logger.debug(f"No more files to process for {CONST_ACTION_SUMMARY} on filehost_id {filehost_id}")
            return

        # Parse batch file and insert on queue per item
        leases = parsed['leases']

        self.__logger.debug(f"Received batch of {len(leases)} files for {CONST_ACTION_SUMMARY} from filehost_id:{filehost_id}")

        for lease in leases:

            # Put encrypted job on work queue
            try:
                encrypted_job = self.__prepare_job(parsed, lease)
            except JobPreparationException as e:
                if e.tuuid and e.fuuid:
                    await self.__cancel_job(e.tuuid, e.fuuid)
                else:
                    self.__logger.error("Preparation exception on job and unable to get tuuid/fuuid to cancel: %s" % e)
                continue
            except asyncio.TimeoutError:
                self.__logger.warning(f"Timeout on submitting job, will retry")
                return

            await producer.command(encrypted_job, 'ollama_relai', self.routing_action_work, Constantes.SECURITE_PRIVE, nowait=True)

        pass

    def __prepare_job(self, message, lease) -> dict:
        secret_keys: list = message['secret_keys']

        metadata = lease.get('metadata')
        version = lease.get('version')
        cuuids = lease.get('cuuids')
        tuuid: Optional[str] = lease.get('tuuid')
        fuuid: Optional[str] = None

        cle_id = None
        mimetype = lease.get('mimetype')
        if version:
            fuuid = version['fuuid']
            cle_id = version.get('cle_id')
            mimetype = mimetype or version.get('mimetype')

        if not fuuid:
            raise Exception("Unandled job type - no fuuid")

        if metadata and not cle_id:
            cle_id = cle_id or metadata.get('cle_id') or metadata.get('ref_hachage_bytes')
        cle_id = cle_id or fuuid

        try:
            key = [k for k in secret_keys if k['cle_id'] == cle_id].pop()
        except IndexError:
            self.__logger.warning(f"Missing key for fuuid {fuuid}, canceling")
            raise JobPreparationException(tuuid, fuuid)

        image_file = None
        media = lease.get('media')
        if mimetype == 'application/pdf' or mimetype.startswith('text/'):
            job_type = CONST_JOB_SUMMARY_TEXT
        elif mimetype.startswith('image/'):
            job_type = CONST_JOB_SUMMARY_IMAGE
        else:
            self.__logger.info(f"Unsupported mimetype {mimetype} for fuuid {fuuid}, canceling")
            raise JobPreparationException(tuuid, fuuid)

        try:
            # Check to find a webp fuuid, will be smaller and easier to digest
            images = media['images']
            webp_img = [m for m in images.keys() if m.startswith('image/webp')].pop()
            image_file = images[webp_img]
            image_file['fuuid'] = image_file['hachage']
        except (AttributeError, IndexError, TypeError):
            pass

        self.__logger.debug("Lease:\n%s", lease)
        secret_key: bytes = multibase.decode('m' + key['cle_secrete_base64'])
        decrypted_metadata = json.loads(dechiffrer_bytes_secrete(secret_key, metadata))

        info: FileInformation = {
            'job_type': job_type,
            'lease_action': CONST_ACTION_SUMMARY,
            'tuuid': lease.get('tuuid'),
            'fuuid': fuuid,
            'user_id': lease['user_id'],
            'language': 'en_US',
            'domain': Constantes.DOMAINE_GROS_FICHIERS,
            'cuuids': cuuids,
            'metadata': decrypted_metadata,
            'mimetype': lease.get('mimetype'),
            'version': lease.get('version'),
            'key': key,
            'tmp_file': None,
            'image_tmp_file': None,
            'media': lease.get('media'),
            'image_file': image_file,
        }

        # Encrypt the contents to put in a queue
        # TODO - figure out a way to encrypt for all ollama_relai instances
        certs = [self.__context.signing_key.enveloppe]
        encrypted_job = chiffrer_document_cles(certs, info)

        return encrypted_job

    async def __process_work_item(self, job: dict):
        self.__logger.debug("Processing work item\n%s", job)
        signing_key = self.__context.signing_key
        fingerprint = signing_key.fingerprint
        try:
            encrypted_key = job['cles'][fingerprint]
        except KeyError:
            self.__logger.error("No keys available to decrypt job, skipping")
            return

        # Decrypt key
        decrypted_job: FileInformation = dechiffrer_document(signing_key, encrypted_key, job)
        self.__logger.debug("Decrypted job\n%s" % decrypted_job)

        fuuid = decrypted_job['fuuid']
        secret_key_str = decrypted_job['key']['cle_secrete_base64']
        # filename = decrypted_job['metadata']['nom']

        file_to_download, image_file_to_download = await self.__get_files_to_download(decrypted_job)

        try:
            with tempfile.NamedTemporaryFile(mode='wb+') as tmp_file:
                if file_to_download:
                    await self.__download_file(fuuid, secret_key_str, file_to_download, tmp_file)
                    tmp_file.seek(0)    # Reposition for reading open handle
                    decrypted_job['tmp_file'] = tmp_file

                if image_file_to_download:
                    with tempfile.NamedTemporaryFile(mode='wb+') as tmp_img_file:
                        await self.__download_file(fuuid, secret_key_str, image_file_to_download, tmp_img_file)
                        tmp_img_file.seek(0)    # Reposition for reading open handle
                        decrypted_job['image_tmp_file'] = tmp_img_file

                        # Pre-process decrypted file image
                        try:
                            image_mimetype = job['image_file']['mimetype']
                        except (AttributeError, KeyError):
                            image_mimetype = 'image/webp'
                        await conditional_convert_to_png(image_mimetype, tmp_img_file)
                        tmp_img_file.seek(0)

                        summary = await self.__run_summarize_file(decrypted_job, tmp_file, tmp_img_file)
                else:
                    summary = await self.__run_summarize_file(decrypted_job, tmp_file)

                self.__logger.debug("Summary:\n%s", summary)

        except nacl.exceptions.RuntimeError as e:
            tuuid = job.get('tuuid')
            if tuuid:
                self.__logger.error(
                    f"Error decrypting fuuid {fuuid}, params nonce:{image_file_to_download} CANCELLING: {e}")
                await self.__cancel_job('tuuid', job['fuuid'])
            else:
                self.__logger.error(
                    f"Error decrypting fuuid {fuuid}, params nonce:{image_file_to_download}, missing tuuid, unable to cancel: {e}")
        except:
            self.__logger.exception("Error downloading files")

    async def __get_files_to_download(self, job: FileInformation) -> (Optional[dict], Optional[dict]):
        file_to_download = None
        image_file_to_download = job.get('image_file')
        key = job['key']
        version = job['version']

        # Special case - a PDF file will have both image and original file
        mimetype = job['mimetype']
        override_get_original = mimetype in ['application/pdf']

        # Combine version and key to ensure legacy decryption info is available
        if image_file_to_download is None or override_get_original:
            # This is a standard file, use legacy key fallback approach
            file_to_download = version.copy()
            nonce = file_to_download.get('nonce') or key.get('nonce')
            if nonce is None:
                header = file_to_download.get('header') or key.get('header')
                nonce = header[1:]

            # Override the nonce to ensure the proper value is used
            file_to_download['nonce'] = nonce

            # info_decryption.update(job['key'])
            file_to_download['format'] = file_to_download.get('format') or key.get(
                'format') or 'mgs4'  # Default format
        else:
            # This is an attached/generated file, e.g. media
            try:
                nonce = image_file_to_download['nonce']
            except KeyError:
                nonce = image_file_to_download['header'][1:]

            # Override the nonce to ensure the proper value is used
            image_file_to_download['nonce'] = nonce

            # info_decryption.update(job['key'])
            image_file_to_download['format'] = image_file_to_download.get('format') or key.get(
                'format') or 'mgs4'  # Default format

        return file_to_download, image_file_to_download

    async def __download_file(self, fuuid: str, secret_key_str: str, file_to_download: dict, tmp_file: tempfile.TemporaryFile) -> int:
        # For media encoded thumbnails/images, need to stick to file_to_download
        try:
            filesize = await self.__attachment_handler.download_decrypt_file(
                secret_key_str, file_to_download, tmp_file)
            self.__logger.debug(f"Downloaded {filesize} bytes for file {fuuid}")
            return filesize
        except* asyncio.CancelledError:
            raise Exception(f"Error downloading fuuid {fuuid}, will retry")

    async def __cancel_job(self, tuuid: str, fuuid: str):
        self.__logger.debug(f"Canceling job on fuuid {fuuid}")
        command = {'tuuid': tuuid, 'fuuid': fuuid}
        producer = await self.__context.get_producer()
        await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, "fileSummary",
                               Constantes.SECURITE_PROTEGE, timeout=45)

    async def __run_summarize_file(
            self,
            job: FileInformation,
            tmp_file: Optional[tempfile.NamedTemporaryFile],
            image_tmp_file: Optional[tempfile.NamedTemporaryFile()] = None
    ) -> SummaryText:
        # Make sure to get the encryption information first to send the results to GrosFichiers, avoids working for nothing.

        # Ensure file pointers are reset
        if tmp_file:
            tmp_file.seek(0)
        if image_tmp_file:
            image_tmp_file.seek(0)

        summary = await self.summarize_file(job, tmp_file, image_tmp_file)

        await self.__submit_summary(job, summary)

        return summary

    async def __submit_summary(self, job: FileInformation, summary: SummaryText):
        key = job['key']
        secret_key: bytes = decode_base64_nopad(key['cle_secrete_base64'])
        key_id = key['cle_id']
        cleartext_summary = json.dumps({"comment": f"{summary.title}\n\n{summary.summary}"})
        cipher, encrypted_summary = chiffrer_mgs4_bytes_secrete(secret_key, cleartext_summary)
        encrypted_summary['cle_id'] = key_id

        if summary.labels:
            cleartext_tags = json.dumps({"tags": summary.labels})
            cipher, encrypted_tags = chiffrer_mgs4_bytes_secrete(secret_key, cleartext_tags)
            encrypted_tags['cle_id'] = key_id
        else:
            encrypted_tags = None

        # Send result as new comment for file
        summary_command = {
            'tuuid': job['tuuid'],
            'fuuid': job['version']['fuuid'],
            'comment': encrypted_summary,
            'tags': encrypted_tags,
        }

        producer = await self.__context.get_producer()
        for _ in range(5):
            try:
                await producer.command(summary_command, Constantes.DOMAINE_GROS_FICHIERS, "fileSummary",
                                       Constantes.SECURITE_PROTEGE, timeout=45)
                break
            except asyncio.TimeoutError:
                self.__logger.warning("Timeout sending summary result, will retry")
                await self.__context.wait(5)


    def get_client(self) -> OpenaiAsyncClient:
        configuration = self.__context.configuration
        connection_url = self.__api_url
        if connection_url.lower().startswith('https://'):
            # Use a millegrille certificate authentication
            cert = (configuration.private_cert_path, configuration.key_path)
            params = {'verify': configuration.ca_path, 'cert': cert}
        else:
            params = {}

        if params.get('verify'):
            ssl_context = self.__context.ssl_context
        else:
            ssl_context = None
        httpx_client = httpx.AsyncClient(verify=ssl_context, cert=params.get('cert'))
        return OpenaiAsyncClient(http_client=httpx_client, base_url=connection_url, api_key="DUMMY")


    async def summarize_file(self, job: FileInformation,
                             tmp_file: tempfile.NamedTemporaryFile,
                             image_tmp_file: tempfile.NamedTemporaryFile,
                             noformat=False) -> SummaryText:
        client = self.get_client()
        job_type = job['job_type']
        language = job['language']

        if noformat:
            format = None
        else:
            format = SummaryText

        token_padding = 768

        if job_type == CONST_JOB_SUMMARY_TEXT and tmp_file:
            raise NotImplementedError('TODO')
            tmp_file.seek(0)
            effective_context = self.__context_length
            if self.__supports_vision and image_tmp_file:
                # Also add image, can help with PDFs that have no text content
                image_tmp_file.seek(0)
                image_content = await asyncio.to_thread(image_tmp_file.read)
                image_tmp_file.seek(0)
                images = [image_content]
                effective_context -= 512  # Give enough space for the image
            else:
                images = None

            for i in range(0, 3):
                tmp_file.seek(0)
                system_prompt, command_prompt = await format_text_prompt(
                    self.__document_prompt,
                    language,
                    effective_context,
                    CONST_SUMMARY_NUM_PREDICT,
                    job['mimetype'],
                    tmp_file,
                    token_padding=token_padding
                )

                try:
                    response = await client.chat.completions.create(
                        model=self.__model,
                        prompt=command_prompt,
                        system=system_prompt,
                        response_format=format,
                        max_len=CONST_SUMMARY_NUM_PREDICT,
                        temperature=self.__temperature,
                        images=images,
                    )
                    # response = await client.generate(
                    #     model=self.__model,
                    #     prompt=command_prompt,
                    #     system=system_prompt,
                    #     response_format=format,
                    #     max_len=CONST_SUMMARY_NUM_PREDICT,
                    #     temperature=self.__temperature,
                    #     images=images,
                    # )
                    break
                except openai.BadRequestError as bre:
                    self.__logger.warning("Error summarizing file tuuid:%s/fuuid:%s padding %s: %s", job.get('tuuid'), job.get('fuuid'), token_padding, bre)
                    # Likely that context was exceeded (wrong tokenizer), retry by reducing availble context size
                    try:
                        response_json = bre.response.json()
                        error = response_json['error']
                        n_ctx = error['n_ctx']
                        n_prompt_tokens = error['n_prompt_tokens']
                        # Calculate ratio, remove the padding (sys prompt) from the returned context
                        ratio = n_prompt_tokens / (n_ctx - token_padding)
                    except (KeyError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
                        # Default to 33% excess
                        ratio = 1.33

                    # Reduce the effective available context by ratio sent back through error response
                    effective_context = int(effective_context / ratio)
            else:
                raise ValueError(f"Unable to summarize file: tuuid:{job.get('tuuid')}/fuuid:{job.get('fuuid')}")

        elif job_type == CONST_JOB_SUMMARY_IMAGE:
            try:
                mimetype = job['image_file']['mimetype']
            except KeyError:
                mimetype = 'image/png'

            if not image_tmp_file:
                image_tmp_file = tmp_file
                mimetype = job['mimetype']
            if not image_tmp_file:
                ValueError('No image to review')

            image_tmp_file.seek(0)
            system_prompt, command_prompt = self.format_image_prompt(language)

            image_uri = await asyncio.to_thread(encode_image_to_data_uri, image_tmp_file, mimetype)
            input_history = [{
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': command_prompt},
                    {'type': 'input_image', 'image_url': image_uri},
                ]
            }]

            for i in range(3):
                if i > 0:
                    self.__logger.debug("Resubmitting\n%s", input_history[1:])
                response = await client.responses.create(
                    model=self.__model,
                    instructions=system_prompt,
                    max_output_tokens=CONST_SUMMARY_NUM_PREDICT,
                    temperature=self.__temperature,
                    input=input_history,
                )

                if response.status != 'completed':
                    raise Exception(f"Error response, status {response.status}")

                result_text = response.output_text
                self.__logger.debug("Output text: %s", result_text)

                content = cleanup_json_output(result_text)
                self.__logger.debug("JSON content: %s", content)
                try:
                    summary = SummaryText.model_validate_json(content)
                    break
                except pydantic.ValidationError as ve:
                    self.__logger.warning("Error validating response: %s", ve)
                    validation_error = str(ve)
                    input_history.append({
                        'role': 'user',
                        'content': result_text
                    })
                    input_history.append({
                        'role': 'user',
                        'content': f'Pydantic reported a validation error, this may be due to unreported schema changes. Check the error and try again.\n{validation_error}'
                    })
                    pass
        else:
            raise ValueError(f"Unsupported job type: {job_type}")

        self.__logger.debug("Summary: %s", summary)
        return summary

    def format_image_prompt(self, language: str):
        system_prompt = self.__image_prompt + f"""

## Output format

The output is validated with this Pydantic schema: {SummaryText.model_fields}

# Personalized information

   * User language: {language}

You **MUST** reply in the user's language.
        """
        command_prompt = f"Describe this image. Make sure your response is in proper **JSON** formatting. It **MUST** begin with {{ and end with }}."

        return system_prompt, command_prompt


class JobPreparationException(Exception):

    def __init__(self, tuuid: Optional[str], fuuid: Optional[str], *args, **kwargs):
        super().__init__(args, kwargs)
        self.tuuid = tuuid
        self.fuuid = fuuid

