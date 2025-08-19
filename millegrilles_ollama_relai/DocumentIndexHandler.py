import asyncio
import datetime
import json
import logging
import pathlib
import tempfile
import math

from asyncio import TaskGroup
from typing import Optional, TypedDict

import nacl.exceptions
import openai
import tiktoken
from pydantic import Field, ValidationError

from ollama import AsyncClient, ResponseError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike, BaseRetriever
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfStreamError

from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_document_secrete, dechiffrer_bytes_secrete
from millegrilles_messages.chiffrage.Mgs4 import chiffrer_mgs4_bytes_secrete
from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_messages.Filehost import FilehostConnection
from millegrilles_ollama_relai.InstancesDao import InstanceDao

from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstance, model_name_to_id, OllamaInstanceManager
from millegrilles_ollama_relai.Util import decode_base64_nopad, cleanup_json_output, conditional_convert_to_png
from millegrilles_ollama_relai.Structs import SummaryText
from millegrilles_ollama_relai.prompts.index_system_prompts import PROMPT_INDEXING_SYSTEM_IMAGES, \
    PROMPT_INDEXING_SYSTEM_DOCUMENT

LOGGER = logging.getLogger(__name__)

CONST_ACTION_RAG = 'leaseForRag'
CONST_ACTION_SUMMARY = 'leaseForSummary'

CONST_JOB_RAG = 'rag'
CONST_JOB_SUMMARY_TEXT = 'summaryText'
CONST_JOB_SUMMARY_IMAGE = 'summaryImage'
CONST_CHAR_MULTIPLIER = 4.5
CONST_SUMMARY_NUM_PREDICT = 1024


class FileInformation(TypedDict):
    job_type: Optional[str]
    lease_action: str
    tuuid: Optional[str]
    fuuid: Optional[str]
    user_id: Optional[str]
    language: str
    domain: str
    cuuids: Optional[list[str]]
    metadata: Optional[dict]
    mimetype: Optional[str]
    version: Optional[dict]
    key: Optional[dict]
    decrypted_metadata: Optional[dict]
    tmp_file: Optional[tempfile.NamedTemporaryFile]
    image_tmp_file: Optional[tempfile.NamedTemporaryFile]
    media: Optional[dict]
    image_file: Optional[dict]

QUERY_BATCH_RAG_LEN = 30
# CONTEXT_LEN = 4096
# DOC_LEN = 1024
# DOC_OVERLAP = math.floor(DOC_LEN / 20)
# EMBEDDING_MODEL = "all-minilm"
# EMBEDDING_MODEL = "nomic-embed-text:137m-v1.5-fp16"

class DocumentIndexHandler:

    def __init__(self, context: OllamaContext, ollama_instances: OllamaInstanceManager, attachment_handler: FilehostConnection):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__ollama_instances = ollama_instances
        self.__attachment_handler = attachment_handler

        self.__semaphore_db = asyncio.BoundedSemaphore(1)
        self.__event_fetch_jobs = asyncio.Event()

        self.__intake_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=QUERY_BATCH_RAG_LEN)
        self.__indexing_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=2)

        self.__vector_store_cache: dict[str, VectorStore] = dict()

    async def setup(self):
        pass

    async def register_rag(self, message: MessageWrapper):
        producer = await self.__context.get_producer()

        # Re-emit the message on the model's process Q
        query_model = self.__context.model_configuration['rag_query_model_name']
        model_id = model_name_to_id(query_model)

        # attachements = {'correlation_id': message.correlation_id, 'reply_to': message.reply_to}
        # attachements.update(message.original['attachements'])

        await producer.command(
            message.original, domain='ollama_relai', partition=model_id, action='process',
            reply_to=message.reply_to, correlation_id=message.correlation_id,
            exchange=Constantes.SECURITE_PROTEGE,
            noformat=True, nowait=True)

        return False  # The response is handled by the processing instance

    async def trigger_indexing(self, delay: Optional[float] = None):
        if delay:
            await self.__context.wait(delay)
        self.__event_fetch_jobs.set()

    async def query_rag(self, instance: OllamaInstance, message: MessageWrapper) -> Optional[dict]:
        # Recover the keys, send to MaitreDesCles to get the decrypted value
        encrypted_query = message.parsed['encrypted_query']
        domain_signature = encrypted_query['key']['signature']
        decryption_keys = encrypted_query['key']['keys']
        dechiffrage = {'signature': domain_signature, 'cles': decryption_keys}
        producer = await self.__context.get_producer()
        try:
            decryption_key_response = await producer.request(
                dechiffrage, 'MaitreDesCles', 'dechiffrageMessage',
                exchange=Constantes.SECURITE_PROTEGE, timeout=3)
        except asyncio.TimeoutError:
            self.__logger.error("Error getting conversation decryption key for AI chat message")
            return {'ok': False, 'err': 'Timeout getting decryption key'}

        decryption_key: bytes = decode_base64_nopad(decryption_key_response.parsed['cle_secrete_base64'])
        query_bytes = await asyncio.to_thread(dechiffrer_bytes_secrete, decryption_key, encrypted_query)
        query_content = json.loads(query_bytes.decode('utf-8'))
        query = query_content['query']
        cuuid = query_content.get('cuuid')  # Directory to use
        user_id = message.certificat.get_user_id

        if user_id is None:
            return {'ok': False, 'code': 403, 'err': 'Access denied, no user_id in certificate'}

        model_configuration = self.__context.model_configuration
        rag_configuration = self.__context.rag_configuration
        if model_configuration is None or rag_configuration is None:
            raise Exception("No Model/RAG configuration provided - configure it with MilleGrilles AiChat")

        query_model = model_configuration['rag_query_model_name']
        embedding_model = model_configuration['embedding_model_name']
        context_len = rag_configuration.get('context_len')
        doc_chunk_len = rag_configuration['document_chunk_len']

        async with instance.semaphore:
            vector_store = await asyncio.to_thread(self.open_vector_store, Constantes.DOMAINE_GROS_FICHIERS, user_id, instance, embedding_model)

            # Determine is we need to use a document filter
            if cuuid and cuuid != '':
                # Filter by directory
                retriever = FilteredCuuidsRetriever(vectorstore=vector_store.as_retriever(), cuuid=cuuid)
            else:
                # No filter
                retriever = vector_store.as_retriever()

            # client = instance.get_async_client(self.__context.configuration)
            client = instance.connection
            # Calculate doc limit using ~2 * context chars
            limit = math.floor(context_len * 2.5 / doc_chunk_len)
            try:
                system_prompt, command_prompt, doc_ref = await format_prompt(retriever, query, limit)
            except NoDocumentsFoundException:
                return {'ok': False, 'code': 404, 'err': 'No matching documents found'}

            self.__logger.debug(f"PROMPT (limit: {limit}, len:{len(system_prompt) + len(command_prompt)})\n{command_prompt}")
            response = await client.generate(
                model=query_model,
                # prompt=system_prompt + "\n" + command_prompt,
                prompt=command_prompt,
                system=system_prompt,
                max_len=context_len,
                temperature=0.0,
            )

        self.__logger.debug("Response: %s" % response.message['content'])

        response = {'ok': True, 'response': response.message['content'], 'ref': doc_ref}
        producer = await self.__context.get_producer()
        await producer.encrypt_reply([message.certificat], response,
                                     correlation_id=message.correlation_id,
                                     reply_to=message.reply_to)

        return None  # Already replied

    async def run(self):
        if not self.__context.configuration.rag_active and not self.__context.configuration.summary_active:
            return  # Nothing to do, abort thread

        self.__logger.info("DocumentIndexHandler thread starting")
        async with TaskGroup() as group:
            group.create_task(self.__query_thread())
            group.create_task(self.__intake_thread())
            group.create_task(self.__index_thread())
            group.create_task(self.__trigger_fetch_interval())
        self.__logger.info("DocumentIndexHandler thread done")

    async def __trigger_fetch_interval(self):
        while self.__context.stopping is False:
            self.__event_fetch_jobs.set()
            self.__vector_store_cache.clear()  # Rudely clear the cache
            await self.__context.wait(120)  # Trigger once every 2 minutes

        # Unblock the query thread
        self.__event_fetch_jobs.set()

    async def __query_thread(self):
        # Wait for initialization
        await self.__context.ai_configuration_loaded.wait()
        while not self.__ollama_instances.ready:
            await self.__context.wait(3)
            if self.__context.stopping:
                return

        while self.__context.stopping is False:
            self.__event_fetch_jobs.clear()

            if self.__intake_queue.qsize() == 0:
                lease_actions: list[str] = list()
                try:
                    if self.__context.configuration.summary_active:
                        try:
                            # Ensure that the summary model is available locally
                            model_configuration = self.__context.model_configuration
                            embedding_model = model_configuration['embedding_model_name']
                            if embedding_model:
                                _instance = self.__ollama_instances.pick_instance_for_model(embedding_model)
                            lease_actions.append(CONST_ACTION_SUMMARY)
                        except (TypeError, KeyError) as e:
                            self.__logger.debug(f"RAG configuration not initialized yet: {e}")

                    if self.__context.configuration.rag_active:
                        try:
                            # Ensure that the RAG model is available locally
                            model_configuration = self.__context.model_configuration
                            embedding_model = model_configuration['embedding_model_name']
                            if embedding_model:
                                _instance = self.__ollama_instances.pick_instance_for_model(embedding_model)
                                lease_actions.append(CONST_ACTION_RAG)
                        except (TypeError, KeyError) as e:
                            self.__logger.debug(f"RAG configuration not initialized yet: {e}")

                    for lease_action in lease_actions:
                        # Try to fetch more items
                        self.__logger.debug("Triggering fetch of new batch of files for RAG/Summary (low/empty intake queue)")
                        try:
                            await self.__query_batch(lease_action)
                        except asyncio.TimeoutError:
                            self.__logger.warning("Timeout querying for files to index")
                        except (AttributeError, KeyError, ValueError) as e:
                            self.__logger.warning("Error loading file or filehost configuration: %s" % e)
                except:
                    self.__logger.exception("__query_thread Unhandled exception - stopping")
                    self.__context.stop()

            await self.__event_fetch_jobs.wait()

        # Stop threads on queues
        self.__intake_queue.empty()
        # self.__indexing_queue.empty()  # Don't clear the indexing queue, let it finish
        await self.__intake_queue.put(None)
        await self.__indexing_queue.put(None)

    async def __intake_thread(self):
        while self.__context.stopping is False:
            # Trigger fetch
            self.__event_fetch_jobs.set()

            # Wait for jobs
            job = await self.__intake_queue.get()
            if job is None:
                self.__logger.info("Stopping intake thread (None job received)")
                return  # Stopping

            job_type = job['job_type']

            # Decrypt all content
            try:
                secret_key_str = job['key']['cle_secrete_base64']
                secret_key = decode_base64_nopad(secret_key_str)

                if job == CONST_ACTION_RAG:
                    metadata = dechiffrer_document_secrete(secret_key, job['metadata'])
                    filename = metadata['nom']
                    job['decrypted_metadata'] = metadata
                else:
                    filename = job['fuuid']

                version = job.get('version')
                fuuid = job.get('fuuid') or version['fuuid']
            except (TypeError, KeyError, UnicodeDecodeError) as e:
                self.__logger.warning(f"__intake_thread Error getting value for tuuid: {job.get('tuuid')} / fuuid: {job.get('fuuid')}: {str(e)}, skipping")
                await self.__cancel_job(job)
                continue  # Skip this job
            except:
                self.__logger.exception("Unhandled exception, quitting intake_thread")
                self.__context.stop()
            else:
                if job_type and fuuid:  # Job to do, download file
                    image_file_to_download = job.get('image_file')
                    file_to_download: Optional[dict] = None
                    tmp_file = None
                    image_tmp_file = None
                    try:
                        key = job['key']

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
                            image_file_to_download['format'] = image_file_to_download.get('format') or key.get('format') or 'mgs4'  # Default format

                        if file_to_download:
                            tmp_file = tempfile.NamedTemporaryFile(mode='wb+')
                            await self.__download_file(filename, fuuid, secret_key_str, file_to_download, tmp_file)
                            # Process decrypted file
                            tmp_file.seek(0)
                            job['tmp_file'] = tmp_file

                        if image_file_to_download:
                            image_tmp_file = tempfile.NamedTemporaryFile(mode='wb+')
                            await self.__download_file(filename, fuuid, secret_key_str, image_file_to_download, image_tmp_file)

                            # Replaces the content of tmp_file with a PNG if the file is not either png or jpeg.
                            image_tmp_file.seek(0)
                            try:
                                image_mimetype = job['image_file']['mimetype']
                            except (AttributeError, KeyError):
                                image_mimetype = 'image/webp'
                            await conditional_convert_to_png(image_mimetype, image_tmp_file)
                            image_tmp_file.seek(0)

                            job['image_tmp_file'] = image_tmp_file

                    except nacl.exceptions.RuntimeError as e:
                        if tmp_file:
                            tmp_file.close()
                        if image_tmp_file:
                            image_tmp_file.close()
                        self.__logger.error(f"Error decrypting fuuid {fuuid}, params nonce:{image_file_to_download} CANCELLING: {e}")
                        await self.__cancel_job(job)
                        continue
                    except:
                        if tmp_file:
                            tmp_file.close()
                        if image_tmp_file:
                            image_tmp_file.close()
                        self.__logger.exception(f"Error downloading fuuid {fuuid}, will retry")
                        continue  # Ignore this file for now, don't mark it processed

            # Pass content on to indexing
            await self.__indexing_queue.put(job)

    async def __download_file(self, filename: str, fuuid: str, secret_key_str: str, file_to_download: dict, tmp_file: tempfile.TemporaryFile):
        # For media encoded thumbnails/images, need to stick to file_to_download
        try:
            filesize = await self.__attachment_handler.download_decrypt_file(
                secret_key_str, file_to_download, tmp_file)
            self.__logger.debug(f"Downloaded {filesize} bytes for file {filename}")
        except* asyncio.CancelledError:
            raise Exception(f"Error downloading fuuid {fuuid}, will retry")

    async def __cancel_job(self, job: FileInformation):
        lease_action = job['lease_action']
        try:
            producer = await self.__context.get_producer()
            if lease_action == CONST_ACTION_SUMMARY:
                tuuid = job['tuuid']
                fuuid = job['version']['fuuid']
                command = {'tuuid': tuuid, 'fuuid': fuuid}
                await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, "fileSummary",
                                       Constantes.SECURITE_PROTEGE, timeout=45)
            elif lease_action == CONST_ACTION_RAG:
                command = {"tuuid": job['tuuid'], "user_id": job['user_id']}
                await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, "confirmRag",
                                       Constantes.SECURITE_PROTEGE, timeout=45)
        except:
            self.__logger.exception("Error cancelling job\n %s" % job)

    async def __index_thread(self):
        while self.__context.stopping is False:
            job: Optional[FileInformation] = await self.__indexing_queue.get()
            if job is None:
                return  # Stopping

            job_type = job['job_type']

            tmp_file = job.get('tmp_file')
            image_tmp_file = job.get('image_tmp_file')
            tuuid = job.get('tuuid')
            fuuid = job.get('fuuid')
            if tmp_file or image_tmp_file:
                try:
                    rag_configuration = self.__context.rag_configuration
                    if rag_configuration is None:
                        raise Exception("No RAG configuration provided - configure it with MilleGrilles AiChat")

                    try:
                        if job_type == CONST_JOB_RAG:
                            if tmp_file:
                                await self.__run_rag_indexing(job, tmp_file)
                        elif job_type in [CONST_JOB_SUMMARY_TEXT, CONST_JOB_SUMMARY_IMAGE]:
                            await self.__run_summarize_file(job, tmp_file, image_tmp_file)
                        else:
                            self.__logger.error(f"Unknown job type {job_type}, CANCELLING for tuuid:{tuuid}/fuuid:{fuuid}")
                            await self.__cancel_job(job)
                    except* asyncio.CancelledError as e:
                        if self.__context.stopping:
                            pass
                        else:
                            self.__logger.warning(f"RAG/Summary job cancelled: {e}")
                except (FatalSummaryException, ValidationError, ResponseError, UnicodeDecodeError) as e:
                    self.__logger.error(f"Error handling file tuuid:{tuuid}/fuuid:{fuuid}, CANCELLING: {e}")
                    await self.__cancel_job(job)
                except:
                    self.__logger.exception(f"Error processing file tuuid:{tuuid}/fuuid:{fuuid}, will retry")
                    continue
                finally:
                    self.__logger.debug("Closing tmp file for tuuid:%s/fuuid%s", tuuid, fuuid)
                    if tmp_file:
                        tmp_file.close()
                    if image_tmp_file:
                        image_tmp_file.close()
            else:
                # Nothing to do
                await self.__cancel_job(job)

    async def __run_rag_indexing(self, job: FileInformation, tmp_file: tempfile.NamedTemporaryFile):
        model_configuration = self.__context.model_configuration
        rag_configuration = self.__context.rag_configuration
        if rag_configuration is None:
            raise Exception("No RAG configuration provided - configure it with MilleGrilles AiChat")

        job_type = job['job_type']
        tuuid = job.get('tuuid')
        user_id = job['user_id']
        domain = job['domain']
        cuuids = job.get('cuuids')
        metadata = job.get('decrypted_metadata')
        try:
            filename = metadata['nom']
            self.__logger.debug(f"Processing file {filename} with jon type {job_type}")
        except (TypeError, KeyError):
            filename = tuuid

        embedding_model = model_configuration['model_embedding_name']
        instance = self.__ollama_instances.pick_instance_for_model(embedding_model)
        if instance is None:
            raise Exception(f'Unsupported model: {embedding_model}')

        try:
            async with instance.semaphore:
                vector_store = await asyncio.to_thread(self.open_vector_store, domain, user_id, instance, embedding_model)
                await asyncio.to_thread(index_pdf_file, vector_store, tuuid, filename, cuuids, tmp_file)
        except (TypeError, ValueError, PdfStreamError) as e:
            self.__logger.exception(f"Error processing file tuuid:{tuuid}, rejecting: %s" % str(e))
            pass  # The file will be marked as processed later on

        # Mark the file as processed
        producer = await self.__context.get_producer()
        command = {"tuuid": tuuid, "user_id": user_id}
        for i in range(0, 5):
            try:
                await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, "confirmRag",
                                       Constantes.SECURITE_PROTEGE, timeout=45)
                break
            except asyncio.TimeoutError:
                self.__logger.warning("Timeout sending RAG result, will retry")
                await self.__context.wait(5)

    async def __run_summarize_file(self, job: FileInformation, tmp_file: tempfile.NamedTemporaryFile, image_tmp_file: tempfile.NamedTemporaryFile()):
        llm_configuration = self.__context.chat_configuration
        model_configuration = self.__context.model_configuration
        rag_configuration = self.__context.rag_configuration
        if rag_configuration is None:
            raise Exception("No RAG configuration provided - configure it with MilleGrilles AiChat")

        job_type = job['job_type']
        if job_type in [CONST_JOB_RAG, CONST_JOB_SUMMARY_TEXT]:
            model = model_configuration.get('summary_model_name') or llm_configuration['model_name']
        elif job_type == CONST_JOB_SUMMARY_IMAGE:
            model = model_configuration.get('vision_model_name') or model_configuration.get('summary_model_name') or llm_configuration['model_name']
        else:
            raise ValueError(f'Unsupported job type: {job_type}')

        instance = self.__ollama_instances.pick_instance_for_model(model)
        if instance is None:
            raise Exception(f'Unsupported model: {model}')

        instance_model = instance.get_model(model)
        context_length = instance_model.context_length
        supports_vision = 'vision' in instance_model.capabilities

        encryption_key = job['key']
        secret_key: bytes = decode_base64_nopad(encryption_key['cle_secrete_base64'])
        key_id = encryption_key['cle_id']

        async with instance.semaphore:
            # Make a few attempts to generate valid JSON, increase the temperature each time to vary the result.
            temperature = 0.1
            client = instance.connection
            for i in range(0, 2):
                # Ensure file pointers are reset
                if tmp_file:
                    tmp_file.seek(0)
                if image_tmp_file:
                    image_tmp_file.seek(0)

                try:
                    # client = instance.get_async_client(self.__context.configuration)
                    summary = await summarize_file(client, job, model, tmp_file, image_tmp_file, context_len=context_length, temperature=temperature, vision=supports_vision)
                    break
                except ValidationError as e:
                    self.__logger.warning(f"JSON validation error on generate response for file tuuid:{job.get('tuuid')}/fuuid:{job.get('fuuid')} (temperature: {temperature}): {e}")
                    temperature += 0.2
            else:
                # Unable to get a structured JSON summary output, revert to string output

                # Ensure file pointers are reset
                if tmp_file:
                    tmp_file.seek(0)
                if image_tmp_file:
                    image_tmp_file.seek(0)

                summary = await summarize_file(client, job, model, tmp_file, image_tmp_file,
                                               context_len=context_length, temperature=0.0, noformat=True, vision=supports_vision)
                # raise FatalSummaryException(f'Unable to get a valid JSON output for file tuuid:{job.get('tuuid')}/fuuid:{job.get('fuuid')}')

        # Encrypt result
        cleartext_summary = json.dumps({"comment": summary.summary})
        cipher, encrypted_summary = chiffrer_mgs4_bytes_secrete(secret_key, cleartext_summary)
        encrypted_summary['cle_id'] = key_id

        if summary.tags:
            cleartext_tags = json.dumps({"tags": summary.tags})
            cipher, encrypted_tags = chiffrer_mgs4_bytes_secrete(secret_key, cleartext_tags)
            encrypted_tags['cle_id'] = key_id
        else:
            encrypted_tags = None

        # Send result as new comment for file
        summary_command = {
            'tuuid': job['tuuid'],
            'fuuid': job['version']['fuuid'],
            'comment': encrypted_summary,
            # 'file_language': summary.language,
            'tags': encrypted_tags,
        }
        producer = await self.__context.get_producer()
        for i in range(0, 5):
            try:
                await producer.command(summary_command, Constantes.DOMAINE_GROS_FICHIERS, "fileSummary",
                                       Constantes.SECURITE_PROTEGE, timeout=45)
                break
            except asyncio.TimeoutError:
                self.__logger.warning("Timeout sending summary result, will retry")
                await self.__context.wait(5)


    async def __query_batch(self, lease_action:str):
        producer = await self.__context.get_producer()

        # Calculate batch size from space left in processing queue
        batch_size = self.__intake_queue.maxsize - self.__intake_queue.qsize()
        if batch_size == 0:
            return  # Nothing to do

        try:
            filehost_id = self.__context.filehost.filehost_id
        except AttributeError:
            # Filehost not loaded yet, wait and retry
            await self.__context.wait(5)
            filehost_id = self.__context.filehost.filehost_id

        batch_size = self.__intake_queue.maxsize - self.__intake_queue.qsize()

        # Get the batch
        command = {"batch_size": batch_size, "filehost_id": filehost_id}
        try:
            response = await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, lease_action,
                                              Constantes.SECURITE_PROTEGE, timeout=60)
        except asyncio.TimeoutError:
            self.__logger.warning(f"Timeout on {lease_action}, will retry")
            return

        parsed = response.parsed
        if parsed['ok'] is not True:
            self.__logger.warning("Error retrieving batch of files: %s" % parsed)
        elif parsed.get('code') == 1:
            self.__logger.debug(f"No more files to process for {lease_action} on filehost_id {filehost_id}")
            return

        # Parse batch file and insert on queue per item
        leases = parsed['leases']
        secret_keys: list = parsed['secret_keys']

        self.__logger.debug(f"Received batch of {len(leases)} files for {lease_action} from filehost_id:{filehost_id}")

        for lease in leases:
            metadata = lease.get('metadata')
            version = lease.get('version')
            cuuids = lease.get('cuuids')
            fuuid: Optional[str] = None

            cle_id = None
            mimetype = lease.get('mimetype')
            if version:
                fuuid = version['fuuid']
                cle_id = version.get('cle_id')
                mimetype = mimetype or version.get('mimetype')

            if metadata and not cle_id:
                cle_id = cle_id or metadata.get('cle_id') or metadata.get('ref_hachage_bytes')
            cle_id = cle_id or fuuid

            try:
                key = [k for k in secret_keys if k['cle_id'] == cle_id].pop()
            except IndexError:
                self.__logger.warning(f"Missing key for fuuid {fuuid}, skipping")
                key = None

            image_file = None
            if lease_action == CONST_ACTION_RAG:
                if mimetype == 'application/pdf' or mimetype.startswith('text/'):
                    job_type = CONST_JOB_RAG
                else:
                    job_type = None  # Nothing to do
            elif lease_action == CONST_ACTION_SUMMARY:
                media = lease.get('media')
                if mimetype == 'application/pdf' or mimetype.startswith('text/'):
                    job_type = CONST_JOB_SUMMARY_TEXT
                elif mimetype.startswith('image/'):
                    job_type = CONST_JOB_SUMMARY_IMAGE
                    # Check to find a webp fuuid, will be smaller and easier to digest
                else:
                    job_type = None

                try:
                    images = media['images']
                    webp_img = [m for m in images.keys() if m.startswith('image/webp')].pop()
                    image_file = images[webp_img]
                    image_file['fuuid'] = image_file['hachage']
                except (AttributeError, IndexError, TypeError):
                    pass

            else:
                job_type = None

            info: FileInformation = {
                'job_type': job_type,
                'lease_action': lease_action,
                'tuuid': lease.get('tuuid'),
                'fuuid': lease.get('fuuid'),
                'user_id': lease['user_id'],
                'language': 'en_US',
                'domain': Constantes.DOMAINE_GROS_FICHIERS,
                'cuuids': cuuids,
                'metadata': metadata,
                'mimetype': lease.get('mimetype'),
                'version': lease.get('version'),
                'key': key,
                'tmp_file': None,
                'image_tmp_file': None,
                'decrypted_metadata': None,
                'media': lease.get('media'),
                'image_file': image_file,
            }

            await self.__intake_queue.put(info)

        pass

    def open_vector_store(self, domain: str, user_id: str, instance: OllamaInstance, embedding_model: str) -> VectorStore:
        # Collection name must be between 3 and 63 chars, truncate the user_id to the last 32 chars
        user_id_trunc = user_id[-32:]
        collection_name = f'{domain}_{user_id_trunc}'

        vector_store = self.__vector_store_cache.get(collection_name)
        if vector_store:
            return vector_store

        configuration = self.__context.configuration
        options = instance.get_client_options(configuration)
        base_url = options['host']
        del options['host']

        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
            # model="llama3.2:3b-instruct-q8_0",
            # num_gpu=0,  # Disable GPU to avoid swapping models on ollama
            client_kwargs=options,
        )

        path_db = pathlib.Path(configuration.dir_rag, "chroma_langchain_db")
        path_db.mkdir(exist_ok=True)

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(path_db),  # Where to save data locally, remove if not necessary
        )

        self.__vector_store_cache[collection_name] = vector_store

        return vector_store


def index_pdf_file(vector_store: VectorStore, tuuid: str, filename: str, cuuids: Optional[list[str]], tmp_file: tempfile.NamedTemporaryFile, doc_len=1000, overlap=50):
    loader = PyPDFLoader(tmp_file.name, mode="single")
    document_list = loader.load()
    document = document_list[0]
    splitter = RecursiveCharacterTextSplitter(chunk_size=doc_len, chunk_overlap=overlap)
    chunks = splitter.split_documents([document])
    chunk_no = 1
    for chunk in chunks:
        doc_id = f"{tuuid}/{chunk_no}"
        chunk_no += 1
        chunk.id = doc_id
        chunk.metadata['source'] = filename
        if cuuids is not None:
            chunk.metadata['cuuids'] = ','.join(cuuids)
    vector_store.add_documents(list(chunks))


async def summarize_file(client: InstanceDao, job: FileInformation, model: str,
                         tmp_file: tempfile.NamedTemporaryFile, image_tmp_file: tempfile.NamedTemporaryFile,
                         context_len=4096, temperature=0.0, noformat=False, vision=False) -> SummaryText:
    job_type = job['job_type']
    language = job['language']

    if noformat:
        format = None
    else:
        format = SummaryText

    token_padding = 768

    if job_type == CONST_JOB_SUMMARY_TEXT and tmp_file:
        tmp_file.seek(0)
        effective_context = context_len
        # if vision and image_tmp_file:
        #     # Also add image, can help with PDFs that have no text content
        #     image_tmp_file.seek(0)
        #     image_content = await asyncio.to_thread(image_tmp_file.read)
        #     image_tmp_file.seek(0)
        #     images = [image_content]
        #     effective_context -= 512  # Give enough space for the image
        # else:
        #     images = None

        for i in range(0, 3):
            tmp_file.seek(0)
            system_prompt, command_prompt = await format_text_prompt(
                language, effective_context, CONST_SUMMARY_NUM_PREDICT, job['mimetype'], tmp_file, token_padding=token_padding)

            try:
                response = await client.generate(
                    model=model,
                    prompt=command_prompt,
                    system=system_prompt,
                    response_format=format,
                    max_len=CONST_SUMMARY_NUM_PREDICT,
                    temperature=temperature,
                    # images=images,
                    # options={"temperature": temperature, "num_ctx": context_len, "num_predict": CONST_SUMMARY_NUM_PREDICT}
                )
                break
            except openai.BadRequestError as bre:
                LOGGER.warning("Error summarizing file tuuid:%s/fuuid:%s padding %s: %s", job.get('tuuid'), job.get('fuuid'), token_padding, bre)
                # Likely that context was exceeded, retry
                token_padding *= 2  # Double the padding
        else:
            raise ValueError(f"Unable to summarize file: tuuid:{job.get('tuuid')}/fuuid:{job.get('fuuid')}")

    elif job_type == CONST_JOB_SUMMARY_IMAGE:
        if not image_tmp_file:
            image_tmp_file = tmp_file
        if not image_tmp_file:
            ValueError('No image to review')

        image_tmp_file.seek(0)
        system_prompt, command_prompt = await format_image_prompt(language)
        image_content = await asyncio.to_thread(image_tmp_file.read)

        response = await client.generate(
            model=model,
            prompt=command_prompt,
            system=system_prompt,
            response_format=format,
            max_len=CONST_SUMMARY_NUM_PREDICT,
            temperature=temperature,
            images=[image_content]
            # options={"temperature": temperature, "num_ctx": context_len, "num_predict": CONST_SUMMARY_NUM_PREDICT},
        )
    else:
        raise ValueError(f"Unsupported job type: {job_type}")

    if noformat:
        summary = SummaryText(summary=response.message['content'], tags=None)
    else:
        content = cleanup_json_output(response.message['content'])
        summary = SummaryText.model_validate_json(content)

    return summary


class FilteredCuuidsRetriever(BaseRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    cuuid: str

    def get_relevant_documents(self, query: str) -> list[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        output = list()
        for doc in results:
            cuuids = doc.metadata.get('cuuids')
            if cuuids is not None and self.cuuid in cuuids:
                output.append(doc)
        return output


class NoDocumentsFoundException(Exception):
    pass


async def format_prompt(retriever: RetrieverLike, query: str, limit: int, filtre: Optional[dict[str, str]] = None) -> (str, str, list[dict]):
    context_response = await retriever.ainvoke(query, k=limit)

    doc_ref = list()
    for elem in context_response:
        doc_ref.append({"id": elem.id, "page_content": elem.page_content, "metadata": dict(elem.metadata)})

    if len(doc_ref) == 0:
        raise NoDocumentsFoundException()

    # Wrap the documents in <source /> tags with id
    context_tags: list[str] = list()
    for elem in context_response:
        tag = f'<source id="{elem.id}"'
        try:
            source: str = elem.metadata['source']
            source = source.replace("\"", "'")  # Replace all double-quotes by single quotes
            tag += f' src="{source}"'
        except KeyError:
            pass
        tag += f'>{elem.page_content}</source>'
        # context_tags = [f'<source id="{elem.id}">{elem.page_content}</source>' for elem in context_response]
        context_tags.append(tag)

    now = datetime.datetime.now()

    # Prompt source - open-webui (https://openwebui.com/)
    system_prompt = f"""
### Task:
Respond to the user query using the provided context, incorporating inline citations in the format [id] **only when the <source> tag includes an explicit id attribute** (e.g., <source id="abcd-1234/1">).

### Guidelines:
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
- **Only include inline citations using [id] (e.g., [abcd-01], [efgh-22]) when the <source> tag includes an id attribute.**
- Do not cite if the <source> tag does not contain an id attribute.
- Do not use XML tags in your response.
- Ensure citations are concise and directly related to the information provided.
- The current date and time is {now}, use this as now when appropriate.

### Example of Citation:
If the user asks about a specific topic and the information is found in a source with a provided id attribute, the response should include the citation like in the following example:
* "According to the study, the proposed method increases efficiency by 20% [abcd-01]."

### Output:
Provide a clear and direct response to the user's query, including inline citations in the format [id] only when the <source> tag with id attribute is present in the context.
    """

    command_prompt = f"""
<context>
{"\n".join(context_tags)}
</context>

<user_query>
{query}
</user_query>    
    """

    return system_prompt, command_prompt, doc_ref


async def format_image_prompt(language: str):
    params = {'language': language}
    system_prompt = PROMPT_INDEXING_SYSTEM_IMAGES.format(**params)
    command_prompt = f"Describe this image"

    return system_prompt, command_prompt


async def format_text_prompt(language: str, context_len: int, completion_len: int, mimetype: str, tmp_file: tempfile.NamedTemporaryFile, token_padding=1024) -> (str, str):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    char_multiplier = CONST_CHAR_MULTIPLIER

    params = {'language': language}
    system_prompt = PROMPT_INDEXING_SYSTEM_DOCUMENT.format(**params)

    if mimetype == 'application/pdf':
        extraction_kwargs = {'strict': False}
        loader = PyPDFLoader(tmp_file.name, mode="single", extraction_kwargs=extraction_kwargs)
        try:
            document_list = await asyncio.to_thread(loader.load)
        except (AttributeError, PdfStreamError) as e:
            raise FatalSummaryException(e)
        content = document_list[0].page_content

    elif mimetype.startswith('text/'):
        content = await asyncio.to_thread(tmp_file.read, char_multiplier * context_len)
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError as e:
            raise FatalSummaryException(e)

    else:
        raise ValueError(f"Unsupported document mimetype: {mimetype}")

    command_prompt = f"<Document>\n{content}\n</Document>"

    # Trim content
    system_prompt_len = len(encoding.encode(system_prompt))
    free_space = context_len - system_prompt_len - completion_len - token_padding
    try:
        encoded_content = encoding.encode(content)
        encoded_content_len = len(encoded_content)
    except ValueError:
        # Unable to encode, using estimate of 2.5 chars per token
        encoded_content = None
        encoded_content_len = int(math.floor(len(content) / 2.5))

    if encoded_content_len > free_space:
        try:
            # Truncate tokens,
            if encoded_content:
                encoded_content = encoded_content[0:free_space]
                content = encoding.decode(encoded_content)
            else:
                free_chars = int(free_space * 2.5)
                content = content[0:free_chars]
            LOGGER.info(f"Truncated document to {free_space} tokens ({len(content)} chars). Initial len:{len(system_prompt + command_prompt)}")
            # Prepare a new prompt with truncated output
            command_prompt = f"<Document>\n{content}\n</Document>"
        except IndexError:
            LOGGER.debug(f"Document not truncated, len: {len(content)}/{context_len}")
    else:
        LOGGER.debug(f"Document not truncated, {len(encoded_content)+system_prompt_len}/{context_len} tokens ({len(system_prompt + command_prompt)} chars)")
        # break

    tmp_file.seek(0)

    return system_prompt, command_prompt


class FatalSummaryException(Exception):
    pass