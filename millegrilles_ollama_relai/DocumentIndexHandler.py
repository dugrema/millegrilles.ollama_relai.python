import asyncio
import logging
import tempfile
from asyncio import TaskGroup
from typing import Optional, TypedDict

import multibase

from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_document_secrete
from millegrilles_messages.messages import Constantes

from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.Util import decode_base64_nopad


class FileInformation(TypedDict):
    tuuid: str
    user_id: str
    metadata: dict
    mimetype: Optional[str]
    version: Optional[dict]
    key: dict
    decrypted_metadata: Optional[dict]
    tmp_file: Optional[tempfile.TemporaryFile]


class DocumentIndexHandler:

    def __init__(self, context: OllamaContext, attachment_handler: AttachmentHandler):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__attachment_handler = attachment_handler

        self.__semaphore_db = asyncio.BoundedSemaphore(1)
        self.__event_fetch_jobs = asyncio.Event()

        self.__intake_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=5)
        self.__indexing_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=2)

    async def setup(self):
        pass

    async def index_documents(self, message: MessageWrapper):
        # Fetch the file
        with tempfile.TemporaryFile(mode="wb+") as tmp_file:
            tmp_file.seek(0)
            async with self.__semaphore_db:
                raise NotImplementedError("TODO")

            # return {"ok": False}

    async def query_documents(self, message: MessageWrapper):
        raise NotImplementedError("TODO")
        # return {"ok": False}

    async def run(self):
        async with TaskGroup() as group:
            group.create_task(self.__query_thread())
            group.create_task(self.__intake_thread())
            group.create_task(self.__index_thread())
            group.create_task(self.__trigger_fetch_interval())

    async def __trigger_fetch_interval(self):
        while self.__context.stopping is False:
            self.__event_fetch_jobs.set()
            await self.__context.wait(30)

        # Unblock the query thread
        self.__event_fetch_jobs.set()

    async def __query_thread(self):
        while self.__context.stopping is False:
            self.__event_fetch_jobs.clear()
            if self.__indexing_queue.qsize() < 3:
                # Try to fetch more items
                try:
                    await self.__query_batch_rag()
                except asyncio.TimeoutError:
                    self.__logger.warning("Timeout querying for files to index")

            await self.__event_fetch_jobs.wait()

        # Stop threads on queues
        self.__intake_queue.empty()
        # self.__indexing_queue.empty()  # Don't clear the indexing queue, let it finish
        await self.__intake_queue.put(None)
        await self.__indexing_queue.put(None)

    async def __intake_thread(self):
        while self.__context.stopping is False:
            job = await self.__intake_queue.get()
            if job is None:
                return  # Stopping

            # Decrypt all content
            secret_key_str = job['key']['cle_secrete_base64']
            secret_key = decode_base64_nopad(secret_key_str)
            metadata = dechiffrer_document_secrete(secret_key, job['metadata'])
            filename = metadata['nom']
            job['decrypted_metadata'] = metadata

            mimetype = job.get('mimetype')
            version = job.get('version')

            if mimetype and version:
                mimetype = mimetype.lower()
                if mimetype in ['application/pdf']:
                    # Download file
                    tmp_file = tempfile.TemporaryFile(mode='wb+')
                    try:
                        filesize = await self.__attachment_handler.download_decrypt_file(secret_key_str, version, tmp_file)
                        self.__logger.debug(f"Downloaded {filesize} bytes for file {filename}")
                    except:
                        tmp_file.close()
                        self.__logger.exception("Error downloading file")

                    # Process decrypted file
                    tmp_file.seek(0)
                    job['tmp_file'] = tmp_file

            # Pass content on to indexing
            await self.__indexing_queue.put(job)

            # Check if done with queued jobs
            if self.__intake_queue.qsize() == 0:
                self.__logger.debug("Triggering fetch of new batch of files for RAG (empty intake queue)")
                self.__event_fetch_jobs.set()


    async def __index_thread(self):
        while self.__context.stopping is False:
            job: Optional[FileInformation] = await self.__indexing_queue.get()
            if job is None:
                return  # Stopping

            tuuid = job['tuuid']
            user_id = job['user_id']
            metadata = job['decrypted_metadata']
            filename = metadata['nom']
            self.__logger.debug(f"Indexing file {filename} with RAG")

            tmp_file = job.get('tmp_file')
            if tmp_file:
                self.__logger.debug("Closing tmp file")
                tmp_file.close()
            else:
                # Nothing to do
                pass

            # Mark the file as processed
            producer = await self.__context.get_producer()
            command = {"tuuid": tuuid, "user_id": user_id}
            await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, "confirmRag", Constantes.SECURITE_PROTEGE)


    async def __query_batch_rag(self):
        producer = await self.__context.get_producer()
        response = await producer.command({"batch_size": 5}, Constantes.DOMAINE_GROS_FICHIERS, "leaseForRag", Constantes.SECURITE_PROTEGE)
        parsed = response.parsed
        if parsed['ok'] is not True:
            self.__logger.warning("Error retrieving batch of files for RAG: %s" % parsed)
        elif parsed.get('code') == 1:
            self.__logger.debug("No more files to process for RAG")
            return

        # Parse batch file and insert on queue per item
        leases = parsed['leases']
        secret_keys: list = parsed['secret_keys']

        self.__logger.debug(f"Received batch of files {len(leases)} for RAG indexing")

        for lease in leases:
            cle_id = lease['metadata']['cle_id']

            key = [k for k in secret_keys if k['cle_id'] == cle_id].pop()

            info: FileInformation = {
                'tuuid': lease['tuuid'],
                'user_id': lease['user_id'],
                'metadata': lease['metadata'],
                'mimetype': lease.get('mimetype'),
                'version': lease.get('version'),
                'key': key,
                'tmp_file': None,
                'decrypted_metadata': None
            }

            await self.__intake_queue.put(info)

        pass