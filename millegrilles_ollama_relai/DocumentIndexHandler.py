import asyncio
import datetime
import json
import logging
import pathlib
import tempfile
import math

from asyncio import TaskGroup
from typing import Optional, TypedDict

from pydantic import Field
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike, BaseRetriever
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfStreamError

from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_document_secrete, dechiffrer_bytes_secrete
from millegrilles_messages.messages import Constantes

from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext, OllamaInstance
from millegrilles_ollama_relai.Util import decode_base64_nopad


class FileInformation(TypedDict):
    tuuid: str
    user_id: str
    domain: str
    cuuids: Optional[list[str]]
    metadata: dict
    mimetype: Optional[str]
    version: Optional[dict]
    key: Optional[dict]
    decrypted_metadata: Optional[dict]
    tmp_file: Optional[tempfile.NamedTemporaryFile]

QUERY_BATCH_RAG_LEN = 30
# CONTEXT_LEN = 4096
# DOC_LEN = 1024
# DOC_OVERLAP = math.floor(DOC_LEN / 20)
# EMBEDDING_MODEL = "all-minilm"
# EMBEDDING_MODEL = "nomic-embed-text:137m-v1.5-fp16"

class DocumentIndexHandler:

    def __init__(self, context: OllamaContext, attachment_handler: AttachmentHandler):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__attachment_handler = attachment_handler

        self.__semaphore_db = asyncio.BoundedSemaphore(1)
        self.__event_fetch_jobs = asyncio.Event()

        self.__intake_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=QUERY_BATCH_RAG_LEN)
        self.__indexing_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=2)

        self.__vector_store_cache: dict[str, VectorStore] = dict()

    async def setup(self):
        pass

    async def trigger_indexing(self):
        self.__event_fetch_jobs.set()

    async def query_rag(self, message: MessageWrapper) -> Optional[dict]:
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

        rag_configuration = self.__context.rag_configuration
        if rag_configuration is None:
            raise Exception("No RAG configuration provided - configure it with MilleGrilles AiChat")

        query_model = rag_configuration['model_query_name']
        embedding_model = rag_configuration['model_embedding_name']
        context_len = rag_configuration.get('context_len')
        doc_chunk_len = rag_configuration.get('document_chunk_len')

        try:
            instance = self.__context.pick_ollama_instance(query_model)
        except IndexError:
            return {'ok': False, 'err': 'No ollama instance available'}

        async with instance.semaphore:
            vector_store = await asyncio.to_thread(self.open_vector_store, Constantes.DOMAINE_GROS_FICHIERS, user_id, instance, embedding_model)

            # Determine is we need to use a document filter
            if cuuid and cuuid != '':
                # Filter by directory
                retriever = FilteredCuuidsRetriever(vectorstore=vector_store.as_retriever(), cuuid=cuuid)
            else:
                # No filter
                retriever = vector_store.as_retriever()

            client = instance.get_async_client(self.__context.configuration)
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
                options={"temperature": 0.0, "num_ctx": context_len}
            )

        self.__logger.debug("Response: %s" % response['response'])

        response = {'ok': True, 'response': response['response'], 'ref': doc_ref}
        producer = await self.__context.get_producer()
        await producer.encrypt_reply([message.certificat], response,
                                     correlation_id=message.correlation_id,
                                     reply_to=message.reply_to)

        return None  # Already replied

    async def run(self):
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
            await self.__context.wait(300)  # Trigger once every 5 minutes

        # Unblock the query thread
        self.__event_fetch_jobs.set()

    async def __query_thread(self):
        while self.__context.stopping is False:
            self.__event_fetch_jobs.clear()
            if self.__intake_queue.qsize() < 2:
                # Try to fetch more items
                self.__logger.debug("Triggering fetch of new batch of files for RAG (low/empty intake queue)")
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
            # Trigger fetch
            self.__event_fetch_jobs.set()

            # Wait for jobs
            job = await self.__intake_queue.get()
            if job is None:
                return  # Stopping

            # Decrypt all content
            try:
                secret_key_str = job['key']['cle_secrete_base64']
                secret_key = decode_base64_nopad(secret_key_str)
                metadata = dechiffrer_document_secrete(secret_key, job['metadata'])
                filename = metadata['nom']
                job['decrypted_metadata'] = metadata

                mimetype = job.get('mimetype')
                version = job.get('version')
                fuuid = version['fuuid']
            except (TypeError, KeyError, UnicodeDecodeError) as e:
                self.__logger.warning(f"__intake_thread Error getting value for tuuid: {job['tuuid']}: {str(e)}, skipping")
            else:
                if mimetype and version:
                    mimetype = mimetype.lower()
                    if mimetype in ['application/pdf']:
                        # Download file
                        tmp_file = tempfile.NamedTemporaryFile(mode='wb+')
                        try:
                            # Combine version and key to ensure legacy decryption info is available
                            info_decryption = version.copy()
                            # info_decryption.update(job['key'])
                            key = job['key']
                            info_decryption['format'] = info_decryption.get('format') or key.get('format') or 'mgs4'  # Default format
                            nonce = info_decryption.get('nonce') or key.get('nonce')
                            if nonce is None:
                                header = info_decryption.get('header') or key.get('header')
                                nonce = header[1:]
                            info_decryption['nonce'] = nonce
                            try:
                                filesize = await self.__attachment_handler.download_decrypt_file(secret_key_str, info_decryption, tmp_file)
                                self.__logger.debug(f"Downloaded {filesize} bytes for file {filename}")
                            except* asyncio.CancelledError:
                                raise Exception(f"Error downloading fuuid {fuuid}, will retry")
                        except:
                            tmp_file.close()
                            self.__logger.exception(f"Error downloading fuuid {fuuid}, will retry")
                            continue  # Ignore this file for now, don't mark it processed

                        # Process decrypted file
                        tmp_file.seek(0)
                        job['tmp_file'] = tmp_file

            # Pass content on to indexing
            await self.__indexing_queue.put(job)

    async def __index_thread(self):
        while self.__context.stopping is False:
            job: Optional[FileInformation] = await self.__indexing_queue.get()
            if job is None:
                return  # Stopping

            tuuid = job['tuuid']
            user_id = job['user_id']
            domain = job['domain']
            cuuids = job.get('cuuids')
            metadata = job['decrypted_metadata']
            try:
                filename = metadata['nom']
                self.__logger.debug(f"Indexing file {filename} with RAG")
            except (TypeError, KeyError):
                filename = tuuid

            tmp_file = job.get('tmp_file')
            if tmp_file:
                try:
                    rag_configuration = self.__context.rag_configuration
                    if rag_configuration is None:
                        raise Exception("No RAG configuration provided - configure it with MilleGrilles AiChat")

                    embedding_model = rag_configuration['model_embedding_name']
                    instance = self.__context.pick_ollama_instance(embedding_model)
                    async with instance.semaphore:
                        vector_store = await asyncio.to_thread(self.open_vector_store, domain, user_id, instance, embedding_model)
                        await asyncio.to_thread(index_pdf_file, vector_store, tuuid, filename, cuuids, tmp_file)
                except (TypeError, ValueError, PdfStreamError) as e:
                    self.__logger.exception(f"Error processing file tuuid {tuuid}), rejecting: %s" % str(e))
                    pass  # The file will be marked as processed later on
                except:
                    self.__logger.exception(f"Error processing file tuuid {tuuid}, will retry")
                    continue
                finally:
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

        # Get the batch
        command = {"batch_size": batch_size, "filehost_id": filehost_id}
        try:
            response = await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, "leaseForRag",
                                              Constantes.SECURITE_PROTEGE, timeout=60)
        except asyncio.TimeoutError:
            self.__logger.warning("Timeout on leaseForRag, will retry")
            return

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
            metadata = lease['metadata']
            version = lease.get('version')
            cuuids = lease.get('cuuids')
            fuuid: Optional[str] = None
            if version:
                fuuid = version['fuuid']
            cle_id = metadata.get('cle_id') or metadata.get('ref_hachage_bytes') or fuuid

            try:
                key = [k for k in secret_keys if k['cle_id'] == cle_id].pop()
            except IndexError:
                self.__logger.warning(f"Missing key for fuuid {fuuid}, skipping")
                key = None

            info: FileInformation = {
                'tuuid': lease['tuuid'],
                'user_id': lease['user_id'],
                'domain': Constantes.DOMAINE_GROS_FICHIERS,
                'cuuids': cuuids,
                'metadata': metadata,
                'mimetype': lease.get('mimetype'),
                'version': lease.get('version'),
                'key': key,
                'tmp_file': None,
                'decrypted_metadata': None
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
