import asyncio
import logging
import pathlib
import tempfile
import math

from asyncio import TaskGroup
from typing import Optional, TypedDict, Union

from langchain_chroma import Chroma
from langchain_core.retrievers import RetrieverLike
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_document_secrete, dechiffrer_bytes_secrete
from millegrilles_messages.messages import Constantes

from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.Util import decode_base64_nopad


class FileInformation(TypedDict):
    tuuid: str
    user_id: str
    domain: str
    metadata: dict
    mimetype: Optional[str]
    version: Optional[dict]
    key: dict
    decrypted_metadata: Optional[dict]
    tmp_file: Optional[tempfile.NamedTemporaryFile]

CONTEXT_LEN = 4096
DOC_LEN = 1024
DOC_OVERLAP = math.floor(DOC_LEN / 20)
# EMBEDDING_MODEL = "all-minilm"
EMBEDDING_MODEL = "nomic-embed-text:137m-v1.5-fp16"

class DocumentIndexHandler:

    def __init__(self, context: OllamaContext, attachment_handler: AttachmentHandler):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__attachment_handler = attachment_handler

        self.__semaphore_db = asyncio.BoundedSemaphore(1)
        self.__event_fetch_jobs = asyncio.Event()

        self.__intake_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=10)
        self.__indexing_queue: asyncio.Queue[Optional[FileInformation]] = asyncio.Queue(maxsize=2)

    async def setup(self):
        pass

    async def trigger_indexing(self):
        self.__event_fetch_jobs.set()

    async def query_rag(self, message: MessageWrapper) -> Union[dict, bool]:
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
        query = query_bytes.decode('utf-8')
        user_id = message.certificat.get_user_id

        if user_id is None:
            return {'ok': False, 'code': 403, 'err': 'Access denied, no user_id in certificate'}

        vector_store = await asyncio.to_thread(self.open_vector_store, Constantes.DOMAINE_GROS_FICHIERS, user_id)
        retriever = vector_store.as_retriever()
        client = self.__context.get_async_client()
        async with self.__context.ollama_http_semaphore:
            # Calculate doc limit using 3 * context chars
            limit = math.floor(CONTEXT_LEN * 3 / DOC_LEN)
            prompt, doc_ref = await format_prompt(retriever, query, limit)
            self.__logger.debug(f"PROMPT (len:{len(prompt)})\n{prompt}")
            response = await client.generate(
                model="llama3.2:3b-instruct-q8_0",
                prompt=prompt,
                options={"temperature": 0.0, "num_ctx": CONTEXT_LEN}
            )

        response = {'ok': True, 'response': response['response'], 'ref': doc_ref}
        producer = await self.__context.get_producer()
        await producer.encrypt_reply([message.certificat], response,
                                     correlation_id=message.correlation_id,
                                     reply_to=message.reply_to)

        return False  # Already replied {'ok': True, 'response': response['response'], 'ref': doc_ref}

    async def run(self):
        async with TaskGroup() as group:
            group.create_task(self.__query_thread())
            group.create_task(self.__intake_thread())
            group.create_task(self.__index_thread())
            group.create_task(self.__trigger_fetch_interval())

    async def __trigger_fetch_interval(self):
        while self.__context.stopping is False:
            self.__event_fetch_jobs.set()
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
                    tmp_file = tempfile.NamedTemporaryFile(mode='wb+')
                    try:
                        filesize = await self.__attachment_handler.download_decrypt_file(secret_key_str, version, tmp_file)
                        self.__logger.debug(f"Downloaded {filesize} bytes for file {filename}")
                    except:
                        tmp_file.close()
                        self.__logger.exception("Error downloading file, will retry")
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
            metadata = job['decrypted_metadata']
            filename = metadata['nom']
            self.__logger.debug(f"Indexing file {filename} with RAG")

            tmp_file = job.get('tmp_file')
            if tmp_file:
                try:
                    vector_store = await asyncio.to_thread(self.open_vector_store, domain, user_id)
                    async with self.__context.ollama_http_semaphore:
                        await asyncio.to_thread(index_pdf_file, vector_store, tuuid, filename, tmp_file)
                except:
                    self.__logger.exception(f"Error processing file {filename} (tuuid {tuuid}), will retry")
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
                'domain': Constantes.DOMAINE_GROS_FICHIERS,
                'metadata': lease['metadata'],
                'mimetype': lease.get('mimetype'),
                'version': lease.get('version'),
                'key': key,
                'tmp_file': None,
                'decrypted_metadata': None
            }

            await self.__intake_queue.put(info)

        pass

    def open_vector_store(self, domain: str, user_id: str) -> VectorStore:
        # Collection name must be between 3 and 63 chars, truncate the user_id to the last 32 chars
        user_id_trunc = user_id[-32:]
        collection_name = f'{domain}_{user_id_trunc}'
        options = self.__context.get_client_options()
        base_url = options['host']
        del options['host']
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=base_url,
            # model="llama3.2:3b-instruct-q8_0",
            # num_gpu=0,  # Disable GPU to avoid swapping models on ollama
            client_kwargs=options,
        )

        configuration = self.__context.configuration
        path_db = pathlib.Path(configuration.dir_rag, "chroma_langchain_db")
        path_db.mkdir(exist_ok=True)

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(path_db),  # Where to save data locally, remove if not necessary
        )

        return vector_store


def index_pdf_file(vector_store: VectorStore, tuuid: str, filename: str, tmp_file: tempfile.NamedTemporaryFile):
    loader = PyPDFLoader(tmp_file.name, mode="single")
    document_list = loader.load()
    document = document_list[0]
    splitter = RecursiveCharacterTextSplitter(chunk_size=DOC_LEN, chunk_overlap=DOC_OVERLAP)
    chunks = splitter.split_documents([document])
    chunk_no = 1
    for chunk in chunks:
        doc_id = f"{tuuid}/{chunk_no}"
        chunk_no += 1
        chunk.id = doc_id
        chunk.metadata['source'] = filename
    vector_store.add_documents(list(chunks))


async def format_prompt(retriever: RetrieverLike, query: str, limit: int) -> (str, list[dict]):
    context_response = await retriever.ainvoke(query, k=limit)

    doc_ref = list()
    for elem in context_response:
        doc_ref.append({"id": elem.id, "page_content": elem.page_content, "metadata": dict(elem.metadata)})

    # Wrap the documents in <source /> tags with id
    context_tags = [f'<source id="{elem.id}">{elem.page_content}</source>' for elem in context_response]

    # Prompt source - open-webui (https://openwebui.com/)
    prompt = f"""
### Task:
Respond to the user query using the provided context, incorporating inline citations in the format [id] **only when the <source> tag includes an explicit id attribute** (e.g., <source id="1">).

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

### Example of Citation:
If the user asks about a specific topic and the information is found in a source with a provided id attribute, the response should include the citation like in the following example:
* "According to the study, the proposed method increases efficiency by 20% [abcd-01]."

### Output:
Provide a clear and direct response to the user's query, including inline citations in the format [id] only when the <source> tag with id attribute is present in the context.

<context>
{"\n".join(context_tags)}
</context>

<user_query>
{query}
</user_query>    
    """

    return prompt, doc_ref
