import asyncio
import binascii
import logging
import tempfile

from urllib.parse import urljoin
from typing import Optional

import aiohttp

from millegrilles_messages.messages import Constantes
from millegrilles_messages.chiffrage.DechiffrageUtils import get_decipher_cle_secrete
from millegrilles_ollama_relai.OllamaContext import OllamaContext

CONST_MAX_RETRY = 3

class AttachmentHandler:

    def __init__(self, context: OllamaContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context

        self.__session: Optional[aiohttp.ClientSession] = None

    async def download_decrypt_file(self, decrypted_key: str, job: dict, tmp_file: tempfile.TemporaryFile) -> int:
        fuuid = job['fuuid']
        decrypted_key_bytes = decode_base64pad(decrypted_key)
        decipher = get_decipher_cle_secrete(decrypted_key_bytes, job)

        file_size = 0

        session = self.__session
        for i in range(0, CONST_MAX_RETRY):
            if self.__session is None:
                timeout = aiohttp.ClientTimeout(connect=5, total=600)
                connector = self.__context.get_tcp_connector()
                session = aiohttp.ClientSession(timeout=timeout, connector=connector)
                session.verify = self.__context.tls_method != 'nocheck'
                try:
                    await filehost_authenticate(self.__context, session)
                    self.__session = session
                except aiohttp.ClientResponseError:
                    self.__logger.exception("Error authenticating")
                    await self.__context.wait(2)
                    continue  # Retry

            filehost_url = self.__context.filehost_url
            url_fichier = urljoin(filehost_url, f'filehost/files/{fuuid}')
            try:
                tmp_file.seek(0)  # Ensure we are at the beginning in case of client issue later on
                async with session.get(url_fichier) as resp:
                    resp.raise_for_status()

                    async for chunk in resp.content.iter_chunked(64*1024):
                        await asyncio.to_thread(tmp_file.write, decipher.update(chunk))
                        file_size += len(chunk)

                    # Download successful
                    chunk = decipher.finalize()
                    await asyncio.to_thread(tmp_file.write, chunk)
                    file_size += len(chunk)

                    return file_size
            except aiohttp.ClientResponseError as cre:
                if cre.status in [400, 401, 403]:
                    self.__logger.debug("Not authenticated")

                    # Close session
                    session = self.__session
                    self.__session = None
                    if session:
                        await session.close()

                    continue  # Retry with a new session
                elif 500 <= cre.status < 600:
                    self.__logger.info(f"Filehost server error: {cre.status}")
                    await self.__context.wait(3)  # Wait in case the server is restarting
                    continue  # Retry
                else:
                    raise cre

        raise Exception("Attached file download - Too many retries")

    async def prepare_image(self, file_size: int, tmp_file: tempfile.TemporaryFile) -> bytes:
        tmp_file.seek(0)
        content = await asyncio.to_thread(tmp_file.read)
        # content = binascii.b2a_base64(content).decode('utf-8')
        return content

    async def prepare_raw(self, tmp_file: tempfile.TemporaryFile) -> bytes:
        tmp_file.seek(0)
        content = await asyncio.to_thread(tmp_file.read)
        return content

async def filehost_authenticate(context: OllamaContext, session: aiohttp.ClientSession):
    filehost_url = context.filehost_url
    url_authenticate = urljoin(filehost_url, '/filehost/authenticate')
    authentication_message, message_id = context.formatteur.signer_message(
        Constantes.KIND_COMMANDE, dict(), domaine='filehost', action='authenticate')
    authentication_message['millegrille'] = context.formatteur.enveloppe_ca.certificat_pem
    async with session.post(url_authenticate, json=authentication_message) as resp:
        resp.raise_for_status()


def decode_base64pad(value: str) -> bytes:
    value += "=" * ((4 - len(value) % 4) % 4)  # Padding
    key_bytes: bytes = binascii.a2b_base64(value)
    return key_bytes
