import asyncio
import binascii
import tempfile
from urllib.parse import urljoin

import aiohttp

from millegrilles_messages.messages import Constantes
from millegrilles_messages.chiffrage.DechiffrageUtils import get_decipher_cle_secrete
from millegrilles_ollama_relai.OllamaContext import OllamaContext


class AttachmentHandler:

    def __init__(self, context: OllamaContext):
        self.__context = context

    async def download_decrypt_file(self, decrypted_key: str, job: dict, tmp_file: tempfile.TemporaryFile) -> int:
        fuuid = job['fuuid']
        decrypted_key_bytes = decode_base64pad(decrypted_key)
        decipher = get_decipher_cle_secrete(decrypted_key_bytes, job)

        timeout = aiohttp.ClientTimeout(connect=5, total=600)
        connector = self.__context.get_tcp_connector()
        file_size = 0
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            session.verify = self.__context.tls_method != 'nocheck'

            await filehost_authenticate(self.__context, session)

            filehost_url = self.__context.filehost_url
            url_fichier = urljoin(filehost_url, f'filehost/files/{fuuid}')
            async with session.get(url_fichier) as resp:
                resp.raise_for_status()

                async for chunk in resp.content.iter_chunked(64*1024):
                    await asyncio.to_thread(tmp_file.write, decipher.update(chunk))
                    file_size += len(chunk)

        chunk = decipher.finalize()
        await asyncio.to_thread(tmp_file.write, chunk)
        file_size += len(chunk)

        return file_size

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
