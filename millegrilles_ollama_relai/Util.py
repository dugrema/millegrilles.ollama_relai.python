import asyncio
import binascii
import tempfile

import tiktoken

from PIL import Image

from millegrilles_messages.messages.Hachage import hacher


def decode_base64_nopad(value: str) -> bytes:
    value += "=" * ((4 - len(value) % 4) % 4)  # Padding
    value_bytes: bytes = binascii.a2b_base64(value)
    return value_bytes


def check_token_len(prompt: str):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    content_len = len(encoding.encode(prompt))
    return content_len


def model_name_to_id(name: str) -> str:
    """
    :param name: Model name
    :return: A 16 char model id
    """
    return hacher(name.lower(), hashing_code='blake2s-256')[-16:]

def cleanup_json_output(content: str):
    try:
        if content[0] == '`':
            return content.replace('```json', '').replace('```', '').strip()
    except IndexError:
        pass  # Empty
    return content


async def conditional_convert_to_png(mimetype: str, tmp_file: tempfile.TemporaryFile):
    if mimetype not in ['image/png', 'image/jpg', 'image/jpeg']:
        # Convert to PNG, overwrite tmp file
        im = await asyncio.to_thread(Image.open, tmp_file)
        tmp_file.seek(0)  # Will overwrite with PNG
        await asyncio.to_thread(im.save, tmp_file, "png")
        tmp_file.truncate()
        tmp_file.seek(0)
