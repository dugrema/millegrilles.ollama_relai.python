import asyncio
import binascii
import tempfile
import base64
from typing import Optional

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


IMG_SIDE_MAX = 800

async def conditional_convert_to_png(mimetype: str, tmp_file: tempfile.TemporaryFile, file_len: Optional[int] = None):
    # Check that the file is in a supported file format
    must_convert = mimetype not in ['image/png', 'image/jpg', 'image/jpeg', 'image/webp']

    if not must_convert:
        # Check if file is large
        must_convert = file_len is not None and file_len > 512 * 1024

    if must_convert:
        # Convert to PNG, overwrite tmp file
        im = await asyncio.to_thread(Image.open, tmp_file)

        # Check dimensions, the file will be reduced in size when larger than limit
        width = im.width
        height = im.height
        resize = False

        if width > height and width > IMG_SIDE_MAX:
            width = IMG_SIDE_MAX
            height = int(IMG_SIDE_MAX * height / width)
            resize = True
        elif height > width and height > IMG_SIDE_MAX:
            height = IMG_SIDE_MAX
            width = int(IMG_SIDE_MAX * width / height)
            resize = True

        if resize:
            im.resize((width, height))

        tmp_file.seek(0)  # Will overwrite with PNG
        await asyncio.to_thread(im.save, tmp_file, "png")
        tmp_file.truncate()
        tmp_file.seek(0)


def encode_image_to_data_uri(temp_file: tempfile.TemporaryFile, mimetype: str = 'image/png'):
    temp_file.seek(0)
    encoded_string = base64.b64encode(temp_file.read()).decode('utf-8')
    temp_file.seek(0)
    return f'data:{mimetype};base64,{encoded_string}'
