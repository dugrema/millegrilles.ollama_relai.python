import binascii
import tiktoken


def decode_base64_nopad(value: str) -> bytes:
    value += "=" * ((4 - len(value) % 4) % 4)  # Padding
    value_bytes: bytes = binascii.a2b_base64(value)
    return value_bytes


def check_token_len(prompt: str):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    content_len = len(encoding.encode(prompt))
    return content_len
