import binascii

def decode_base64_nopad(value: str) -> bytes:
    value += "=" * ((4 - len(value) % 4) % 4)  # Padding
    value_bytes: bytes = binascii.a2b_base64(value)
    return value_bytes