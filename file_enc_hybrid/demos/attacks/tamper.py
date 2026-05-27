# demos/attacks/tamper.py
from pathlib import Path

def flip_byte_bytes(data: bytes, offset: int = 10) -> bytes:
    b = bytearray(data)
    b[offset] ^= 0x01
    return bytes(b)
