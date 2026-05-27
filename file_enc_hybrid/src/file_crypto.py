# src/file_crypto.py
import os
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption,
    load_pem_public_key, load_pem_private_key
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b'SFEC1'

def generate_ec_keypair():
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    return priv, pub

def serialize_public_key(pub):
    return pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

def serialize_private_key(priv):
    return priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())

def load_public_key(pub_bytes):
    return load_pem_public_key(pub_bytes)

def load_private_key(priv_bytes):
    return load_pem_private_key(priv_bytes, password=None)

def derive_aes_key(shared_secret):
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'SecureFileEC')
    return hkdf.derive(shared_secret)

def encrypt_bytes_ecies(plaintext, receiver_pub_bytes):
    receiver_pub = load_public_key(receiver_pub_bytes)
    eph_priv = ec.generate_private_key(ec.SECP256R1())
    eph_pub = eph_priv.public_key()
    eph_pub_bytes = serialize_public_key(eph_pub)
    shared = eph_priv.exchange(ec.ECDH(), receiver_pub)
    aes_key = derive_aes_key(shared)
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return MAGIC + len(eph_pub_bytes).to_bytes(2, "big") + eph_pub_bytes + nonce + ct

def decrypt_bytes_ecies(package, receiver_priv_bytes):
    if not package.startswith(MAGIC):
        raise ValueError("Invalid file header")
    idx = len(MAGIC)
    ephem_len = int.from_bytes(package[idx:idx+2], "big"); idx += 2
    ephem_pub_bytes = package[idx:idx+ephem_len]; idx += ephem_len
    nonce = package[idx:idx+12]; idx += 12
    ct = package[idx:]
    eph_pub = load_public_key(ephem_pub_bytes)
    receiver_priv = load_private_key(receiver_priv_bytes)
    shared = receiver_priv.exchange(ec.ECDH(), eph_pub)
    aes_key = derive_aes_key(shared)
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ct, None)
