import streamlit as st
import time
from src.file_crypto import (
    generate_ec_keypair,
    serialize_public_key,
    serialize_private_key,
    encrypt_bytes_ecies,
    decrypt_bytes_ecies,
    load_public_key,
    load_private_key,
)
from demos.attacks.tamper import flip_byte_bytes

st.set_page_config(page_title="SecureFileEC Demo", layout="centered")
st.title("SecureFileEC - ECIES Hybrid Encryption Demo")


# ------------------------------------------------------
# INITIAL KEY GENERATION IF NOT PRESENT
# ------------------------------------------------------
if "priv_pem" not in st.session_state:
    priv, pub = generate_ec_keypair()
    st.session_state["priv_pem"] = serialize_private_key(priv)
    st.session_state["pub_pem"] = serialize_public_key(pub)


# ------------------------------------------------------
# SIDEBAR - KEY MANAGEMENT
# ------------------------------------------------------
st.sidebar.header("Key Management")

# Show current keys
st.sidebar.subheader("Active Public Key (PEM)")
st.sidebar.code(st.session_state["pub_pem"].decode()[:300] + "...")

st.sidebar.subheader("Active Private Key (PEM)")
st.sidebar.code(st.session_state["priv_pem"].decode()[:300] + "...")

# Download buttons
st.sidebar.download_button(
    "Download Public Key",
    st.session_state["pub_pem"],
    file_name="my_public_key.pem"
)
st.sidebar.download_button(
    "Download Private Key",
    st.session_state["priv_pem"],
    file_name="my_private_key.pem"
)

# Upload own keys
st.sidebar.subheader("Upload Your Own Keys")

uploaded_pub = st.sidebar.file_uploader("Upload Public Key (PEM)", type=["pem"])
if uploaded_pub:
    st.session_state["pub_pem"] = uploaded_pub.read()
    st.sidebar.success("Public key replaced successfully.")

uploaded_priv = st.sidebar.file_uploader("Upload Private Key (PEM)", type=["pem"])
if uploaded_priv:
    st.session_state["priv_pem"] = uploaded_priv.read()
    st.sidebar.success("Private key replaced successfully.")

# Regenerate new keypair
if st.sidebar.button("Generate Fresh ECC Keypair"):
    priv, pub = generate_ec_keypair()
    st.session_state["priv_pem"] = serialize_private_key(priv)
    st.session_state["pub_pem"] = serialize_public_key(pub)
    st.sidebar.success("New keypair generated.")


# ------------------------------------------------------
# MAIN UI – FILE ENCRYPTION
# ------------------------------------------------------
uploaded = st.file_uploader("Upload a file to encrypt", type=None)

if uploaded:
    data = uploaded.read()
    st.write(f"Uploaded file size: {len(data)} bytes")

    if st.button("Encrypt File"):
        t0 = time.time()
        pkg = encrypt_bytes_ecies(data, st.session_state["pub_pem"])
        st.session_state["pkg"] = pkg
        st.success(f"Encrypted in {time.time() - t0:.4f}s")
        st.download_button("Download Encrypted File", pkg, file_name=f"{uploaded.name}.sfec")


# ------------------------------------------------------
# TAMPERING AND DECRYPTION
# ------------------------------------------------------
if "pkg" in st.session_state:

    st.subheader("Tampering Simulation")

    offset = st.number_input(
        "Byte offset to tamper",
        min_value=0,
        max_value=len(st.session_state["pkg"]) - 1,
        value=30
    )

    if st.button("Create Tampered File"):
        tampered = flip_byte_bytes(st.session_state["pkg"], offset)
        st.session_state["tampered"] = tampered
        st.download_button(
            "Download Tampered File",
            tampered,
            file_name="tampered_file.sfec"
        )
        st.success("Tampered file created.")

    if st.button("Decrypt Original File"):
        try:
            t0 = time.time()
            pt = decrypt_bytes_ecies(st.session_state["pkg"], st.session_state["priv_pem"])
            st.success(f"Decrypted in {time.time() - t0:.4f}s")
            st.download_button("Download Decrypted File", pt, file_name="decrypted_output")
        except Exception as e:
            st.error(f"Decryption Failed: {e}")

    if "tampered" in st.session_state and st.button("Decrypt Tampered File"):
        try:
            decrypt_bytes_ecies(st.session_state["tampered"], st.session_state["priv_pem"])
            st.warning("Unexpected: tampered file decrypted!")
        except Exception as e:
            st.error(f"Expected Failure (Tampered): {e}")
