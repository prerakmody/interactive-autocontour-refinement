"""
This generates a self-signed certificate and private key for the server.
Why?
    - on windows, I noticed my queries became much faster when I used SSL. What?
    - afterwards on MacOS, on hitting /prepare I am now getting "Failed to load resource: net::ERR_SSL_PROTOCOL_ERROR"
        - after generating the certificate and private key, I get "POST https://localhost:55000/prepare net::ERR_CERT_AUTHORITY_INVALID"
        - then I added the certificate (on MAcOS) to the keychain and trusted it, but then I get "ERR_CERT_COMMON_NAME_INVALID"
    - On Unix
        - ??
"""
import sys
import datetime
from pathlib import Path

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import Encoding

DIR_FILE    = Path(__file__).resolve().parent # <root>/src/backend/utils
DIR_BACKEND = DIR_FILE.parent # <root>/src/backend
DIR_SRC     = DIR_BACKEND.parent # <root>/src/
DIR_ROOT    = DIR_SRC.parent # <root>/
DIR_KEYS    = DIR_BACKEND / "_keys"
DIR_KEYS2   = DIR_SRC / "frontend" / "_keys"
Path(DIR_KEYS).mkdir(parents=True, exist_ok=True)
Path(DIR_KEYS2).mkdir(parents=True, exist_ok=True)

pathPrivateKey = DIR_KEYS / "hostKey.pem"
pathCert = DIR_KEYS / "hostCert.pem"

def copy():
    # Copy DIR_KEYS to DIR_KEYS2
    try:
        import shutil
        shutil.copytree(DIR_KEYS, DIR_KEYS2, dirs_exist_ok=True)
        print(f" - Copied SSL keys to {DIR_KEYS2}")
    except Exception as e:
        print(f" - Error copying SSL keys to {DIR_KEYS2}: {e}")
        sys.exit(1)

def check_existing_certificates():
    """Check if valid certificates already exist"""
    
    if not pathPrivateKey.exists() or not pathCert.exists():
        print(f" - Certificate files missing. Private key exists: {pathPrivateKey.exists()}, Cert exists: {pathCert.exists()}")
        return False
    
    try:
        # Try to load and validate the existing certificate
        with open(pathCert, "rb") as f:
            cert_data = f.read()
            cert = x509.load_pem_x509_certificate(cert_data)
        
        # Check if certificate is still valid (not expired)
        now = datetime.datetime.now(datetime.UTC)
        # Use the new timezone-aware property
        cert_expiry = cert.not_valid_after_utc
        if cert_expiry < now:
            print(f" - Certificate expired on {cert_expiry}")
            return False
        
        print(f" - Valid certificate found, expires on {cert_expiry}")
        return True
        
    except Exception as e:
        print(f" - Error validating existing certificate: {e}")
        return False

# Check if we need to generate new certificates
if check_existing_certificates():
    print(" - SSL certificates already exist and are valid. Skipping generation.")
    copy()
    sys.exit(0)

print(" - Generating new SSL certificates...")

# Step 1 - Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Step 2 - Generate a self-signed certificate
subject = issuer = x509.Name([
    # x509.NameAttribute(NameOID.COMMON_NAME, u"interactive-server.py"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
])

# Use timezone-aware datetime objects
now = datetime.datetime.now(datetime.UTC)
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(private_key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now)
    .not_valid_after(now + datetime.timedelta(days=10000))
    .sign(private_key, hashes.SHA256())
)

# Step 3 - Write private key to file
print(f" - Writing private key to file: {pathPrivateKey}")
try:
    with open(pathPrivateKey, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    print(f" - Successfully wrote private key")
except Exception as e:
    print(f" - Error writing private key: {e}")
    sys.exit(1)

# Step 4 -  Write certificate to file
print(f" - Writing certificate to file: {pathCert}")
try:
    with open(pathCert, "wb") as f:
        f.write(cert.public_bytes(Encoding.PEM))
    print(f" - Successfully wrote certificate")
    # Use the new timezone-aware property
    print(f" - Certificate valid until: {cert.not_valid_after_utc}")
except Exception as e:
    print(f" - Error writing certificate: {e}")
    sys.exit(1)

print(" - SSL certificate generation completed successfully!")

