from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
import hashlib
import io
from PIL import Image, PngImagePlugin, TiffImagePlugin
import base64
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature



def extract_signature_from_lsb(image_path, signature_length):
    # Load the image
    image = Image.open(image_path)
    image_data = list(image.getdata())
    
    # Extract the bits from the least significant bit of each pixel
    extracted_bits = []
    for pixel in image_data[:signature_length * 8]:
        extracted_bits.append(str(pixel[2] & 1))
    
    # Group the bits into bytes and convert back to binary data
    extracted_signature = int(''.join(extracted_bits), 2).to_bytes(signature_length, byteorder='big')
    
    return extracted_signature

def extract_signature_from_metadata(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Attempt to extract the signature based on the image format
    signature_base64 = None
    if image.format == 'PNG':
        signature_base64 = image.text.get("Signature")
    
    if signature_base64:
        # Decode the base64 string back to binary
        signature = base64.b64decode(signature_base64)
        return signature
    else:
        return None


def verify_signature(public_key, image_path, extracted_signature):
    # Generate a hash of the current image
    current_hash = generate_image_hash(image_path)
    
    # Decrypt the extracted signature using the public key
    try:
        # Verify the signature
        public_key.verify(
            extracted_signature,
            current_hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False
