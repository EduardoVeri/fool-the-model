from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
import hashlib
import io
from PIL import Image, PngImagePlugin, TiffImagePlugin
import base64

def generate_rsa_keys():
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Generate public key
    public_key = private_key.public_key()

    # Serialize private key to save it
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Serialize public key to save it
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Return the keys as serialized PEMs
    return private_pem, public_pem


def generate_image_hash(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        image_hash = hashlib.sha256(image_data).hexdigest()
    return image_hash



def sign_hash(private_key, data_hash):
    signature = private_key.sign(
        data=data_hash.encode(),
        padding=padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        algorithm=hashes.SHA256()
    )
    return signature

def embed_signature_in_image(image_path, signature):
    # Load the image
    image = Image.open(image_path)
    image_data = list(image.getdata())

    # Convert the signature to a binary string
    signature_bits = ''.join(format(byte, '08b') for byte in signature)

    # Embed the signature bits into the least significant bits of the image pixels
    modified_image_data = []
    for i, pixel in enumerate(image_data):
        if i < len(signature_bits):
            modified_pixel = (pixel[0], pixel[1], (pixel[2] & ~1) | int(signature_bits[i]))
            modified_image_data.append(modified_pixel)
        else:
            modified_image_data.append(pixel)

    # Save the modified image
    modified_image = Image.new(image.mode, image.size)
    modified_image.putdata(modified_image_data)
    modified_image_path = image_path.split('.')[0] + '_signed.' + image_path.split('.')[1]
    modified_image.save(modified_image_path)

    return modified_image_path

def add_signature_to_metadata(original_image_path, signature):
    # Convert the signature to a base64 string for embedding
    signature_base64 = base64.b64encode(signature).decode('utf-8')
    
    # Load the original image
    image = Image.open(original_image_path)
    
    # Depending on the image format, the approach to embedding metadata differs
    if image.format == 'JPEG':
        # For JPEG images, we can use the piexif library to handle EXIF data
        # This example assumes you have a way to modify the EXIF data accordingly
        # This is a placeholder for the process as manipulating EXIF data
        # directly for JPEGs can be complex and might require an additional library like piexif
        pass
    elif image.format == 'PNG':
        # For PNG images, we can directly add to the text chunks
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Signature", signature_base64)
        image.save(original_image_path.split('.')[0] + '_signed.png', "PNG", pnginfo=meta)
    else:
        # Other formats can be handled here
        pass

    return original_image_path.split('.')[0] + '_signed.' + image.format.lower()