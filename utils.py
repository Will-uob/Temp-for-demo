import base64
import io
from PIL import Image

def pillow_image_to_base64_string(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")