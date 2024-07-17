from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import sys
import os
import logging
from PIL import Image
import io
import base64

# Ensure the correct module paths
current_dir = os.path.dirname(os.path.abspath(__file__))
face_reaging_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(face_reaging_dir)

root_dir = os.path.abspath(os.path.join(face_reaging_dir, os.pardir))
sys.path.append(root_dir)

from model.models import UNet
from scripts.test_functions import process_image

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(root_dir, "face_re-aging", "best_unet_model.pth")
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load(model_path, map_location=device))
unet_model.eval()

@app.route('/')
def index():
    return "Server is running"

@app.route('/process-image', methods=['POST'])
def process_image_route():
    data = request.json
    image_data = data['image']
    source_age = data['source_age']
    target_age = data['target_age']
    
    logging.debug(f"Received data: source_age={source_age}, target_age={target_age}")

    try:
        # Decode the image from base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        logging.debug("Image decoded successfully")

        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            logging.debug(f"Converting image mode from {image.mode} to RGB")
            image = image.convert('RGB')

        # Process the image
        processed_image = process_image(unet_model, image, video=False, source_age=source_age, target_age=target_age)

        logging.debug("Image processed successfully")

        # Convert the processed image to base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        logging.debug("Processed image encoded to base64")

        return jsonify({"processed_image": processed_image_base64})
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
