from flask import Flask, request, jsonify
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
import io
import requests
from PIL import Image

app = Flask(__name__)

# Load the Stable Diffusion pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)
pipe.enable_model_cpu_offload()

# Define API route for image enhancement
@app.route('/enhance', methods=['POST'])
def enhance_image():
    data = request.get_json()

    # Get image URL and prompt from the request body
    image_url = data.get('image_url')
    prompt = data.get('prompt')

    if not image_url or not prompt:
        return jsonify({"error": "Image URL and prompt are required."}), 400

    try:
        # Load the image from the provided URL
        response = requests.get(image_url)
        init_image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Generate the image using the model
        generated_image = pipe(prompt, image=init_image).images[0]

        # Save the generated image to a buffer
        buffer = io.BytesIO()
        generated_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Return the image as a response
        return jsonify({"message": "Image enhanced successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
