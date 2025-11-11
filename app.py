from __future__ import division, print_function
import os
import requests
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from dotenv import load_dotenv

# ==========================================================
# Load environment variables (for local and Render)
# ==========================================================
load_dotenv()

# ==========================================================
# Flask app setup
# ==========================================================
app = Flask(__name__)

# ==========================================================
# Model Configuration
# ==========================================================
MODEL_PATH = "AlexNet_Optimized.h5"
MODEL_URL = "https://huggingface.co/ajaykumar-6/apple_model/resolve/main/AlexNet_Optimized.h5"

# ==========================================================
# Function to download model from Hugging Face
# ==========================================================
def download_model():
    """Download the model from Hugging Face Hub if not present or corrupted."""
    hf_token = os.getenv("hf_token")  # Securely loaded token from environment
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        print("📥 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, headers=headers, stream=True)

        # Unauthorized (private repo + no token)
        if response.status_code == 401:
            raise PermissionError(
                "❌ Unauthorized: Your Hugging Face model is private. "
                "Please add your HF_TOKEN to Render environment variables."
            )

        # Not found (bad URL or filename)
        if response.status_code == 404:
            raise FileNotFoundError(
                "❌ Model not found. Please verify your MODEL_URL and filename on Hugging Face."
            )

        response.raise_for_status()

        # Ensure it’s an actual binary file, not HTML
        if b"<html" in response.content[:500]:
            raise ValueError("❌ Invalid file download — got HTML instead of a model file!")

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists locally. Skipping download.")


# ==========================================================
# Load Model Safely
# ==========================================================
try:
    download_model()
    print("🔍 Loading Model ...")
    model = load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# ==========================================================
# Class Labels
# ==========================================================
class_labels = [
    'Apple__black_rot',
    'Apple__healthy',
    'Apple__rust',
    'Apple__scab'
]

# ==========================================================
# Disease Information
# ==========================================================
disease_info = {
    'Apple__black_rot': {
        'precautions': [
            'Prune out dead or infected branches and mummified fruits.',
            'Avoid overhead irrigation and ensure good air circulation.',
            'Remove fallen leaves and debris to prevent fungal spread.'
        ],
        'fertilizers': [
            'Apply balanced NPK fertilizer (10-10-10).',
            'Use compost or organic manure for soil enrichment.'
        ],
        'pesticides': [
            'Use fungicides with Captan or Mancozeb.',
            'Spray during the early growing season.'
        ]
    },
    'Apple__rust': {
        'precautions': [
            'Remove nearby juniper plants (alternate rust hosts).',
            'Prune affected twigs early.',
            'Ensure proper spacing for airflow.'
        ],
        'fertilizers': [
            'Apply nitrogen-rich fertilizers to boost recovery.',
            'Use compost to improve soil health.'
        ],
        'pesticides': [
            'Use Myclobutanil or Propiconazole-based fungicides.',
            'Repeat every 10–14 days if infection persists.'
        ]
    },
    'Apple__scab': {
        'precautions': [
            'Remove infected leaves and fruits immediately.',
            'Avoid overhead watering.',
            'Use resistant varieties if available.'
        ],
        'fertilizers': [
            'Use potassium- and phosphorus-rich fertilizers.',
            'Avoid excessive nitrogen fertilizers.'
        ],
        'pesticides': [
            'Apply Sulfur or Mancozeb fungicides at bud break.',
            'Repeat during the growing season as needed.'
        ]
    },
    'Apple__healthy': {
        'precautions': ['Maintain regular pruning and tree hygiene.'],
        'fertilizers': ['Apply NPK fertilizer as per soil test results.'],
        'pesticides': ['No pesticide required; continue preventive care.']
    }
}

# ==========================================================
# Prediction Function
# ==========================================================
def model_predict(img_path, model):
    if model is None:
        return "Error: Model not loaded", "", "", 0, {}

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    predicted_label = class_labels[pred_idx]

    crop, disease = predicted_label.split('__')
    disease = disease.replace('_', ' ').title()
    confidence = round(float(preds[0][pred_idx]) * 100, 2)

    all_confidences = {
        cls.replace('__', ': ').replace('_', ' ').title(): round(float(p) * 100, 2)
        for cls, p in zip(class_labels, preds[0])
    }

    return predicted_label, crop, disease, confidence, all_confidences

# ==========================================================
# Routes
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if model is None:
        return "<div class='alert alert-danger'>Model not loaded properly. Check server logs.</div>", 500

    if 'file' not in request.files:
        return "<div class='alert alert-danger'>No file uploaded!</div>", 400

    f = request.files['file']
    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, secure_filename(f.filename))
    f.save(file_path)

    predicted_label, crop, disease, confidence, all_confidences = model_predict(file_path, model)

    if confidence < 50:
        return (
            "<div class='alert alert-warning text-center mt-3'>"
            "<h4>⚠️ The uploaded image does not appear to be an Apple leaf.</h4>"
            f"<p>Prediction confidence: {confidence}%</p>"
            "<p>Please upload a clear image of an Apple leaf.</p></div>"
        )

    info = disease_info.get(predicted_label, {})
    precautions = info.get('precautions', [])
    fertilizers = info.get('fertilizers', [])
    pesticides = info.get('pesticides', [])

    result = f"""
    <div class="card shadow-sm mt-4">
        <div class="card-body">
            <h3 class="text-success text-center">🍎 Apple Leaf Analysis Result</h3>
            <hr>
            <h4>🧬 Disease Detected: <span class="text-danger">{disease}</span></h4>
            <p><strong>Model Confidence:</strong> {confidence}%</p>
            <hr>
            <h5>📋 Precautions:</h5>
            <ul>{"".join(f"<li>{p}</li>" for p in precautions)}</ul>
            <h5>🌿 Recommended Fertilizers:</h5>
            <ul>{"".join(f"<li>{f}</li>" for f in fertilizers)}</ul>
            <h5>🧪 Recommended Pesticides:</h5>
            <ul>{"".join(f"<li>{p}</li>" for p in pesticides)}</ul>
            <hr>
            <h5>📊 All Class Probabilities:</h5>
            <ul>{"".join(f"<li>{cls}: {conf}%</li>" for cls, conf in all_confidences.items())}</ul>
        </div>
    </div>
    """
    return result


# ==========================================================
# Run App
# ==========================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
