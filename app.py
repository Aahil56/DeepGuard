import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from deepfake_model import load_ensemble_model
import os
import tempfile

# Define the paths to the weight files
weight_paths = [
    'weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36',
    'weights/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19',
    'weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29',
    'weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31',
    'weights/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37',
    'weights/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40',
    'weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23'
]

# Load the ensemble model
model = load_ensemble_model(weight_paths)

# Create a Flask application
app = Flask(__name__)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image):
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()
    
    is_deepfake = prediction > 0.5
    confidence = prediction if is_deepfake else 1 - prediction
    
    return is_deepfake, confidence

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_prediction = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 != 0:  # Process every 30th frame
            continue
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.sigmoid(output).item()
        
        total_prediction += prediction
    
    cap.release()
    
    if frame_count == 0:
        return False, 0
    
    avg_prediction = total_prediction / (frame_count // 30)
    is_deepfake = avg_prediction > 0.5
    confidence = avg_prediction if is_deepfake else 1 - avg_prediction
    
    return is_deepfake, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            is_deepfake, confidence = process_image(file.read())
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                is_deepfake, confidence = process_video(temp_file.name)
            os.unlink(temp_file.name)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        return jsonify({
            'is_deepfake': bool(is_deepfake),
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)