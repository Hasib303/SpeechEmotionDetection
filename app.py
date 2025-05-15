from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import joblib

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
# CORS(app)  # Enable CORS for all routes

# Load your trained model
# Replace this with your actual model loading code
def load_model():
    with open('Emotion Detection Model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Extract features from audio
# Replace with your actual feature extraction logic
scaler = joblib.load("scaler.pkl")  # Make sure this matches what you saved during training

def extract_features(audio_file, n_mfcc=40, max_pad_len=174):
    """
    Extracts and standardizes MFCC features from an audio file.

    Returns:
    - A numpy array of shape (174, 40)
    """
    try:
        # Match training params: load 2 seconds from 0.5s offset
        y, sr = librosa.load(audio_file, sr=None, duration=2, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Pad or truncate to (n_mfcc, max_pad_len)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        # Transpose to shape (174, 40) like during training
        mfcc = mfcc.T

        # Standardize using the loaded scaler
        mfcc = mfcc.reshape(-1, mfcc.shape[1])  # (174, 40)
        mfcc_scaled = scaler.transform(mfcc)

        return mfcc_scaled  # shape: (174, 40)

    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return np.zeros((max_pad_len, n_mfcc)) 

model = load_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file temporarily
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)
    
    try:
        # Extract features
        features = extract_features(temp_path)
        
        # Preprocess features to match model input
        # NOTE: Update this according to your model's requirements
        # features = preprocess_features(features)
        
        # Make prediction using your model
        # NOTE: This is just a placeholder. Replace with actual prediction code.
        # prediction = model.predict(features)
        # predicted_class = np.argmax(prediction)
        # confidence = prediction[0][predicted_class]
        
        # For demonstration, returning random emotion
        # Replace with actual model inference
        features = np.expand_dims(features, axis=0)  # shape: (1, 174, 40)

        # Predict
        predictions = model.predict(features)  # shape: (1, num_classes)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Map to emotion label
        predicted_emotion = emotion_labels[predicted_class]

        
        emotion = emotion_labels[predicted_class]
        
        return jsonify({
            'emotion': emotion,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)