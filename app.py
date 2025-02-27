import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from waitress import serve
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load saved models
rf_model = pickle.load(open("rf_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
tree_model = pickle.load(open("tree_model.pkl", "rb"))
nn_model = pickle.load(open("nn_model.pkl", "rb"))
dnn_model = load_model("dnn_model.h5")

# Load the scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Function to extract features from a URL (You need to define this properly)
def extract_features(url):
    features = []

    # 1. URL Length
    features.append(len(url))

    # 2. Count of `.` in URL
    features.append(url.count('.'))

    # 3. Count of `-` in URL
    features.append(url.count('-'))

    # 4. Count of `@` in URL (phishing indicators)
    features.append(url.count('@'))

    # 5. Presence of `https`
    features.append(1 if "https" in url else 0)

    # 6. Count of numbers in URL
    features.append(sum(c.isdigit() for c in url))

    # 7. Count of special characters
    special_chars = ['#', '?', '=', '&', '%', '_']
    features.append(sum(url.count(c) for c in special_chars))

    # Add more extracted features to match 48 features

    # Ensure the number of features is exactly 48
    while len(features) < 48:
        features.append(0)  # Padding with zeros if needed

    return np.array(features).reshape(1, -1)


# API Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data['url']

        # Extract features
        features = extract_features(url)
        features_scaled = scaler.transform(features)

        # Get predictions from different models
        rf_pred = rf_model.predict(features_scaled)[0]
        svm_pred = svm_model.predict(features_scaled)[0]
        xgb_pred = xgb_model.predict(features_scaled)[0]
        tree_pred = tree_model.predict(features_scaled)[0]
        nn_pred = nn_model.predict(features_scaled)[0]
        dnn_pred = (dnn_model.predict(features_scaled) > 0.5).astype(int)[0][0]

        # Combine predictions (majority vote)
        predictions = [rf_pred, svm_pred, xgb_pred, tree_pred, nn_pred, dnn_pred]
        final_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0

        return jsonify({"url": url, "prediction": final_prediction})  # 0 = Legitimate, 1 = Phishing

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
