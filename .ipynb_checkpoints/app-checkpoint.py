from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask App
app = Flask(__name__)

# Define API endpoint for spam detection
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get input data (JSON)
    message = data['message']  # Extract message text

    # Convert text into TF-IDF features
    processed_message = tfidf.transform([message]).toarray()

    # Make prediction using the model
    prediction = model.predict(processed_message)[0]

    # Convert prediction to readable format
    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"message": message, "prediction": result})  # Return JSON response

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
