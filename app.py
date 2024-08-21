from flask import Flask, render_template, request, jsonify
import json
import random
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Load JSON data and model
with open('indian_restaurant.json') as file:
    data = json.load(file)

# Extract intents and responses
intents = data['indian_restaurant']
responses = {intent['tag']: intent['responses'] for intent in intents}

# Load the trained model and vectorizer
with open('chatbot_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function (define if needed)
def preprocess_text(text):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

# Function to get chatbot response
def get_response(user_input):
    user_input = preprocess_text(user_input)
    X_test = loaded_vectorizer.transform([user_input])
    predicted_tag = loaded_model.predict(X_test)[0]
    return random.choice(responses[predicted_tag])

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint for chatbot
@app.route("/get_response", methods=["POST"])
def chatbot_response():
    user_input = request.json.get("message")
    response = get_response(user_input)
    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)