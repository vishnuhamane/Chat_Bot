import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import json
import pickle
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load JSON data
with open('indian_restaurant.json') as file:
    data = json.load(file)
print(repr(data['indian_restaurant']))
intents =data['indian_restaurant']


# Preprocessing
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

# Prepare training data
training_sentences = []
training_labels = []

labels = []
responses = {}

for intent in intents:
    for pattern in intent['patterns']:
        training_sentences.append(preprocess_text(pattern))
        training_labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

# Create and train the model
model = LogisticRegression()
model.fit(X, training_labels)


# Save the model and vectorizer
model_filename = 'chatbot_model.pkl'
vectorizer_filename = 'vectorizer.pkl'

with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
