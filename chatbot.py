import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pickle
import random
import json
from nltk.stem import WordNetLemmatizer
# stemmer = LancasterStemmer()
# with open('indian_restaurant.json') as file:
#     data = json.load(file)
# # Load data from pickle
# with open('data.pickle', 'rb') as f:
#     words, labels, model = pickle.load(f)

# def bag_of_words(s, words):
#     bag = [0 for _ in range(len(words))]
#     s_words = nltk.word_tokenize(s)
#     s_words = [stemmer.stem(word.lower()) for word in s_words]

#     for se in s_words:
#         for i, w in enumerate(words):
#             if w == se:
#                 bag[i] = 1
#     return np.array(bag)

# def chat_response(inp):
#     results = model.predict([bag_of_words(inp, words)])
#     print('results,,,,,,,,,,,,,,,,,,,,,,,',results)
#     results_index = np.argmax(results)
#     tag = labels[results_index]

#     for tg in data['indian_restaurant']:
#         print('111111111')
#         if tg['tag'] == tag:
#             responses = tg['responses']
#             print(responses)
#             return random.choice(responses)
# inp=input()
# chat_response(inp)


# Load JSON data
with open('indian_restaurant.json') as file:
    data = json.load(file)
print(repr(data['indian_restaurant']))
intents =data['indian_restaurant']

model_filename = 'chatbot_model.pkl'
vectorizer_filename = 'vectorizer.pkl'
# Load the model and vectorizer
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)




lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)


responses = {}
for intent in intents:
    responses[intent['tag']] = intent['responses']
def get_response(user_input):
    user_input = preprocess_text(user_input)
    X_test = loaded_vectorizer.transform([user_input])
    predicted_tag = loaded_model.predict(X_test)[0]
    return random.choice(responses[predicted_tag])

# Chat with the bot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    response = get_response(user_input)
    print("Bot:",response)