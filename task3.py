import nltk
import numpy as np
import random
import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('punkt')

# Define a set of intents and responses
template_intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello there!", "Hi, how can I help you today?", "Hey! What can I do for you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you later!", "Have a great day!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful"],
            "responses": ["You're welcome!", "Happy to help!", "Any time!"]
        },
        {
            "tag": "age",
            "patterns": ["How old are you?", "What is your age?"],
            "responses": ["I'm a bot created in 2025."]
        },
        {
            "tag": "name",
            "patterns": ["What is your name?", "Who are you?"],
            "responses": ["I'm your friendly NLP chatbot."]
        }
    ]
}

# Save intents to file
with open('intents.json', 'w') as f:
    json.dump(template_intents, f, indent=4)

# Load intents
with open('intents.json') as f:
    data = json.load(f)

# Prepare training data
corpus = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        labels.append(intent['tag'])

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=True)
X = vectorizer.fit_transform(corpus)
y = np.array(labels)

# Train classifier
clf = LogisticRegression()
clf.fit(X, y)

# Chat function
def chatbot_response(text):
    # Clean input
    text = text.translate(str.maketrans('', '', string.punctuation))
    features = vectorizer.transform([text])
    tag = clf.predict(features)[0]
    # Choose random response
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Main loop
if __name__ == "__main__":
    print("NLP Chatbot is running! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Bot: {response}")
