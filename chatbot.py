import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Load intents JSON file
intents = json.loads(open('C:/python/create_chatbot_using_python/intents.json').read())

# Load words and classes from the pickle files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    corrected_words = [spell.correction(word) for word in sentence_words]
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in corrected_words if word]  # Only lemmatize non-None words
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Debug statements to check the intermediate values
    print(f"Bag of words: {bow}")
    print(f"Model prediction: {res}")
    print(f"Filtered results: {results}")

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        try:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        except IndexError:
            print(f"IndexError: {r} is out of bounds for classes with length {len(classes)}")
    
    # Check if return_list is empty and handle the case
    if not return_list:
        return_list.append({"intent": "no_match", "probability": "0"})
    
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! Bot is running!")

while True:
    message = input("You: ")
    ints = predict_class(message, model)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
