import requests
from bs4 import BeautifulSoup
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# URL of the OUSL website page
url = 'https://ou.ac.lk/fengtec/'

# Send an HTTP GET request to the URL
response = requests.get(url)

# Parse the HTML content of the webpage
soup = BeautifulSoup(response.text, 'html.parser')

# Find and extract text from the webpage
text = soup.get_text()

# Tokenize and lemmatize the text
words = nltk.word_tokenize(text)
words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]

# Save the extracted words to a file named 'words.pkl'
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

print('Text has been extracted and saved to words.pkl')
