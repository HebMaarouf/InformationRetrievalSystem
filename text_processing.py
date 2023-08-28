import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from num2words import num2words
from tashaphyne.stemming import ArabicLightStemmer
import unicodedata


# Initialize NLTK's Word Net Lemmatizer
lemmatizer = WordNetLemmatizer()

# Create an instance of the ArabicLightStemmer
arabic_lemmatizater = ArabicLightStemmer()

def process_english_text(data):
     # Convert numeric tokens to words
    converted_tokens = convert_numbers_to_words_and_lower_case(data)

    # normalize the document contents
    normalized_text = normalize_english_text(converted_tokens)

    # tokenize the document contents
    tokens = tokenize(normalized_text)
    
    # remove stop words 
    filtered_tokens = remove_english_stop_words(tokens)

    # apply lemmatization to the tokens
    lemmatized_tokens = english_lemmatization(filtered_tokens)

    # print("Original Text: ",data)
    # print("--------")
    # print("Converted Tokens: ",converted_tokens)
    # print("--------")
    # print("Normalized Text: ",normalized_text)
    # print("--------")
    # print("Tokens: ",tokens)
    # print("--------")
    # print("Filtered Tokens: ",filtered_tokens)
    # print("--------")
    # print("Lemmatized Tokens: ",lemmatized_tokens)
    # print("--------\n-----------\n--------")

    return lemmatized_tokens


def process_arabic_text(data):
     # Convert numeric tokens to words
    converted_tokens = convert_numbers_to_words_and_lower_case(data,language='ar')

    # normalize the document contents
    normalized_text = normalize_arabic_text(converted_tokens)

    # tokenize the document contents
    tokens = tokenize(normalized_text)
    
    # remove stop words 
    filtered_tokens = remove_arabic_stop_words(tokens)

    # apply lemmatization to the tokens
    lemmatized_tokens = arabic_lemmatization(filtered_tokens)

    # print("Original Text: ",data)
    # print("--------")
    # print("Converted Tokens: ",converted_tokens)
    # print("--------")
    # print("Normalized Text: ",normalized_text)
    # print("--------")
    # print("Tokens: ",tokens)
    # print("--------")
    # print("Filtered Tokens: ",filtered_tokens)
    # print("--------")
    # print("Lemmatized Tokens: ",lemmatized_tokens)
    # print("--------\n-----------\n--------")

    return lemmatized_tokens

def tokenize(data):
    return word_tokenize(data)

def normalize_english_text(data):
    # remove punctuation and special characters from the data, keep only words and whitespace.
    return re.sub(r'[^\w\s]', '', data)

def normalize_arabic_text(data):
    # remove diacritics
    normalized_text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u0656-\u065F]', '', data)
    # normalize arabic letters
    normalized_text = unicodedata.normalize('NFKD', normalized_text)
    # remove punctuation and special characters from the data, keep only words and whitespace.
    return re.sub(r'[^\w\s]', '', data)

def remove_english_stop_words(data):
    english_stopwords = set(stopwords.words('english'))
    return [token for token in data if token.lower() not in english_stopwords]

def remove_arabic_stop_words(data):
    arabic_stopwords = set(stopwords.words('arabic'))
    return [token for token in data if token.lower() not in arabic_stopwords]

def english_lemmatization(data):
    return [lemmatizer.lemmatize(filtered_token) for filtered_token in data]

def arabic_lemmatization(data):
    return [arabic_lemmatizater.light_stem(token) for token in data]

def convert_numbers_to_words_and_lower_case(data,language="en"):
    tokenized = tokenize(data)
    converted_tokens = []
    for token in tokenized:
        # if the token is a number, change it to be written (2 => two), (٢ => اثنين)
        if token.isdigit():
            try:
                converted_token = num2words(int(token),lang=language)
                new_converted_token = np.char.replace(converted_token, "-", " ")
                converted_tokens.append(new_converted_token)
            except:
                converted_tokens.append(token)
        else:
            new_converted_token = np.char.replace(token, "-", " ")
            converted_tokens.append(new_converted_token)
    return ' '.join(str(old_value).lower() for old_value in converted_tokens)