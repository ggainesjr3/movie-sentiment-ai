import tensorflow as tf
from tensorflow.keras import datasets, preprocessing
import numpy as np

# 1. Load the word index to map text to integers
word_index = datasets.imdb.get_word_index()

def encode_review(text):
    # Standardize and tokenize
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    # Map tokens to integers, skipping words not in our top 10,000
    encoded = [word_index.get(word, 0) + 3 for word in tokens]
    encoded = [i if i < 10000 else 2 for i in encoded]
    return encoded

def predict_sentiment(review_text, model_path='movie_review_model.h5'):
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the input
    encoded = encode_review(review_text)
    padded = preprocessing.sequence.pad_sequences([encoded], maxlen=250)
    
    # Predict
    prediction = model.predict(padded)[0][0]
    
    sentiment = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\nReview: {review_text}")
    print(f"Sentiment: {sentiment} ({confidence*100:.2f}% confidence)")

# Test it out
user_review = "This movie was an absolute masterpiece. The acting was incredible!"
predict_sentiment(user_review)