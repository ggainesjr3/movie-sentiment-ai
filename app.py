import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, preprocessing
import numpy as np

# Page config
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬")

@st.cache_resource
def load_model_and_index():
    # Load the trained model
    model = tf.keras.models.load_model('movie_review_model.h5')
    # Load the word index
    word_index = datasets.imdb.get_word_index()
    return model, word_index

model, word_index = load_model_and_index()

def predict_sentiment(text):
    # Preprocess text to match training format
    tokens = text.lower().split()
    encoded = [word_index.get(w, 0) + 3 for w in tokens]
    encoded = [i if i < 10000 else 2 for i in encoded]
    padded = preprocessing.sequence.pad_sequences([encoded], maxlen=250)
    
    # Run prediction
    prediction = model.predict(padded)[0][0]
    return prediction

# UI Elements
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type a movie review below to see if the AI thinks it's Positive or Negative.")

user_input = st.text_area("Enter your review here:", "I really enjoyed the cinematography, but the plot felt a bit rushed.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        score = predict_sentiment(user_input)
        
        # Display Result
        st.divider()
        if score > 0.5:
            st.success(f"### Positive Sentiment (Confidence: {score*100:.1f}%)")
            st.balloons()
        else:
            st.error(f"### Negative Sentiment (Confidence: {(1-score)*100:.1f}%)")