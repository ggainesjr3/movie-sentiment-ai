import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Suppress TensorFlow logs to keep the terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 1. Load the "NLP Brain" and the Word Index
# Using load_model as discussed earlier
model = tf.keras.models.load_model('sentiment_model.h5')
word_index = imdb.get_word_index()

def predict_mood(custom_review):
    # Standardize the text: lowercase and split into words
    words = custom_review.lower().split()
    
    # The IMDb dataset uses specific offsets:
    # 0 = Padding, 1 = Start of sequence, 2 = Unknown (OOV), 3 = Unused
    sequence = []
    for w in words:
        index = word_index.get(w, 0) + 3 
        # Check if the word is within the 10,000 word vocabulary we trained on
        if index < 10000:
            sequence.append(index)
        else:
            sequence.append(2) # Assign to "Unknown" if out of bounds
    
    # Pad the sequence to 200 (must match the training length)
    padded = pad_sequences([sequence], maxlen=200)
    
    # 2. Get the prediction (verbose=0 hides the progress bar for a single guess)
    prediction = model.predict(padded, verbose=0)[0][0]
    
    # 3. Interpret the result
    if prediction > 0.5:
        return f"👍 POSITIVE ({prediction*100:.1f}%)"
    else:
        return f"👎 NEGATIVE ({(1-prediction)*100:.1f}%)"

# --- INTERACTIVE TESTING ---
if __name__ == "__main__":
    print("\n--- Sentiment Analysis Tool ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_text = input("Enter a movie review to analyze: ")
        if user_text.lower() == 'exit':
            break
            
        result = predict_mood(user_text)
        print(f"Analysis Result: {result}\n")
