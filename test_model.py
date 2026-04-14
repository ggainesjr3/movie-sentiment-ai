import tensorflow as tf
from tensorflow.keras import datasets, preprocessing

# Load the saved model
model = tf.keras.models.load_model('movie_review_model.h5')

# Get the word index to translate your text into numbers
word_index = datasets.imdb.get_word_index()

def predict_review(text):
    # Prepare the text: lowercase and split
    tokens = text.lower().split()
    # Convert words to integers based on the IMDb index
    encoded = [word_index.get(w, 0) + 3 for w in tokens]
    # Pad to match the 250-word input the model expects
    padded = preprocessing.sequence.pad_sequences([encoded], maxlen=250)
    
    # Predict!
    prediction = model.predict(padded)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"\nReview: {text}")
    print(f"Prediction: {sentiment} ({prediction*100:.2f}%)")

# Try it out
predict_review("The cinematography was beautiful but the plot was a bit slow.")
predict_review("I absolutely loved this film, it was the best experience ever!")