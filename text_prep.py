import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Some sample reviews
reviews = [
    "This movie was amazing and I loved it!",
    "Absolutely terrible experience, do not watch.",
    "The acting was okay, but the plot was amazing."
]

# Create the tokenizer
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(reviews)

# Turn sentences into sequences of numbers
sequences = tokenizer.texts_to_sequences(reviews)

print("Word Index Dictionary:")
print(tokenizer.word_index)
print("\nSentences turned into Numbers:")
for i, seq in enumerate(sequences):
    print(f"Review {i+1}: {seq}")
