import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load the IMDb Dataset (Top 10,000 words)
print("Downloading IMDb data...")
vocab_size = 10000
max_length = 200
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# 2. Pad sequences so they are all 200 numbers long
train_data = pad_sequences(train_data, maxlen=max_length)
test_data = pad_sequences(test_data, maxlen=max_length)

# 3. Build the Model
model = models.Sequential([
    # Embedding turns a word ID (e.g. 1) into a list of 16 numbers (a vector)
    layers.Embedding(vocab_size, 16, input_length=max_length),
    
    # This "flattens" the 2D data into 1D so the brain can process it
    layers.GlobalAveragePooling1D(),
    
    # Hidden layers to find patterns
    layers.Dense(16, activation='relu'),
    
    # Output layer: 1 unit with 'sigmoid' gives a probability between 0 and 1
    # 0 = Negative, 1 = Positive
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train
print("Training sentiment model...")
model.fit(train_data, train_labels, epochs=10, 
          validation_data=(test_data, test_labels), batch_size=512)

# 5. Save the NLP Brain
model.save('sentiment_model.h5')
print("Saved as sentiment_model.h5")
