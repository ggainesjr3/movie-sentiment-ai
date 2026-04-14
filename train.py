import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing
import os

# --- Configuration ---
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250
BATCH_SIZE = 512
EPOCHS = 10

def train_model():
    # 1. Load the IMDb Dataset
    print("Step 1: Loading data...")
    # Words are already indexed by frequency; we only take the top 10,000
    (train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=VOCAB_SIZE)

    # 2. Preprocessing (Padding)
    # We pad/truncate all reviews to exactly 250 words so they fit the input layer
    print("Step 2: Padding sequences...")
    train_data = preprocessing.sequence.pad_sequences(train_data, maxlen=MAX_SEQUENCE_LENGTH)
    test_data = preprocessing.sequence.pad_sequences(test_data, maxlen=MAX_SEQUENCE_LENGTH)

    # 3. Build the Model Architecture
    print("Step 3: Building the model with Dropout layers...")
    model = models.Sequential([
        # Embedding: Turns word integers into 16-dimensional vectors
        layers.Embedding(VOCAB_SIZE, 16, input_length=MAX_SEQUENCE_LENGTH),
        
        # Dropout 1: Prevents the embedding from becoming too specific to training data
        layers.Dropout(0.2),
        
        # Pooling: Reduces dimensionality by averaging across the review length
        layers.GlobalAveragePooling1D(),
        
        # Dropout 2: Forces the dense layers to be more robust
        layers.Dropout(0.2),
        
        # Dense: Learns relationships between the averaged word vectors
        layers.Dense(16, activation='relu'),
        
        # Output: Sigmoid scales output between 0 (Negative) and 1 (Positive)
        layers.Dense(1, activation='sigmoid')
    ])

    # 4. Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 5. Training
    print("\nStep 4: Starting training...")
    history = model.fit(
        train_data, 
        train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # 20% for real-time performance monitoring
        verbose=1
    )

    # 6. Evaluation on Test Set
    print("\nStep 5: Evaluating on unseen test data...")
    results = model.evaluate(test_data, test_labels)
    print(f"\nFinal Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")

    # 7. Save for Inference
    model.save('movie_review_model.h5')
    print("Model successfully saved as 'movie_review_model.h5'")

if __name__ == "__main__":
    train_model()