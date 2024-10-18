import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from my_mfcc_extractor import MFCCExtractor  # Import your MFCCExtractor class

# Configuration settings
TRAIN_TAKES = range(1, 41)
TEST_TAKES = range(1, 2)
PIECES = ["king", "queen", "bishop", "knight", "rook", "pawn"]
DATA_DIR = "./Segments/pieces/"
NUM_CLASSES = len(PIECES)
MAX_FRAMES = 100  # You can adjust this based on your data
N_COEFFS = 100  # Number of MFCC coefficients

def extract_mfcc_features(file_path):
    """Extract MFCC features from a given audio file."""
    if not os.path.isfile(file_path):
        print(f"Warning: File not found - {file_path}")
        return None
    mfcc_extractor = MFCCExtractor(n_coeffs=N_COEFFS)
    mfcc_feat = mfcc_extractor.extract(file_path)
    
    # Pad MFCC to ensure consistent input size
    if mfcc_feat.shape[0] < MAX_FRAMES:
        mfcc_feat = np.pad(mfcc_feat, ((0, MAX_FRAMES - mfcc_feat.shape[0]), (0, 0)), mode='constant')
    elif mfcc_feat.shape[0] > MAX_FRAMES:
        mfcc_feat = mfcc_feat[:MAX_FRAMES, :]  # Truncate if too long

    return mfcc_feat

def load_data(pieces, takes, data_dir):
    """Load and return the MFCC features and corresponding labels."""
    data = []
    labels = []

    for i, piece in enumerate(pieces):
        for take in takes:
            file_path = os.path.join(data_dir, f"greg_{piece}_{take}.wav")
            mfcc_feat = extract_mfcc_features(file_path)
            if mfcc_feat is not None:
                data.append(mfcc_feat)
                labels.append(i)  # Label is the index of the piece in PIECES

    return np.array(data), np.array(labels)

# Load training data
X_train, y_train = load_data(PIECES, TRAIN_TAKES, DATA_DIR)

# Load test data
X_test, y_test = load_data(PIECES, TEST_TAKES, DATA_DIR)

# Reshape data for CNN input (adding channel dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(MAX_FRAMES, N_COEFFS, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # NUM_CLASSES output units
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Function to predict the piece based on audio input
def predict_piece(audio_file):
    """Predict the chess piece based on the audio input."""
    mfcc_feat = extract_mfcc_features(audio_file)
    if mfcc_feat is None:
        return "Error: Could not extract features."

    # Reshape for model input
    mfcc_feat = mfcc_feat.reshape(1, MAX_FRAMES, N_COEFFS, 1)  # Add batch dimension

    # Make prediction
    prediction = model.predict(mfcc_feat)
    predicted_class = np.argmax(prediction)  # Get the index of the class with the highest probability
    predicted_piece = PIECES[predicted_class]  # Map index to class name

    print(f"Predicted piece: {predicted_piece}")
    return predicted_piece

correct = 0
total = 0

for piece in PIECES:
    for i in range(41, 51):
        # Example usage: Change 'path_to_audio_file.wav' to your audio input
        audio_file_path = f'./Segments/pieces/greg_{piece}_{i}.wav'  # Replace with your actual audio file path
        predicted_piece = predict_piece(audio_file_path)
        total += 1
        if predicted_piece == piece:
            correct += 1


print(correct / total)

# Optionally, save the trained model
# model.save('cnn_word_classifier.h5')  # Uncomment this line if you want to save the model
