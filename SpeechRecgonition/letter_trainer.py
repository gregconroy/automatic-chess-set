import neural_network as cnn
from mfcc_extractor import MFCCExtractor
import numpy as np
import time
from matplotlib import pyplot as plt

FOLDER = './Training Data/letters/'
MFCC_COEFFS = 100
MAX_FRAMES = 100
PIECES = ["a", "b", "c", "d", "e", "f", "g", "h"]

TRAIN_MODE = True

piece_speaker_info = {
    "greg": {
        "train range": (1, 100),
        "test range": (1, 2),
    },
    "liam": {
        "train range": (1, 20),
        "test range": (1, 2),
    },
    "mike": {
        "train range": (1, 20),
        "test range": (1, 2),
    }
}

weights = cnn.generateWeights(MFCC_COEFFS * MAX_FRAMES, 64, len(PIECES), 3)
mfcc_extractor = MFCCExtractor(num_coeffs=MFCC_COEFFS)

if not TRAIN_MODE:
    loaded = np.load('./CNN/letter_weights.npz')
    weights = [loaded[key] for key in loaded]

def create_training_set():
    sample_data = []
    expected_data = []

    # Loop through all speakers in piece_speaker_info
    for speaker, info in piece_speaker_info.items():
        train_range = info["train range"]
        
        for i in range(train_range[0], train_range[1] + 1):
            for piece in PIECES:
                path = f'{FOLDER}{speaker}_{piece}_{i}.wav'
                mfcc_feats = mfcc_extractor.extract(path, frames=MAX_FRAMES)
                sample_data.append(mfcc_feats)
                
                # Create expected output for the piece
                temp = [-1] * len(PIECES)
                temp[PIECES.index(piece)] = 1
                expected_data.append(temp)

    return np.array(sample_data), np.array(expected_data)

def create_testing_set():
    sample_data = []
    expected_data = []

    # Loop through all speakers in piece_speaker_info
    for speaker, info in piece_speaker_info.items():
        test_range = info["test range"]
        
        for i in range(test_range[0], test_range[1] + 1):
            for piece in PIECES:
                path = f'{FOLDER}{speaker}_{piece}_{i}.wav'
                mfcc_feats = mfcc_extractor.extract(path, frames=MAX_FRAMES)
                sample_data.append(mfcc_feats)
                
                # Create expected output for the piece
                temp = [0] * len(PIECES)
                temp[PIECES.index(piece)] = 1
                expected_data.append(temp)

    return np.array(sample_data), np.array(expected_data)

# Create training set
sample, expected = create_training_set()

if TRAIN_MODE:
    l2, weights = cnn.backProp(sample, expected, weights, 0.01, 500, 0.05, 1)

# Create testing set
test_samples, test_expected = create_testing_set()
now = time.time_ns()
outputs = cnn.feedForward(test_samples, weights)
print((time.time_ns() - now)/1e6)

# Evaluate the model
total = test_expected.shape[0]
summing = 0
correct = [0] * len(PIECES)

for i, j in zip(outputs, test_expected):
    summing += int(np.argmax(i) == np.argmax(j))
    
    expected = np.argmax(j)
    predicted = np.argmax(i)

    if expected == predicted:
        correct[expected] += 1

# Print the accuracy for each piece
for i in range(len(PIECES)):
    print(f'{PIECES[i]}: {100 * correct[i] / (piece_speaker_info["greg"]["test range"][1] - piece_speaker_info["greg"]["test range"][0])}%')

# Print overall performance
print(f'Overall performance: {summing/total*100}')

# Save the updated weights
np.savez('./CNN/letter_weights_new.npz', *weights)
