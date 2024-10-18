import numpy as np

# Assuming you have an HMM class implemented
from GMHMM import GMHMM  # Replace with your actual module that contains the GMHMM class
from mfcc_class import MFCCExtractor  # Replace with the actual path to MFCCExtractor class

# Step 1: Extract MFCC Features
def extract_mfcc_features(file_paths):
    all_mfccs = []
    for file_path in file_paths:
        mfcc = MFCCExtractor.extract(file_path)  # Extract MFCCs from each file
        all_mfccs.append(mfcc.T)  # Transpose to shape (samples, features)
    return np.concatenate(all_mfccs, axis=0)  # Concatenate MFCCs from all files

# Step 2: Initialize parameters for the GMHMM model
def initialize_gmhmm_parameters(num_states, num_mixtures, mfcc_dim):
    # Initialize transition probabilities with equal values
    transProb = np.ones((num_states, num_states)) / num_states
    
    # Initialize means, covariances, and weighting for Gaussian Mixtures
    means = np.random.rand(num_states, num_mixtures, mfcc_dim)  # Randomly initialized means
    covariances = np.random.rand(num_states, num_mixtures, mfcc_dim)  # Random covariances
    weighting = np.ones((num_states, num_mixtures)) / num_mixtures  # Equal weighting for each mixture

    # Initial state distribution (randomly initialized for now)
    initialState = np.ones(num_states) / num_states  # Equal probability for each state initially

    return transProb, means, covariances, weighting, initialState

# Step 3: Train the custom GMHMM using the extracted MFCCs
def train_hmm(mfccs, n_states, n_mixtures=3, n_iterations=100, threshold=1e-3, convergeThresh=1e-3):
    # Get the dimension of the MFCC features
    mfcc_dim = mfccs.shape[1]

    # Initialize the parameters for the GMHMM model
    transProb, means, covariances, weighting, initialState = initialize_gmhmm_parameters(n_states, n_mixtures, mfcc_dim)

    # Create an instance of the GMHMM with initialized parameters
    hmm_model = GMHMM(
        mixtures=n_mixtures,
        transProb=transProb,
        means=means,
        covariances=covariances,
        weighting=weighting,
        initialState=initialState
    )

    # Train HMM with the MFCCs
    hmm_model.train(mfccs, iterations=n_iterations, threshold=threshold, convergeThresh=convergeThresh)

    return hmm_model

# Step 4: Test the custom HMM on new data
def test_hmm(hmm_model, test_mfccs):
    log_likelihood = hmm_model.loglikelihood(test_mfccs)  # Ensure 'loglikelihood' is the correct method name
    return log_likelihood

# Example usage:

path = './Segments/pieces/'

# File paths (replace these with actual paths to your audio files)
train_files = [path + 'greg_king_1.wav', path + 'greg_king_2.wav']  # Training audio files
test_files = [path + 'greg_king_3.wav', path + 'greg_king_4.wav']  # Testing audio files

# Extract MFCC features from training and testing data
train_mfccs = extract_mfcc_features(train_files)
test_mfccs = extract_mfcc_features(test_files)

# Train HMM on the training data
hmm_model = train_hmm(train_mfccs, n_states=6, n_mixtures=3, n_iterations=100)

# Test the HMM on the test data
log_likelihood = test_hmm(hmm_model, test_mfccs)

# Output the result
print(f"Log-likelihood of test data: {log_likelihood}")
