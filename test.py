import numpy as np
from hmmlearn import hmm
from pydub import AudioSegment
from python_speech_features import mfcc
import os

# Class to handle all HMM-related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = {}
    
    def train(self, X, piece_name):
        model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
        model.fit(X)
        self.models[piece_name] = model

    def get_likelihood_scores(self, input_data):
        scores = {piece: model.score(input_data) for piece, model in self.models.items()}
        return scores

# Function to extract MFCC features from MP3 files
def extract_mfcc_features(file_path):
    audio = AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples = samples / (2**15)  # Normalize for 16-bit audio
    return mfcc(samples, audio.frame_rate, nfft=1200)

# Main training and testing function
def main():
    pieces = ["king", "queen", "bishop", "knight", "rook", "pawn"]
    hmm_trainer = HMMTrainer(n_components=5)

    # Collect training data and train the model for each piece
    for piece in pieces:
        train_features = []
        for take in range(1, 6):
            file_path = f"./segments/greg_{piece}_{take}.mp3"
            mfcc_feat = extract_mfcc_features(file_path)
            train_features.append(mfcc_feat)
        X_train = np.vstack(train_features)
        hmm_trainer.train(X_train, piece)

        for take in range(1, 2):
            file_path = f"./segments/mike_{piece}_{take}.mp3"
            mfcc_feat = extract_mfcc_features(file_path)
            train_features.append(mfcc_feat)
        X_train = np.vstack(train_features)
        hmm_trainer.train(X_train, piece)

        for take in range(1, 2):
            file_path = f"./segments/dewald_{piece}_{take}.mp3"
            mfcc_feat = extract_mfcc_features(file_path)
            train_features.append(mfcc_feat)
        X_train = np.vstack(train_features)
        hmm_trainer.train(X_train, piece)

        for take in range(1, 4):
            file_path = f"./segments/rox_{piece}_{take}.mp3"
            mfcc_feat = extract_mfcc_features(file_path)
            train_features.append(mfcc_feat)
        X_train = np.vstack(train_features)
        hmm_trainer.train(X_train, piece)

    # Initialize counters for summary
    total_tests = 0
    correct_predictions = 0

    # Test the model with takes 6 and 7 for each piece
    for piece in pieces:
        for take in range(1, 3):
            test_file_path = f"./segments/dewald_{piece}_{take}.mp3"
            mfcc_test_feat = extract_mfcc_features(test_file_path)
            likelihood_scores = hmm_trainer.get_likelihood_scores(mfcc_test_feat)
            
            # Print the likelihood scores
            print(f"Likelihood scores for {piece} (Take {take}):")
            for p, score in likelihood_scores.items():
                print(f"{p}: {score:.2f}")
            
            # Determine the best match based on highest likelihood score
            best_match_piece = max(likelihood_scores, key=likelihood_scores.get)
            best_score = likelihood_scores[best_match_piece]
            is_correct = best_match_piece == piece
            
            # Output classification result
            print(f"Classified as: {best_match_piece} (Correct: {is_correct})")
            print("-" * 50)
            
            # Update summary counters
            total_tests += 1
            if is_correct:
                correct_predictions += 1

    # Print summary of results
    print("Summary of Classification Results:")
    print(f"Total Tests: {total_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions / total_tests * 100:.2f}%")

if __name__ == "__main__":
    main()
