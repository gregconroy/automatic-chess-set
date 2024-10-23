import os
import numpy as np
from hmmlearn import hmm
from mfcc_class import MFCCExtractor  # Your MFCCExtractor class

# Class to handle all HMM-related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=100):
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

# Main training and testing function
def main():
    pieces = ["king", "queen", "bishop", "knight", "rook", "pawn"]  # List of chess pieces
    person_name = "greg"  # External variable for the person's name
    sample_number = 1  # External variable for the sample number
    directory_path = './Segments/pieces/'  # Directory containing piece audio files

    # Initialize the HMM trainer
    hmm_trainer = HMMTrainer(n_components=5)  # Adjust the number of states as needed

    # Collect training data and train the model for each piece
    for piece in pieces:
        file_name = f'{person_name}_{piece}_{sample_number}.wav'  # Use external variables for name and sample number
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.isfile(file_path):  # Check if the file exists
            # Use your MFCCExtractor to get MFCCs
            mfcc_features = MFCCExtractor.extract(file_path)
            X_train = mfcc_features  # X_train is already in the correct format
            
            # Train the HMM model
            hmm_trainer.train(X_train, piece)

    # Initialize counters for summary
    total_tests = 0
    correct_predictions = 0
    summary_results = {piece: {"total": 0, "correct": 0} for piece in pieces}

    # Test the model with the same pieces
    for piece in pieces:
        for take in range(1, 8):  # Adjust the range for your testing takes
            test_file_name = f'{person_name}_{piece}_{take}.wav'  # Use the same naming convention
            test_file_path = os.path.join(directory_path, test_file_name)

            if os.path.isfile(test_file_path):
                mfcc_test_feat = MFCCExtractor.extract(test_file_path)
                likelihood_scores = hmm_trainer.get_likelihood_scores(mfcc_test_feat)

                # Print the likelihood scores
                print(f"Likelihood scores for {piece} (Take {take}):")
                for p, score in likelihood_scores.items():
                    print(f"{p}: {score:.2f}")

                # Determine the best match based on highest likelihood score
                best_match_piece = max(likelihood_scores, key=likelihood_scores.get)
                is_correct = best_match_piece == piece

                # Output classification result
                print(f"Classified as: {best_match_piece} (Correct: {is_correct})")
                print("-" * 50)

                # Update summary counters
                total_tests += 1
                summary_results[piece]["total"] += 1
                if is_correct:
                    correct_predictions += 1
                    summary_results[piece]["correct"] += 1

    # Print summary of results
    print("Summary of Classification Results:")
    print(f"Total Tests: {total_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions / total_tests * 100:.2f}%")

    # Print detailed summary for each piece
    print("\nDetailed Summary for Each Piece:")
    for piece in pieces:
        total = summary_results[piece]["total"]
        correct = summary_results[piece]["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"Piece '{piece}': Total Tests = {total}, Correct Predictions = {correct}, Accuracy = {accuracy:.2f}%")

if __name__ == "__main__":
    main()
