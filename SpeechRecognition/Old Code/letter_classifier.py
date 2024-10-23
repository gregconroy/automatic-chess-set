import os
import numpy as np
from hmmlearn import hmm
from mfcc_class import MFCCExtractor

# Configuration settings
TRAIN_TAKES = range(1, 41)  # Training takes (1 to 4)
TEST_TAKES = range(41, 51)  # Testing takes (1 to 7)
PIECES = ["a", "b", "c", "d", "e", "f", "g", "h"]
DATA_DIR = "./Segments/letters/"  # Directory path to audio files
N_COMPONENTS = 12  # Number of HMM components

class HMMTrainer:
    """Class to handle HMM training and scoring."""
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=100):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = {}

    def train(self, X, letter_name):
        model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
        model.fit(X)
        self.models[letter_name] = model

    def get_likelihood_scores(self, input_data):
        return {letter: model.score(input_data) for letter, model in self.models.items()}


def extract_mfcc_features(file_path):
    """Extract MFCC features from a given audio file."""
    if not os.path.isfile(file_path):
        print(f"Warning: File not found - {file_path}")
        return None
    mfcc_extractor = MFCCExtractor()
    return np.transpose(mfcc_extractor.extract(file_path))


def train_model(hmm_trainer, letters, train_takes, data_dir):
    """Train HMM models for each letter based on MFCC features from training takes."""
    for letter in letters:
        train_features = []
        for take in train_takes:
            file_path = os.path.join(data_dir, f"greg_{letter}_{take}.wav")
            mfcc_feat = extract_mfcc_features(file_path)
            if mfcc_feat is not None:
                train_features.append(mfcc_feat)

        if train_features:
            X_train = np.vstack(train_features)
            hmm_trainer.train(X_train, letter)
            print(f"Trained model for letter: {letter}")


def test_model(hmm_trainer, letters, test_takes, data_dir):
    """Test the trained HMM models and calculate accuracy."""
    total_tests = 0
    correct_predictions = 0
    summary_results = {letter: {"total": 0, "correct": 0} for letter in letters}

    for letter in letters:
        for take in test_takes:
            test_file_path = os.path.join(data_dir, f"greg_{letter}_{take}.wav")
            mfcc_test_feat = extract_mfcc_features(test_file_path)
            if mfcc_test_feat is not None:
                likelihood_scores = hmm_trainer.get_likelihood_scores(mfcc_test_feat)

                # Print each likelihood score with a newline for clarity
                print(f"\nLikelihood scores for {letter} (Take {take}):")
                for p, score in likelihood_scores.items():
                    print(f"{p}: {score:.2f}")

                # Determine the best match based on highest likelihood score
                best_match_letter = max(likelihood_scores, key=likelihood_scores.get)
                is_correct = best_match_letter == letter

                # Update summary counters
                total_tests += 1
                summary_results[letter]["total"] += 1
                if is_correct:
                    correct_predictions += 1
                    summary_results[letter]["correct"] += 1

                # Print classification result with a newline
                print(f"\nClassified as: {best_match_letter} (Correct: {is_correct})\n")

    # Final summary print
    accuracy = (correct_predictions / total_tests) * 100 if total_tests else 0
    print(f"\nTotal Tests: {total_tests}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy:.2f}%\n")
    
    return summary_results, total_tests, correct_predictions


def print_summary(summary_results, total_tests, correct_predictions):
    """Print a detailed summary of classification results."""
    accuracy = (correct_predictions / total_tests) * 100 if total_tests else 0
    print("Summary of Classification Results:")
    print(f"Total Tests: {total_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    print("\nDetailed Summary for Each Letter:")
    for letter, result in summary_results.items():
        total = result["total"]
        correct = result["correct"]
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Letter '{letter}': Total Tests = {total}, Correct Predictions = {correct}, Accuracy = {accuracy:.2f}%")


def main():
    hmm_trainer = HMMTrainer(n_components=N_COMPONENTS)

    print("Training models...")
    train_model(hmm_trainer, PIECES, TRAIN_TAKES, DATA_DIR)

    print("Testing models...")
    summary_results, total_tests, correct_predictions = test_model(hmm_trainer, PIECES, TEST_TAKES, DATA_DIR)

    print_summary(summary_results, total_tests, correct_predictions)


if __name__ == "__main__":
    main()
