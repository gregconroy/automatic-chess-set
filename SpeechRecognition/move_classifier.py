from neural_network_classifier import NNClassifier
from mfcc_extractor import MFCCExtractor
import numpy as np

class MoveClassifier:
    PIECES = ['king', 'queen', 'bishop', 'knight', 'rook', 'pawn']
    LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8']

    def __init__(self):
        self.mfcc_extractor = MFCCExtractor(num_coeffs=100)
        self.piece_classifier = NNClassifier(mfcc_coeffs=100, mfcc_frames=100, num_layers=3, num_hidden_nodes=64, labels=self.PIECES, weights_dir='./CNN/piece_weights.npz')
        self.letter_classifier = NNClassifier(mfcc_coeffs=100, mfcc_frames=100, num_layers=3, num_hidden_nodes=64, labels=self.LETTERS, weights_dir='./CNN/letter_weights.npz')
        self.number_classifier = NNClassifier(mfcc_coeffs=100, mfcc_frames=100, num_layers=3, num_hidden_nodes=64, labels=self.NUMBERS, weights_dir='./CNN/number_weights.npz')


    def classify(self, audio_segments):
        piece_mfcc_feature = self.mfcc_extractor.extract(audio_data=audio_segments['piece'], frames=100)
        piece_prediction = self.piece_classifier.classify(piece_mfcc_feature)

        letter_mfcc_feature = self.mfcc_extractor.extract(audio_data=audio_segments['letter'], frames=100)
        letter_prediction = self.letter_classifier.classify(letter_mfcc_feature)

        number_mfcc_feature = self.mfcc_extractor.extract(audio_data=audio_segments['number'], frames=100)
        number_prediction = self.number_classifier.classify(number_mfcc_feature)
        
        predictions = {
            'piece': piece_prediction,
            'letter': letter_prediction,
            'number': number_prediction
        }

        return predictions

if __name__ == "__main__":
    pass