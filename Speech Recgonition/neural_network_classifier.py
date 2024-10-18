import neural_network as nn
import numpy as np

class NNClassifier:
    def __init__(self, mfcc_coeffs, mfcc_frames, num_layers, num_hidden_nodes, labels, weights_dir=None, learning_rate=0.01, epochs=200, bias=1):
        self.NUM_INPUTS = mfcc_coeffs * mfcc_frames
        self.NUM_LAYERS = num_layers
        self.NUM_HIDDEN_NODES = num_hidden_nodes
        self.LABELS = labels
        self.LEARNING_RATE = learning_rate
        self.EPOCHS = epochs
        self.BIAS = bias
        self.NUM_OUTPUTS = len(labels)

        if weights_dir is None:
            self.weights = nn.generateWeights(self.NUM_INPUTS, self.NUM_HIDDEN_NODES, self.NUM_OUTPUTS, self.NUM_LAYERS)
        else:
            loaded = np.load(weights_dir)
            self.weights = [loaded[key] for key in loaded]
        
    
    def classify(self, mfcc_feature):
        mfcc_feature = np.array([mfcc_feature])
        output = nn.feedForward(mfcc_feature, self.weights)[0]
        classification = self.LABELS[np.argmax(output)]
        output = output.tolist()
        output.append(classification)
        return output


    
        