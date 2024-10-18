import numpy as np

restrict = 10_000

def errorCost(value, target):
    return (1_000_000/restrict) * 0.5 * np.sum((value - target) ** 2)

def normalise(value):
    min_val = np.min(value)
    max_val = np.max(value)
    normalized_value = (value - min_val) / (max_val - min_val)
    return normalized_value

def generateWeights(numIn, hiddenNodes, outputs, layers):
    # we subtract 2 as we don't want to include input and output weights
    layers -= 2
    cap = 1.0/np.sqrt(hiddenNodes) # we set the max/min value for the weights
    weights = []
    weights.append(np.random.uniform(-cap,cap,(numIn+1, hiddenNodes)))
    for i in range(layers-1):
        weights.append(np.random.uniform(-cap,cap,(hiddenNodes+1,hiddenNodes)))
    weights.append(np.random.uniform(-cap,cap,(hiddenNodes+1,outputs)))
    return weights

def activation(x, derive = False):
    if derive:
        return 1 - activation(x)**2
    return np.tanh(x)

def backProp(inData, targetData, weights, lr, tEpoch, tError, bias):
    normalError = []
    L2 = []
    for z in range(tEpoch):
        # first we have to do feedforward algorithim
        layerOutputs = []
        layerOutputs.append(np.hstack((np.full((inData.shape[0], 1), bias), inData)))
        for i in weights:
            layerOutputs.append(np.hstack((np.full((layerOutputs[-1].shape[0], 1), bias), activation(np.dot(layerOutputs[-1], i)))))
        layerOutputs[-1] = layerOutputs[-1][:, 1:]
        
        # After forward propogation we need to calculate backwards propogation to update the weights
        layerOutputs = layerOutputs[::-1]
        weights = weights[::-1]
        errors = []
        errors.append(layerOutputs[0] - targetData)
        for i, j in zip(layerOutputs[1:-1], weights):
            errors.append(activation(i[:,1:], True)*np.dot(errors[-1],j.T[:, 1:]))
        layerOutputs = layerOutputs[::-1]
        weights = weights[::-1]
        errors = errors[::-1]

        # we now have to calculate the partial derivatives and at the same time we can update the weights
        for i in range(len(weights)):
            PD = layerOutputs[i][:, :, np.newaxis] * errors[i][: , np.newaxis, :]
            gradient = np.average(PD, axis=0)
            weights[i] += -lr * gradient
        normalError.append(np.abs(np.mean(normalise(layerOutputs[-1] - targetData))))
        L2.append(np.mean(errorCost(layerOutputs[-1], targetData)))
        # L2.append(np.mean(np.abs(layerOutputs[-1] - targetData)))
        # print(L2[-1], normalError[-1])
        # if normalError < 0.0005:
        #     break
    return L2, weights

def feedForward(inData, weights):
    bias = 1
    layerOutputs = []
    layerOutputs.append(np.hstack((np.full((inData.shape[0], 1), bias), inData)))
    for i in weights:
        layerOutputs.append(np.hstack((np.full((layerOutputs[-1].shape[0], 1), bias), activation(np.dot(layerOutputs[-1], i)))))
    layerOutputs[-1] = layerOutputs[-1][:, 1:]

    return layerOutputs[-1]