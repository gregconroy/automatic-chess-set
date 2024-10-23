import numpy as np
from my_gaussian_hmm import GaussianHMM
from my_mfcc_extractor import MFCCExtractor

MFCC_COEFFICIENTS = 20

mfcc_extractor = MFCCExtractor(MFCC_COEFFICIENTS)

model = GaussianHMM(n_states=3, n_components=MFCC_COEFFICIENTS)
random_observations = np.random.rand(49, MFCC_COEFFICIENTS)

for i in range(1, 10):
    mfcc_feats = mfcc_extractor.extract(f'./Segments/pieces/greg_bishop_{i}.wav')
    # random_observations = np.random.rand(49, MFCC_COEFFICIENTS)

    model.train(mfcc_feats)
    print(i)

print(model.transition_probabilities)


# total_likelihood = model.likelihood(mfcc_feats)
# print(total_likelihood)
# total_likelihood = model.likelihood(random_observations)
# print(total_likelihood)

# print(f"Total Likelihood of the observations: {total_likelihood}")