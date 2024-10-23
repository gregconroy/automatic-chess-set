import numpy as np
import os
from pydub import AudioSegment
from scipy.fftpack import dct
from scipy.stats import multivariate_normal

# Pre-emphasis filter
def pre_emphasis(signal, pre_emph_coeff=0.97):
    return np.append(signal[0], signal[1:] - pre_emph_coeff * signal[:-1])

# Framing and windowing
def frame_signal(signal, frame_size, frame_stride, sample_rate):
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    return frames * np.hamming(frame_length)

# Fourier-Transform and Power Spectrum
def fourier_transform(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    return pow_frames

# Compute filter banks
def mel_filterbanks(pow_frames, sample_rate, nfilt=40, NFFT=512):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700)) 
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return 20 * np.log10(filter_banks)

# Discrete Cosine Transform for MFCC
def compute_mfcc(filter_banks, num_ceps=13):
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc

# Extract MFCC Features (Full pipeline)
def extract_mfcc_from_audio(file_path):
    audio = AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples = samples / (2**15)  # Normalize for 16-bit audio

    emphasized_signal = pre_emphasis(samples)
    frames = frame_signal(emphasized_signal, frame_size=0.025, frame_stride=0.01, sample_rate=audio.frame_rate)
    pow_frames = fourier_transform(frames)
    filter_banks = mel_filterbanks(pow_frames, sample_rate=audio.frame_rate)
    mfcc_features = compute_mfcc(filter_banks)
    return mfcc_features

# Forward-Backward Algorithm for HMM
def forward(X, transmat, start_prob, means, covars):
    N, T = len(start_prob), len(X)
    log_alpha = np.zeros((N, T))

    # Initialization
    for i in range(N):
        log_alpha[i, 0] = np.log(start_prob[i]) + multivariate_normal.logpdf(X[0], means[i], covars[i])

    # Recursion
    for t in range(1, T):
        for j in range(N):
            log_sum = np.logaddexp.reduce(log_alpha[:, t-1] + np.log(transmat[:, j]))
            log_alpha[j, t] = log_sum + multivariate_normal.logpdf(X[t], means[j], covars[j])

    return log_alpha

def backward(X, transmat, means, covars):
    N, T = len(transmat), len(X)
    log_beta = np.zeros((N, T))

    # Initialization (log_beta[:, T-1] = 0)

    # Recursion
    for t in reversed(range(T - 1)):
        for i in range(N):
            log_sum = np.logaddexp.reduce(log_beta[:, t + 1] + np.log(transmat[i, :]) + \
                                          multivariate_normal.logpdf(X[t + 1], means, covars))
            log_beta[i, t] = log_sum

    return log_beta

def log_likelihood(log_alpha):
    return np.logaddexp.reduce(log_alpha[:, -1])

# HMM Implementation
class HMM:
    def __init__(self, n_components=4, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.start_prob = np.full(n_components, 1.0 / n_components)
        self.transmat = np.full((n_components, n_components), 1.0 / n_components)
        self.means = None
        self.covars = None

    # Initialize means and covariances to match the shape of X
    def _init_params(self, X):
        n_features = X.shape[1]  # Number of MFCC features per time step
        self.means = np.mean(X, axis=0) + np.random.randn(self.n_components, n_features)
        self.covars = np.tile(np.cov(X.T), (self.n_components, 1, 1))


    def fit(self, X):
        self._init_params(X)
        for _ in range(self.n_iter):
            # E-step: Expectation (forward-backward)
            log_alpha = forward(X, self.transmat, self.start_prob, self.means, self.covars)
            log_beta = backward(X, self.transmat, self.means, self.covars)

            # M-step: Maximization (update parameters)
            xi = np.zeros((len(X), self.n_components, self.n_components))
            gamma = np.exp(log_alpha + log_beta)

            # Update transition probabilities
            for t in range(len(X) - 1):
                for i in range(self.n_components):
                    for j in range(self.n_components):
                        xi[t, i, j] = np.exp(log_alpha[i, t] + np.log(self.transmat[i, j]) +
                                             multivariate_normal.logpdf(X[t+1], self.means[j], self.covars[j]) +
                                             log_beta[j, t+1])

            self.transmat = np.sum(xi[:-1], axis=0) / np.sum(xi[:-1])

            # Update means and covariances
            for i in range(self.n_components):
                gamma_sum = np.sum(gamma[:, i])
                self.means[i] = np.sum(gamma[:, i].reshape(-1, 1) * X, axis=0) / gamma_sum
                X_centered = X - self.means[i]
                self.covars[i] = np.dot((gamma[:, i].reshape(-1, 1) * X_centered).T, X_centered) / gamma_sum

    def score(self, X):
        log_alpha = forward(X, self.transmat, self.start_prob, self.means, self.covars)
        return log_likelihood(log_alpha)

# HMM Trainer to handle multiple classes
class HMMTrainer:
    def __init__(self, n_components=4, n_iter=100):
        self.models = {}
        self.n_components = n_components
        self.n_iter = n_iter

    def train(self, X, piece_name):
        model = HMM(n_components=self.n_components, n_iter=self.n_iter)
        model.fit(X)
        self.models[piece_name] = model

    def get_likelihood_scores(self, input_data):
        scores = {piece: model.score(input_data) for piece, model in self.models.items()}
        return scores

def main():
    pieces = ["a", "b", "c", "d", "e", "f", "g", "h"]
    hmm_trainer = HMMTrainer()

    # Training Phase
    for piece in pieces:
        # Assume filenames like 'greg_a_1.mp3'
        for i in range(1, 6):  # Training takes 1-5
            file_path = f"Segments/letters/greg_{piece}_{i}.mp3"
            mfcc_features = extract_mfcc_from_audio(file_path)
            hmm_trainer.train(mfcc_features, piece)

    # Testing Phase
    test_file = "greg_a_6.mp3"
    test_features = extract_mfcc_from_audio(test_file)
    scores = hmm_trainer.get_likelihood_scores(test_features)
    
    print("Scores for each piece: ", scores)

if __name__ == "__main__":
    main()
