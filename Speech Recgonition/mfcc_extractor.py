import numpy as np
import scipy.fftpack as fft
from scipy.io import wavfile
from scipy.signal import get_window
import matplotlib.pyplot as plt
from pydub import AudioSegment  # Import pydub for MP3 support
from scipy.interpolate import interp1d

class MFCCExtractor:
    def __init__(self, num_coeffs=20):
        self.FFT_size = 2048
        self.hop_size = 10
        self.mel_filter_num = num_coeffs // 2
        self.dct_filter_num = 2 * self.mel_filter_num
        self.sample_rate = 44100

    def format_mfcc(self, mfcc, frames):
        target_shape = (frames, self.dct_filter_num)
        mfcc = mfcc.copy()
        padding_needed = target_shape[0] - mfcc.shape[0]
        if padding_needed > 0:
            # Pad with zeros if needed
            mfcc = np.pad(mfcc, ((0, padding_needed), (0, 0)), mode='constant')
        elif padding_needed < 0:
            # Truncate if the array is too long
            mfcc = mfcc[:target_shape[0], :]
        
        return mfcc.flatten()



    # def format_mfcc(self, mfcc, frames):
    #     target_shape = (frames, self.dct_filter_num)
    #     mfcc = mfcc.copy()
        
    #     if mfcc.shape[0] != target_shape[0]:
    #         # Create an array of indices for the original and target frames
    #         original_indices = np.linspace(0, 1, mfcc.shape[0])
    #         target_indices = np.linspace(0, 1, target_shape[0])
            
    #         # Interpolate each coefficient dimension independently
    #         mfcc_interp = np.zeros(target_shape)
    #         for i in range(mfcc.shape[1]):
    #             interpolator = interp1d(original_indices, mfcc[:, i], kind='linear', fill_value="extrapolate")
    #             mfcc_interp[:, i] = interpolator(target_indices)
            
    #         mfcc = mfcc_interp
        
    #     return mfcc.flatten()


    def normalise_audio(self, audio):
        """Normalize the audio signal to a range of -1 to 1."""
        return audio / np.max(np.abs(audio))

    def frame_audio(self, audio):
        """Split the audio signal into frames for FFT."""
        audio = np.pad(audio, int(self.FFT_size / 2), mode='constant')
        frame_len = np.round(self.sample_rate * self.hop_size / 1000).astype(int)
        frame_num = int((len(audio) - self.FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, self.FFT_size))

        for n in range(frame_num):
            frames[n] = audio[n * frame_len:n * frame_len + self.FFT_size]

        return frames

    def freq_to_mel(self, freq):
        """Convert a frequency value to mel scale."""
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_freq(self, mel):
        """Convert a mel-scale value back to frequency."""
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    def get_filter_points(self, fmin, fmax):
        """Get filter points for mel filters based on FFT size and sample rate."""
        fmin_mel = self.freq_to_mel(fmin)
        fmax_mel = self.freq_to_mel(fmax)

        mels = np.linspace(fmin_mel, fmax_mel, num=self.mel_filter_num + 2)
        freqs = self.mel_to_freq(mels)

        return np.floor((self.FFT_size + 1) / self.sample_rate * freqs).astype(int), freqs

    def get_filters(self, filter_points):
        """Create mel filter bank."""
        filters = np.zeros((len(filter_points) - 2, int(self.FFT_size / 2 + 1)))

        for n in range(len(filter_points) - 2):
            filters[n, filter_points[n]:filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]:filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])

        return filters

    def dct(self, filter_len):
        """Apply Discrete Cosine Transform (DCT) to the log-mel spectrogram to get MFCCs."""
        basis = np.empty((self.dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)

        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
        for i in range(1, self.dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

        return basis

    def extract(self, file_path=None, audio_data=None, frames=None):
        """Main function to extract MFCC features from an audio file."""
        if file_path is not None:
            sample_rate, audio = wavfile.read(file_path)

        elif audio_data is not None:
            audio = audio_data

        audio = self.normalise_audio(audio)

        # Frame the audio and apply a Hann window
        audio_framed = self.frame_audio(audio)
        window = get_window("hann", self.FFT_size, fftbins=True)
        audio_win = audio_framed * window
        audio_winT = audio_win.T

        # FFT and power spectrum
        audio_fft = np.empty((int(1 + self.FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        audio_fft = np.transpose(audio_fft)
        audio_power = np.square(np.abs(audio_fft))

        # Mel filterbank
        filter_points, mel_freqs = self.get_filter_points(0, self.sample_rate / 2)
        filters = self.get_filters(filter_points)
        enorm = 2.0 / (mel_freqs[2:self.mel_filter_num + 2] - mel_freqs[:self.mel_filter_num])
        filters *= enorm[:, np.newaxis]

        # Filter the power spectrum through the mel filterbank
        audio_filtered = np.dot(filters, np.transpose(audio_power))
        audio_log = 10.0 * np.log10(audio_filtered)

        # DCT to get MFCC
        dct_filters = self.dct(self.mel_filter_num)
        mfcc = np.dot(dct_filters, audio_log).T

        if frames is not None:
            mfcc = self.format_mfcc(mfcc, frames)

        return mfcc

    def visualize_mfcc(self, mfcc):
        """Visualize the MFCC coefficients using frame indices on the x-axis."""
        plt.figure(figsize=(10, 6))
        plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
        plt.title('MFCC Coefficients')
        plt.xlabel('Frame Index')
        plt.ylabel('MFCC Coefficient Index')
        plt.colorbar(label='Amplitude')
        plt.show()
