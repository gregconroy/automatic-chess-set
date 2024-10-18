import numpy as np
import scipy.fftpack as fft
from scipy.io import wavfile
from scipy.signal import get_window
import matplotlib.pyplot as plt
from pydub import AudioSegment  # Import pydub for MP3 support

class MFCCExtractor:
    FFT_size = 2048
    hop_size = 10
    mel_filter_num = 50
    dct_filter_num = 2 * mel_filter_num
    sample_rate = 44100

    @staticmethod
    def normalise_audio(audio):
        """Normalize the audio signal to a range of -1 to 1."""
        return audio / np.max(np.abs(audio))

    @staticmethod
    def frame_audio(audio, FFT_size, hop_size, sample_rate):
        """Split the audio signal into frames for FFT."""
        # print(f"Audio shape before padding: {audio.shape}, Type: {type(audio)}")  # Debugging line
    
        audio = np.pad(audio, int(FFT_size / 2), mode='constant')
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, FFT_size))

        for n in range(frame_num):
            frames[n] = audio[n * frame_len:n * frame_len + FFT_size]

        return frames

    @staticmethod
    def freq_to_mel(freq):
        """Convert a frequency value to mel scale."""
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    @staticmethod
    def mel_to_freq(mel):
        """Convert a mel-scale value back to frequency."""
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    @staticmethod
    def get_filter_points(fmin, fmax, FFT_size, sample_rate):
        """Get filter points for mel filters based on FFT size and sample rate."""
        fmin_mel = MFCCExtractor.freq_to_mel(fmin)
        fmax_mel = MFCCExtractor.freq_to_mel(fmax)

        mels = np.linspace(fmin_mel, fmax_mel, num=MFCCExtractor.mel_filter_num + 2)
        freqs = MFCCExtractor.mel_to_freq(mels)

        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    @staticmethod
    def get_filters(filter_points, FFT_size):
        """Create mel filter bank."""
        filters = np.zeros((len(filter_points) - 2, int(FFT_size / 2 + 1)))

        for n in range(len(filter_points) - 2):
            filters[n, filter_points[n]:filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]:filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])

        return filters

    @staticmethod
    def dct(filter_len):
        """Apply Discrete Cosine Transform (DCT) to the log-mel spectrogram to get MFCCs."""
        basis = np.empty((MFCCExtractor.dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)
        
        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
        for i in range(1, MFCCExtractor.dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
        return basis

    @classmethod
    def extract(cls, file_path):
        """Main function to extract MFCC features from an audio file."""
        # Check the file extension
        if file_path.endswith('.mp3'):
            # Load the MP3 file using pydub
            audio = AudioSegment.from_mp3(file_path)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            samples = samples / (2 ** 15)  # Normalize for 16-bit audio
            sample_rate = audio.frame_rate
        elif file_path.endswith('.wav'):
            sample_rate, audio = wavfile.read(file_path)
            audio = cls.normalise_audio(audio)
        else:
            raise ValueError("Unsupported file format. Only MP3 and WAV formats are supported.")

        # Frame the audio and apply a Hann window
        audio_framed = cls.frame_audio(audio, cls.FFT_size, cls.hop_size, cls.sample_rate)
        window = get_window("hann", cls.FFT_size, fftbins=True)
        audio_win = audio_framed * window
        audio_winT = np.transpose(audio_win)

        # FFT and power spectrum
        audio_fft = np.empty((int(1 + cls.FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        audio_fft = np.transpose(audio_fft)
        audio_power = np.square(np.abs(audio_fft))

        # Mel filterbank
        filter_points, mel_freqs = cls.get_filter_points(0, cls.sample_rate / 2, cls.FFT_size, cls.sample_rate)
        filters = cls.get_filters(filter_points, cls.FFT_size)
        enorm = 2.0 / (mel_freqs[2:cls.mel_filter_num + 2] - mel_freqs[:cls.mel_filter_num])
        filters *= enorm[:, np.newaxis]

        # Filter the power spectrum through the mel filterbank
        audio_filtered = np.dot(filters, np.transpose(audio_power))
        audio_log = 10.0 * np.log10(audio_filtered)

        # DCT to get MFCC
        dct_filters = cls.dct(cls.mel_filter_num)
        mfcc = np.dot(dct_filters, audio_log)
        
        return mfcc

    @staticmethod
    def visualize_mfcc(mfcc):
        """Visualize the MFCC coefficients using frame indices on the x-axis."""
        plt.figure(figsize=(10, 6))
        plt.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        plt.title('MFCC Coefficients')
        plt.xlabel('Frame Index')
        plt.ylabel('MFCC Coefficient Index')
        plt.colorbar(label='Amplitude')
        plt.show()
