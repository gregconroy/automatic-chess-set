import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt

def normalise_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
    audio = np.pad(audio, int(FFT_size / 2), mode='empty')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))

    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]

    return frames


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)

    print(f"MEL min: {fmin_mel}")
    print(f"MEL max: {fmax_mel}")

    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = mel_to_freq(mels)

    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs


def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis


file_path = './Segments/pieces/'
file_name = 'roxanne_knight_1.wav'
ipd.Audio(file_path + file_name)

sample_rate, audio = wavfile.read(file_path + file_name)
print(f"Sample rate: {sample_rate}Hz")
print(f"Audio duration: {len(audio) / sample_rate}s")

audio = normalise_audio(audio)
# plt.figure(figsize=(15,4))
# plt.plot(np.linspace(0, len(audio) / sample_rate, num = len(audio)), audio)
# plt.grid(True)

hop_size = 10 #ms
FFT_size = 2048
sample_rate = 44100 #Hz

audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
print(f"Framed audio shape: {audio_framed.shape}")

print(f"First frame:\n{audio_framed[1]}")
print(f"Last frame:\n{audio_framed[-1]}")

window = get_window("hann", FFT_size, fftbins=True)
# plt.figure(figsize=(15, 4))
# plt.plot(window)
# plt.grid(True)

audio_win = audio_framed * window
index = 0
# plt.figure(figsize=(15,6))
# plt.subplot(2, 1, 1)
# plt.plot(audio_framed[index])
# plt.title('Original Frame')
# plt.grid(True)
# plt.subplot(2, 1, 2)
# plt.plot(audio_win[index])
# plt.title('Frame After Windowing')
# plt.grid(True)

audio_winT = np.transpose(audio_win)

audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

for n in range(audio_fft.shape[1]):
    audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

audio_fft = np.transpose(audio_fft)

audio_power = np.square(np.abs(audio_fft))
print(audio_power.shape)

freq_min = 0
freq_max = sample_rate / 2
mel_filter_num = 10

print(f"Minimum freqeuncy: {freq_min}")
print(f"Maximum freqeuncy: {freq_max}")

filter_points, mel_freqs = get_filter_points(freq_min, freq_max, mel_filter_num, FFT_size, sample_rate=44100)
print(f"Filter points:\n{filter_points}")

filters = get_filters(filter_points, FFT_size)
enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
filters *= enorm[:, np.newaxis]

# plt.figure(figsize=(15, 4))
# for n in range(filters.shape[0]):
#     plt.plot(filters[n])

audio_filtered = np.dot(filters, np.transpose(audio_power))
audio_log = 10.0 * np.log10(audio_filtered)
print(audio_log.shape)

dct_filter_num = mel_filter_num * 2
dct_filters = dct(dct_filter_num, mel_filter_num)
cepstral_coefficients = np.dot(dct_filters, audio_log)
print(cepstral_coefficients[:, 0])

plt.figure(figsize=(15, 5))
plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
plt.imshow(cepstral_coefficients, aspect='auto', origin='lower')
plt.show()
