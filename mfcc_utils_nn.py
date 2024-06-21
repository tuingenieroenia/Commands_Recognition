# mfcc_extraction.py
import os
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from scipy.signal.windows import hamming

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def apply_window(frames):
    return frames * hamming(frames.shape[1], sym=False)

def fft_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames

def mel_filter_banks(pow_frames, sample_rate, nfilt=40, NFFT=512):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

def mfcc(signal, sample_rate, numcep=13):
    emphasized_signal = pre_emphasis(signal)
    frames = framing(emphasized_signal, sample_rate)
    windowed_frames = apply_window(frames)
    pow_frames = fft_spectrum(windowed_frames)
    filter_banks = mel_filter_banks(pow_frames, sample_rate)
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :numcep]
    return mfccs

def extract_features_with_mfcc_length(directory_path, n_mfcc, max_len, commands):
    feature_list = []
    label_list = []
    for label in commands:
        label_dir = os.path.join(directory_path, label)
        if not os.path.exists(label_dir):
            print(f"Directory {label_dir} does not exist")
            continue
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(label_dir, file_name)
                sample_rate, signal = wavfile.read(file_path)
                features = mfcc(signal, sample_rate, numcep=n_mfcc)

                if len(features) > max_len:
                    features = features[:max_len]
                elif len(features) < max_len:
                    pad_width = max_len - len(features)
                    features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')

                feature_list.append(features)
                label_list.append(label)
    return np.array(feature_list), np.array(label_list)
