import librosa
import numpy as np
from scipy.signal import get_window

__all__ = ['parameterize_audio']

def parameterize_audio(filename, window_size=10, frame_size=2048):
    """
    Parametrize audio file.
    The audio is first re-sampled at 22,05 kHz and then windowed at a 10Hz rate with 2048-sample frames. Then, 45 MFC
    coefficients are computed, and the 15 with highest variance across the whole audio are kept. Eventually, those MFCCs
    are rescaled to zero mean and unit variance.
    :param filename: Name of the audio file (.mp3 or .wav)
    :param window_size: Size of the Hamming window (in Hz)
    :param frame_size: Number of samples in each frame
    :return: Parametrized audio (n_samples x 15) and corresponding time axis (n_samples)
    """
    # Load signal
    signal, sr = librosa.load(filename)

    # Compute Mel-Frequency Cepstral Coefficients
    window_size = int(sr / window_size)  # Convert window size from Hz to number of samples
    melspec_args = {"n_fft": window_size, "hop_length": frame_size, "window": get_window("hamming", window_size)}
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, S=None, n_mfcc=45, **melspec_args)

    # Compute MFCCs variance across the whole audio and keep the fifteen with highest variance
    mfccs_variance = mfccs.var(axis=1)
    mfccs = mfccs[mfccs_variance.argsort()[-15:], :]

    # Scale the coefficients to unit variance and zero mean
    parameterized_audio = (mfccs - mfccs.mean(axis=1)[:, np.newaxis]) / mfccs.std(axis=1)[:, np.newaxis]

    # Compute time axis
    time_axis = frame_size * np.arange(parameterized_audio.shape[1]) / sr

    return parameterized_audio.T, time_axis

if __name__ == '__main__':
    # Path to audio file
    file_path = 'data/metallica_seek_and_destroy.mp3'

    # Parametrize audio
    parameterized_audio, time_axis = parameterize_audio(file_path)

    end = True