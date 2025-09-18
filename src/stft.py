import librosa


def apply_stft(x, n_fft=1024, hop_length=512, window="hann"):
    """
    Converts time-domain signal to time-frequency representation using STFT.

    Args:
        x: Input time-domain signal
        n_fft: FFT window size
        hop_length: Hop length between frames
        window: Window function type

    Returns:
        Complex-valued STFT matrix
    """
    x_stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
    print(f"Shape of the STFT of x -> {x_stft.shape}")
    return x_stft
