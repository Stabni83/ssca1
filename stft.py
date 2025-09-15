def apply_multi_scale_stft(x, n_fft_list=[256, 512, 1024, 2048]):
    """
    Apply STFT with multiple window sizes
    """
    stft_list = []
    for n_fft in n_fft_list:
        hop_length = n_fft // 2
        X_stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window="hamming")
        stft_list.append(X_stft)
    return stft_list