import librosa

def apply_stft(x , n_fft = 1024 , hop_length = 512 , window = "hamming"):
    x_stft = librosa.stft(x , n_fft=n_fft , hop_length=hop_length , window=window)
    print(f"shape of the stft of x -> {x_stft.shape}")
    return x_stft