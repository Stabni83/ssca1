import librosa

def load_wav_file(path):
    x, sr = librosa.load(path=path, mono=False, sr=None)
    print(f'Shape of loaded wav file -> {x.shape}')
    return x, sr