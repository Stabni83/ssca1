import librosa


def load_wav_file(path):
    """
    Loads WAV audio file.

    Args:
        path: Path to audio file

    Returns:
        x: Audio signal array
        sr: Sample rate
    """
    x, sr = librosa.load(path=path, mono=False, sr=None)
    print(f"Shape of loaded wav file -> {x.shape}")
    return x, sr
