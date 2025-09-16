import librosa
import numpy as np

def create_mel_filterbank(sr, n_fft, n_mels=100):
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sr/2)

def compress_to_mel(X_stft, mel_fb):
    X_mel = []
    for p in range(X_stft.shape[0]):
        mag = np.abs(X_stft[p])
        X_mel.append(mel_fb @ mag)
    return np.stack(X_mel, axis=0)

def reconstruct_from_mel(S_mel, X_stft, mel_fb, p=2.0):
    X_mix = X_stft[0]
    X_phase = np.angle(X_mix)
    N, F_mel, T = S_mel.shape
    
    mag_sources_mel = np.abs(S_mel) ** p
    denom = np.sum(mag_sources_mel, axis=0, keepdims=True) + 1e-8
    masks_mel = mag_sources_mel / denom
    
    masks_linear = np.array([mel_fb.T @ masks_mel[n] for n in range(len(S_mel))])
    
    return masks_linear, X_phase