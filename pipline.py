import os
import librosa
from wav_loader import load_wav_file
from paths import root_dir
from mixing_matrix import A_estimation
from stft import apply_stft
from separation_algorithm import douglas_rachford
from reconstruction import apply_istft
from mel import create_mel_filterbank, compress_to_mel

def ssca():    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                x , sr = load_wav_file(full_path)
                A_est = A_estimation(x , 3 if "three_sources" in full_path else 4)
                x_stft = apply_stft(x)
            
                mel_fb = create_mel_filterbank(sr, 1024, n_mels=80)
                x_mel = compress_to_mel(x_stft, mel_fb)
                
                sources = douglas_rachford(x_mel , A_est)
                
                apply_istft(sources , x_stft , full_path, mel_fb=mel_fb, sr=sr)
