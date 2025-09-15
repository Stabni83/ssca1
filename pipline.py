import os
import librosa
from wav_loader import load_wav_file
from paths import root_dir
from matlab_loader import activate_matlab
from mixing_matrix import A_estimation
from stft import apply_stft
from separation_algorithm import douglas_rachford
from reconstruction import apply_istft
from mel import create_mel_filterbank, compress_to_mel
from weight import multi_scale_weights_estimation

def ssca():
    eng_matlab = activate_matlab()
    n_fft_list = [256, 512, 1024, 2048]  # مقیاس‌های مختلف

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                x, sr = load_wav_file(full_path)
                A_est = A_estimation(x, 3.0 if "three_sources" in full_path else 4.0, eng_matlab)
                
                # چندمقیاسی
                x_stft_list = apply_stft(x, n_fft_list=n_fft_list)
                
                # فشرده‌سازی به Mel برای هر مقیاس
                mel_fb = create_mel_filterbank(sr, 1024, n_mels=80)
                x_mel_list = [compress_to_mel(x_stft, mel_fb) for x_stft in x_stft_list]
                
                # محاسبه وزن چندمقیاسی
                w_combined = multi_scale_weights_estimation(x_mel_list, A_est)
                
                # جداسازی با وزن ترکیبی
                sources = douglas_rachford(x_mel_list[0], A_est, w_combined)
                
                apply_istft(sources, x_stft_list[0], full_path, mel_fb=mel_fb, sr=sr)

    eng_matlab.exit()