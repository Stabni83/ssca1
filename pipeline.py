import os
from wav_loader import load_wav_file
from paths import root_dir
from mixing_matrix import A_estimation
from stft import apply_stft
from mel import create_mel_filterbank, compress_to_mel
from separation_algorithm import douglas_rachford_linear_scale, douglas_rachford_mel_scale
from reconstruction import apply_istft_multiscale

def multiscale_ssca():
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                print(f"\nProcessing: {full_path}")

                x, sr = load_wav_file(full_path)
                n_sources = 3 if "three_sources" in full_path else 4
                A_est = A_estimation(x, n_sources)

                x_stft = apply_stft(x, n_fft=1024, hop_length=512, window="hann")
                mel_fb = create_mel_filterbank(sr, n_fft=1024, n_mels=80)
                x_mel = compress_to_mel(x_stft, mel_fb)

                print("Running separation on linear frequency scale...")
                s_linear, w_linear = douglas_rachford_linear_scale(x_stft, A_est, gamma=0.1, max_iter=1000)

                print("Running separation on Mel scale...")
                s_mel, w_mel = douglas_rachford_mel_scale(x_mel, A_est, gamma=0.1, max_iter=1000)

                print("Reconstructing using multi-scale fusion...")
                apply_istft_multiscale(s_linear, s_mel, x_stft, full_path, mel_fb, sr, p=2.0)

    print("\nMulti-scale SSCA processing completed!")