import os
import sys

# Add current directory to system path to enable module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules for the multi-scale SSCA pipeline
from wav_loader import load_wav_file
from paths import root_dir
from mixing_matrix import A_estimation
from stft import apply_stft
from mel import create_mel_filterbank, compress_to_mel
from separation_algorithm import (
    douglas_rachford_linear_scale,
    douglas_rachford_mel_scale,
)
from reconstruction import apply_istft_multiscale


def multiscale_ssca():
    """
    Main pipeline for Multi-scale Sparse Spatial Component Analysis (SSCA).
    Processes all WAV files in the dataset directory, performing blind source separation
    using a dual-scale (linear + Mel) approach with confidence-based fusion.
    """
    # Walk through all files in the dataset directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                print(f"\nProcessing: {full_path}")

                # Load audio file and determine number of sources from filename
                x, sr = load_wav_file(full_path)
                n_sources = 3 if "three_sources" in full_path else 4

                # Estimate mixing matrix using hierarchical clustering
                A_est = A_estimation(x, n_sources)

                # Apply STFT to convert signal to time-frequency domain
                x_stft = apply_stft(x, n_fft=1024, hop_length=512, window="hann")

                # Create Mel filterbank and compress STFT to Mel scale
                mel_fb = create_mel_filterbank(sr, n_fft=1024, n_mels=80)
                x_mel = compress_to_mel(x_stft, mel_fb)

                # Perform source separation in linear frequency scale
                print("Running separation on linear frequency scale...")
                s_linear, w_linear = douglas_rachford_linear_scale(
                    x_stft, A_est, gamma=0.1, max_iter=1000
                )

                # Perform source separation in Mel frequency scale
                print("Running separation on Mel scale...")
                s_mel, w_mel = douglas_rachford_mel_scale(
                    x_mel, A_est, gamma=0.1, max_iter=1000
                )

                # Reconstruct sources using multi-scale fusion and save results
                print("Reconstructing using multi-scale fusion...")
                apply_istft_multiscale(
                    s_linear, s_mel, x_stft, full_path, mel_fb, sr, p=2.0
                )

    print("\nMulti-scale SSCA processing completed!")
