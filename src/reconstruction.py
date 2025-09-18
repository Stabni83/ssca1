import numpy as np
import os
import soundfile as sf
import librosa
from wiener import wiener_mask
from mel import reconstruct_from_mel


def apply_istft_multiscale(s_linear, s_mel, x_stft, path, mel_fb, sr, p=2.0):
    """
    Reconstructs time-domain signals from multi-scale separation results using confidence-based fusion.
    Applies inverse STFT to convert fused time-frequency masks back to audio signals.

    Args:
        s_linear (numpy.ndarray): Estimated sources in linear frequency scale, shape (N, F_linear, T)
        s_mel (numpy.ndarray): Estimated sources in Mel scale, shape (N, F_mel, T)
        x_stft (numpy.ndarray): STFT of the mixture signal
        path (str): Path to the original audio file for naming output
        mel_fb (numpy.ndarray): Mel filterbank matrix
        sr (int): Sample rate of the audio signal
        p (float): Power parameter for Wiener mask calculation (default: 2.0)
    """
    # Get dimensions from input arrays
    N, F_linear, T = s_linear.shape
    _, F_mel, _ = s_mel.shape

    # Extract mixture magnitude and phase information
    x_mix = x_stft[0]
    x_phase = np.angle(x_mix)

    # Calculate Wiener masks for both linear and Mel scale sources
    masks_linear = wiener_mask(s_linear, p=p)
    masks_mel_linear, _ = reconstruct_from_mel(s_mel, x_stft, mel_fb, p=p)

    # Calculate confidence measures for each scale based on source sparsity
    confidence_linear = np.mean(
        np.max(np.abs(s_linear), axis=0) / (np.sum(np.abs(s_linear), axis=0) + 1e-8),
        axis=0,
    )
    confidence_mel = np.mean(
        np.max(np.abs(s_mel), axis=0) / (np.sum(np.abs(s_mel), axis=0) + 1e-8), axis=0
    )

    # Compute fusion weights based on confidence measures
    alpha = confidence_linear / (confidence_linear + confidence_mel + 1e-8)
    alpha = alpha[None, None, :]  # Add dimensions for broadcasting

    # Fuse masks from both scales using confidence-based weighting
    combined_masks = alpha * masks_linear + (1 - alpha) * masks_mel_linear

    # Create output directory structure based on input file path
    file_name = os.path.splitext(os.path.basename(path))[0]
    parent_folder = os.path.basename(os.path.dirname(path))
    source_type = "3_source" if "three_sources" in path else "4_source"
    output_dir = os.path.join(
        "output", f"{source_type}_{parent_folder}_{file_name}_multiscale"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Reconstruct and save each separated source
    for n in range(N):
        # Apply combined mask to mixture spectrum
        complex_spec = combined_masks[n] * np.abs(x_mix) * np.exp(1j * x_phase)

        # Convert back to time domain using inverse STFT
        source_time = librosa.istft(
            complex_spec, n_fft=1024, hop_length=512, window="hann"
        )

        # Normalize and save the reconstructed audio
        out_path = os.path.join(output_dir, f"source_{n+1}_multiscale.wav")
        sf.write(out_path, source_time / (np.max(np.abs(source_time)) + 1e-8), sr)

    print(f"Multiscale separation results saved to: {output_dir}")
