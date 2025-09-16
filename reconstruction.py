import numpy as np
import os
import soundfile as sf
import librosa
from wiener import wiener_mask
from mel import reconstruct_from_mel

def apply_istft_multiscale(s_linear, s_mel, x_stft, path, mel_fb, sr, p=2.0):
    N, F_linear, T = s_linear.shape
    _, F_mel, _ = s_mel.shape

    x_mix = x_stft[0]
    x_phase = np.angle(x_mix)

    masks_linear = wiener_mask(s_linear, p=p)
    masks_mel_linear, _ = reconstruct_from_mel(s_mel, x_stft, mel_fb, p=p)

    confidence_linear = np.mean(np.max(np.abs(s_linear), axis=0) / (np.sum(np.abs(s_linear), axis=0) + 1e-8), axis=0)
    confidence_mel = np.mean(np.max(np.abs(s_mel), axis=0) / (np.sum(np.abs(s_mel), axis=0) + 1e-8), axis=0)

    alpha = confidence_linear / (confidence_linear + confidence_mel + 1e-8)
    alpha = alpha[None, None, :]

    combined_masks = alpha * masks_linear + (1 - alpha) * masks_mel_linear

    file_name = os.path.splitext(os.path.basename(path))[0]
    parent_folder = os.path.basename(os.path.dirname(path))
    source_type = "3_source" if "three_sources" in path else "4_source"
    output_dir = os.path.join("output", f"{source_type}_{parent_folder}_{file_name}_multiscale")
    os.makedirs(output_dir, exist_ok=True)

    for n in range(N):
        complex_spec = combined_masks[n] * np.abs(x_mix) * np.exp(1j * x_phase)
        source_time = librosa.istft(complex_spec, n_fft=1024, hop_length=512, window="hann")
        out_path = os.path.join(output_dir, f"source_{n+1}_multiscale.wav")
        sf.write(out_path, source_time / (np.max(np.abs(source_time)) + 1e-8), sr)

    print(f"Multiscale separation results saved to: {output_dir}")