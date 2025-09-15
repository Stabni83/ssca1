import numpy as np 
import os 
import librosa
import soundfile as sf
from wiener import wiener_mask
from mel import reconstruct_from_mel  

def apply_istft(s , x , path , p=2.0, mel_fb=None, sr=None): 
    
    if mel_fb is not None and sr is not None:
        masks_linear, x_phase = reconstruct_from_mel(s, x, mel_fb, p=p)
        N, F, Q = masks_linear.shape
        output_dir_suffix = "_mel"
    else:
        x_mix = x[0]
        x_phase = np.angle(x_mix)
        N , F , Q = s.shape
        masks_linear = wiener_mask(s)
        output_dir_suffix = ""
    
    file_name = os.path.splitext(os.path.basename(path))[0]
    parent_folder = os.path.basename(os.path.dirname(path))
    main_output_dir = "output"
    source = "3_source" if "three_sources" in path else "four_sources"
    os.makedirs(main_output_dir, exist_ok=True)
    output_dir = os.path.join(main_output_dir, f"{source}_{parent_folder}_{file_name}{output_dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    for n in range(N):
        if mel_fb is not None:
            complex_spec = masks_linear[n] * np.abs(x[0]) * np.exp(1j * x_phase)
        else:
            complex_spec = masks_linear[n] * np.abs(x_mix) * np.exp(1j * x_phase)
        
        source = librosa.istft(complex_spec, n_fft=1024, hop_length=512, window="hamming")
        out_path = os.path.join(output_dir, f"source_{n+1}_wiener{output_dir_suffix}.wav")
        sf.write(out_path, source / (np.max(np.abs(source)) + 1e-8), 16000 if sr is None else sr)