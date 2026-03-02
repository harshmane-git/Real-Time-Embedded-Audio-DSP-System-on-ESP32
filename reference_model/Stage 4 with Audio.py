# Stage 4 — Applied to Real Audio File
# Bark-Aligned Boundaries: Bass 0–300Hz | Mid 300–2000Hz | Treble 2000–8000Hz

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd
import sounddevice as sd
import time
import os

# ── Audio Loader ──────────────────────────────────────────────────────────────
def load_audio(filepath, target_fs=16000):
    if not filepath.endswith(".wav"):
        raise ValueError("Only .wav files are supported.")
    fs, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    elif data.dtype == np.float64:
        pass
    else:
        data = data.astype(np.float64)
    if data.ndim == 2:
        print("Stereo detected → converting to mono")
        data = np.mean(data, axis=1)
    if fs != target_fs:
        g    = gcd(target_fs, fs)
        up   = target_fs // g
        down = fs // g
        data = resample_poly(data, up, down)
        print(f"Resampled : {fs} Hz → {target_fs} Hz")
        fs = target_fs
    data = data / np.max(np.abs(data) + 1e-10)
    return data, fs

# ── Biquad Coefficient Functions ──────────────────────────────────────────────
def lowpass_biquad_coeffs(fc, fs, Q=0.707):
    w0     = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha  = sin_w0 / (2 * Q)
    b0 = (1 - cos_w0) / 2
    b1 =  1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 =  1 + alpha
    a1 = -2 * cos_w0
    a2 =  1 - alpha
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

def highpass_biquad_coeffs(fc, fs, Q=0.707):
    w0     = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha  = sin_w0 / (2 * Q)
    b0 =  (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 =  (1 + cos_w0) / 2
    a0 =   1 + alpha
    a1 =  -2 * cos_w0
    a2 =   1 - alpha
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

def bandpass_biquad_coeffs(fc, fs, Q=1.0):
    w0     = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha  = sin_w0 / (2 * Q)
    b0 =  alpha
    b1 =  0
    b2 = -alpha
    a0 =  1 + alpha
    a1 = -2 * cos_w0
    a2 =  1 - alpha
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

def apply_biquad(signal, b0, b1, b2, a0, a1, a2):
    N       = len(signal)
    output  = np.zeros(N)
    x_prev1 = x_prev2 = 0.0
    y_prev1 = y_prev2 = 0.0
    for n in range(N):
        x_n = signal[n]
        y_n = (b0 * x_n
             + b1 * x_prev1
             + b2 * x_prev2
             - a1 * y_prev1
             - a2 * y_prev2)
        output[n]        = y_n
        x_prev2, x_prev1 = x_prev1, x_n
        y_prev2, y_prev1 = y_prev1, y_n
    return output

def apply_eq(signal, low_coeffs, mid_coeffs, high_coeffs,
             g_low=1.0, g_mid=1.0, g_high=1.0):
    low  = apply_biquad(signal, *low_coeffs)
    mid  = apply_biquad(signal, *mid_coeffs)
    high = apply_biquad(signal, *high_coeffs)
    y    = g_low*low + g_mid*mid + g_high*high
    y    = y / np.max(np.abs(y) + 1e-10)
    return low, mid, high, y

# ── EQ Presets ────────────────────────────────────────────────────────────────
EQ_PRESETS = {
    "Flat"        : {"g_low": 1.0, "g_mid": 1.0, "g_high": 1.0, "description": "No EQ — reference sound"},
    "Bass Boost"  : {"g_low": 1.5, "g_mid": 1.0, "g_high": 0.8, "description": "Heavy bass — EDM, hip-hop"},
    "Treble Boost": {"g_low": 0.8, "g_mid": 1.0, "g_high": 1.5, "description": "Bright, airy — acoustic, classical"},
    "Voice Boost" : {"g_low": 0.7, "g_mid": 1.5, "g_high": 0.9, "description": "Speech clarity — podcasts, calls"},
    "Bass Cut"    : {"g_low": 0.3, "g_mid": 1.0, "g_high": 1.0, "description": "Remove rumble — thin speakers"},
    "Loudness"    : {"g_low": 1.5, "g_mid": 0.8, "g_high": 1.5, "description": "V-shape — rock, metal"},
}

colors = ['blue', 'red', 'purple', 'green', 'orange', 'brown']

# ── Load File ─────────────────────────────────────────────────────────────────
filepath   = "your_audio.wav"    # <-- change to your file
signal, fs = load_audio(filepath, target_fs=16000)
t          = np.linspace(0, len(signal)/fs, len(signal), endpoint=False)

print(f"\nFile loaded   : {filepath}")
print(f"Sample rate   : {fs} Hz")
print(f"Duration      : {len(signal)/fs:.2f} seconds")
print(f"Total samples : {len(signal)}")

# ── Bark-Aligned EQ Coefficients ──────────────────────────────────────────────
# Bass   : 0    – 300  Hz  → Bark zones 1–3
# Mid    : 300  – 2000 Hz  → Bark zones 4–13
# Treble : 2000 – 8000 Hz  → Bark zones 14–22
# fc_mid : geometric mean  = sqrt(300 × 2000) = 775 ≈ 800 Hz

fc_low  = 300
fc_mid  = 800
fc_high = 2000

low_coeffs  = lowpass_biquad_coeffs(fc_low,  fs)
mid_coeffs  = bandpass_biquad_coeffs(fc_mid,  fs)
high_coeffs = highpass_biquad_coeffs(fc_high, fs)

print(f"\nBark-Aligned EQ Boundaries")
print(f"  Bass   : 0    – {fc_low}  Hz  (Bark zones 1–3)")
print(f"  Mid    : {fc_low}  – {fc_high} Hz  (Bark zones 4–13)")
print(f"  Treble : {fc_high} – 8000 Hz  (Bark zones 14–22)")

# ── Output Folder ─────────────────────────────────────────────────────────────
output_folder = "eq_outputs"
os.makedirs(output_folder, exist_ok=True)

# ── Process All Presets ───────────────────────────────────────────────────────
results     = {}
saved_files = []

print(f"\nProcessing all presets...")
print("=" * 55)

for preset_name, params in EQ_PRESETS.items():
    _, _, _, mixed = apply_eq(
        signal, low_coeffs, mid_coeffs, high_coeffs,
        g_low  = params["g_low"],
        g_mid  = params["g_mid"],
        g_high = params["g_high"]
    )
    results[preset_name] = mixed
    filename = f"{output_folder}/{preset_name.replace(' ', '_')}.wav"
    wavfile.write(filename, fs, np.int16(mixed * 32767))
    saved_files.append((preset_name, filename))
    print(f"  ✔ {preset_name:<15} → saved: {filename}")

print(f"\nAll presets saved to: ./{output_folder}/")

# ── Playback ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PLAYBACK — Original then each preset")
print("=" * 55)

print(f"\n▶  Playing ORIGINAL")
sd.play(signal.astype(np.float32), fs)
sd.wait()
time.sleep(0.5)

for preset_name in EQ_PRESETS:
    print(f"\n▶  Playing: {preset_name}  —  {EQ_PRESETS[preset_name]['description']}")
    sd.play(results[preset_name].astype(np.float32), fs)
    sd.wait()
    time.sleep(0.5)

print("\n✔  Playback complete")

# ── Plot 1: Input vs Each Preset Output — Waveform ───────────────────────────
view_end    = min(0.05, len(signal)/fs)
preset_list = list(EQ_PRESETS.keys())

fig, axes = plt.subplots(len(EQ_PRESETS) + 1, 1,
                         figsize=(13, 3 * (len(EQ_PRESETS) + 1)))
fig.suptitle("Stage 4 Applied to Audio File — Input vs Output Waveforms\n"
             "Bark-Aligned: Bass 0–300Hz | Mid 300–2000Hz | Treble 2000–8000Hz",
             fontsize=12, fontweight='bold')

axes[0].plot(t, signal, color='black', linewidth=0.6)
axes[0].set_title(f"INPUT — {filepath}")
axes[0].set_xlim(0, view_end)
axes[0].set_ylim(-1.1, 1.1)
axes[0].set_ylabel("Amplitude")
axes[0].grid()

for i, preset_name in enumerate(preset_list):
    axes[i+1].plot(t, results[preset_name], color=colors[i], linewidth=0.6)
    axes[i+1].set_title(
        f"OUTPUT — {preset_name}  "
        f"(g_low={EQ_PRESETS[preset_name]['g_low']}  "
        f"g_mid={EQ_PRESETS[preset_name]['g_mid']}  "
        f"g_high={EQ_PRESETS[preset_name]['g_high']})"
    )
    axes[i+1].set_xlim(0, view_end)
    axes[i+1].set_ylim(-1.1, 1.1)
    axes[i+1].set_ylabel("Amplitude")
    axes[i+1].grid()

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()

# ── Plot 2: FFT — Input vs All Preset Outputs ─────────────────────────────────
freqs     = np.fft.rfftfreq(len(signal), d=1/fs)
fft_input = np.abs(np.fft.rfft(signal))

plt.figure(figsize=(13, 5))
plt.plot(freqs, 20*np.log10(fft_input + 1e-10),
         color='black', linewidth=2, alpha=0.8, label='INPUT', zorder=5)

for i, preset_name in enumerate(preset_list):
    fft_out = np.abs(np.fft.rfft(results[preset_name]))
    plt.plot(freqs, 20*np.log10(fft_out + 1e-10),
             color=colors[i], alpha=0.8, linewidth=1.5, label=preset_name)

plt.axvline(x=fc_low,  color='gray', linestyle=':',  alpha=0.5,
            label=f'fc_low  = {fc_low}Hz  (Bark zone 3)')
plt.axvline(x=fc_high, color='gray', linestyle='--', alpha=0.5,
            label=f'fc_high = {fc_high}Hz (Bark zone 13)')
plt.title("Stage 4 Applied to Audio File — FFT: Input vs All Preset Outputs\n"
          "Bark-Aligned Boundaries")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2)
plt.ylim(-80, 20)
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nSaved Files Summary")
print("=" * 55)
for preset_name, filename in saved_files:
    print(f"  {preset_name:<15} → {filename}")