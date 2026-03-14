# Stage 4 — Applied to Real Audio File
# 4th Order Biquad EQ (Two Cascaded 2nd Order Sections per Band)
# Bark-Aligned Boundaries: Bass 0–300Hz | Mid 300–2000Hz | Treble 2000–8000Hz

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, freqz
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

# ── 2nd Order Biquad Coefficient Functions ────────────────────────────────────
def lowpass_biquad_coeffs(fc, fs, Q=0.707):
    w0=2*np.pi*fc/fs; c=np.cos(w0); s=np.sin(w0); a=s/(2*Q)
    b0=(1-c)/2; b1=1-c; b2=(1-c)/2; a0=1+a; a1=-2*c; a2=1-a
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

def highpass_biquad_coeffs(fc, fs, Q=0.707):
    w0=2*np.pi*fc/fs; c=np.cos(w0); s=np.sin(w0); a=s/(2*Q)
    b0=(1+c)/2; b1=-(1+c); b2=(1+c)/2; a0=1+a; a1=-2*c; a2=1-a
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

def bandpass_biquad_coeffs(fc, fs, Q=1.0):
    w0=2*np.pi*fc/fs; c=np.cos(w0); s=np.sin(w0); a=s/(2*Q)
    b0=a; b1=0; b2=-a; a0=1+a; a1=-2*c; a2=1-a
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

# ── Single 2nd Order Biquad Section ──────────────────────────────────────────
def apply_biquad(signal, b0, b1, b2, a0, a1, a2):
    N=len(signal); out=np.zeros(N); x1=x2=y1=y2=0.0
    for n in range(N):
        x=signal[n]; y=b0*x+b1*x1+b2*x2-a1*y1-a2*y2
        out[n]=y; x2,x1=x1,x; y2,y1=y1,y
    return out

# ── 4th Order Biquad — Two Cascaded Sections ─────────────────────────────────
# Signal passes through Section 1 then Section 2
# Each section is a standard 2nd order Biquad
# Result: -80 dB/decade roll-off vs -40 dB/decade for 2nd order
def apply_biquad_4th(signal, coeffs_s1, coeffs_s2):
    stage1 = apply_biquad(signal, *coeffs_s1)
    stage2 = apply_biquad(stage1, *coeffs_s2)
    return stage2

# ── 4th Order Coefficient Generator ──────────────────────────────────────────
# Butterworth 4th order Q values for the two sections:
# Section 1: Q = 0.5412  (from Butterworth pole at pi/8)
# Section 2: Q = 1.3066  (from Butterworth pole at 3*pi/8)
Q1 = 0.5412
Q2 = 1.3066

def lowpass_4th_order(fc, fs):
    s1 = lowpass_biquad_coeffs(fc, fs, Q=Q1)
    s2 = lowpass_biquad_coeffs(fc, fs, Q=Q2)
    return s1, s2

def highpass_4th_order(fc, fs):
    s1 = highpass_biquad_coeffs(fc, fs, Q=Q1)
    s2 = highpass_biquad_coeffs(fc, fs, Q=Q2)
    return s1, s2

def bandpass_4th_order(fc, fs):
    s1 = bandpass_biquad_coeffs(fc, fs, Q=Q1*2)
    s2 = bandpass_biquad_coeffs(fc, fs, Q=Q2*2)
    return s1, s2

# ── 4th Order EQ ──────────────────────────────────────────────────────────────
def apply_eq_4th(signal,
                 lp_s1, lp_s2,
                 bp_s1, bp_s2,
                 hp_s1, hp_s2,
                 g_low=1.0, g_mid=1.0, g_high=1.0):
    low  = apply_biquad_4th(signal, lp_s1, lp_s2)
    mid  = apply_biquad_4th(signal, bp_s1, bp_s2)
    high = apply_biquad_4th(signal, hp_s1, hp_s2)
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

# ── Bark-Aligned 4th Order EQ Coefficients ────────────────────────────────────
# Bass   : 0    – 300  Hz  → Bark zones 1–3
# Mid    : 300  – 2000 Hz  → Bark zones 4–13
# Treble : 2000 – 8000 Hz  → Bark zones 14–22
# fc_mid : geometric mean  = sqrt(300 × 2000) = 775 ≈ 800 Hz
# Filter order: 4th (two cascaded 2nd order Biquad sections per band)
# Roll-off: -80 dB/decade (vs -40 dB/decade for 2nd order)

fc_low  = 300
fc_mid  = 800
fc_high = 2000

lp_s1, lp_s2 = lowpass_4th_order(fc_low,  fs)
bp_s1, bp_s2 = bandpass_4th_order(fc_mid,  fs)
hp_s1, hp_s2 = highpass_4th_order(fc_high, fs)

print(f"\n4th Order Bark-Aligned EQ Boundaries")
print(f"  Bass   : 0    – {fc_low}  Hz  (Bark zones 1–3)")
print(f"  Mid    : {fc_low}  – {fc_high} Hz  (Bark zones 4–13)")
print(f"  Treble : {fc_high} – 8000 Hz  (Bark zones 14–22)")
print(f"  Filter : 4th order (2 cascaded Biquad sections per band)")
print(f"  Roll-off: -80 dB/decade")

# ── Coefficient Report for Umesh ──────────────────────────────────────────────
print()
print("4th Order Coefficients for C Implementation (Umesh)")
print("=" * 60)
print("// fs = 16000 Hz | Bark-Aligned | 4th Order Butterworth")
print("// Format per section: b0, b1, b2, a1, a2")
print("// Signal passes: input → section1 → section2 → output")
print()
print(f"// Low-Pass 300 Hz (Bass) — 4th Order")
print(f"static const float lp_s1[5] = {{{lp_s1[0]:.8f}f, {lp_s1[1]:.8f}f, {lp_s1[2]:.8f}f, {lp_s1[4]:.8f}f, {lp_s1[5]:.8f}f}};")
print(f"static const float lp_s2[5] = {{{lp_s2[0]:.8f}f, {lp_s2[1]:.8f}f, {lp_s2[2]:.8f}f, {lp_s2[4]:.8f}f, {lp_s2[5]:.8f}f}};")
print()
print(f"// Band-Pass 800 Hz (Mid) — 4th Order")
print(f"static const float bp_s1[5] = {{{bp_s1[0]:.8f}f, {bp_s1[1]:.8f}f, {bp_s1[2]:.8f}f, {bp_s1[4]:.8f}f, {bp_s1[5]:.8f}f}};")
print(f"static const float bp_s2[5] = {{{bp_s2[0]:.8f}f, {bp_s2[1]:.8f}f, {bp_s2[2]:.8f}f, {bp_s2[4]:.8f}f, {bp_s2[5]:.8f}f}};")
print()
print(f"// High-Pass 2000 Hz (Treble) — 4th Order")
print(f"static const float hp_s1[5] = {{{hp_s1[0]:.8f}f, {hp_s1[1]:.8f}f, {hp_s1[2]:.8f}f, {hp_s1[4]:.8f}f, {hp_s1[5]:.8f}f}};")
print(f"static const float hp_s2[5] = {{{hp_s2[0]:.8f}f, {hp_s2[1]:.8f}f, {hp_s2[2]:.8f}f, {hp_s2[4]:.8f}f, {hp_s2[5]:.8f}f}};")

# ── Output Folder ─────────────────────────────────────────────────────────────
output_folder = "eq_outputs_4th_order"
os.makedirs(output_folder, exist_ok=True)

# ── Process All Presets ───────────────────────────────────────────────────────
results     = {}
saved_files = []

print(f"\nProcessing all presets...")
print("=" * 55)

for preset_name, params in EQ_PRESETS.items():
    _, _, _, mixed = apply_eq_4th(
        signal,
        lp_s1, lp_s2,
        bp_s1, bp_s2,
        hp_s1, hp_s2,
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
fig.suptitle("Stage 4 — 4th Order EQ Applied to Audio File\n"
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
plt.title("Stage 4 — 4th Order EQ FFT: Input vs All Preset Outputs\n"
          "Bark-Aligned Boundaries | -80 dB/decade roll-off")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2)
plt.ylim(-80, 20)
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()

# ── Plot 3: Filter Response Comparison — 2nd vs 4th Order ────────────────────
# Shows exactly why 4th order is better — steeper roll-off visible clearly
from scipy.signal import freqz

plt.figure(figsize=(13, 6))

# 2nd order responses (for comparison)
lp_2nd = lowpass_biquad_coeffs(fc_low,  fs)
bp_2nd = bandpass_biquad_coeffs(fc_mid,  fs)
hp_2nd = highpass_biquad_coeffs(fc_high, fs)

for coeffs, label, color in [
    (lp_2nd, f"LP {fc_low}Hz  2nd order", 'blue'),
    (bp_2nd, f"BP {fc_mid}Hz  2nd order", 'orange'),
    (hp_2nd, f"HP {fc_high}Hz 2nd order", 'green'),
]:
    w, h = freqz(list(coeffs[:3]), [coeffs[3], coeffs[4], coeffs[5]], worN=8000, fs=fs)
    plt.plot(w, 20*np.log10(np.abs(h)+1e-10),
             color=color, linewidth=1.2, linestyle='--', alpha=0.6, label=label)

# 4th order responses
for (s1, s2), label, color in [
    ((lp_s1, lp_s2), f"LP {fc_low}Hz  4th order", 'blue'),
    ((bp_s1, bp_s2), f"BP {fc_mid}Hz  4th order", 'orange'),
    ((hp_s1, hp_s2), f"HP {fc_high}Hz 4th order", 'green'),
]:
    # Cascade frequency responses: H_total(z) = H1(z) × H2(z)
    w,  h1 = freqz(list(s1[:3]), [s1[3], s1[4], s1[5]], worN=8000, fs=fs)
    w,  h2 = freqz(list(s2[:3]), [s2[3], s2[4], s2[5]], worN=8000, fs=fs)
    h_total = h1 * h2
    plt.plot(w, 20*np.log10(np.abs(h_total)+1e-10),
             color=color, linewidth=2.0, linestyle='-', label=label)

plt.axvline(x=fc_low,  color='gray', linestyle=':',  alpha=0.5, label=f'{fc_low}Hz')
plt.axvline(x=fc_high, color='gray', linestyle='--', alpha=0.5, label=f'{fc_high}Hz')
plt.title("Filter Response: 2nd Order (dashed) vs 4th Order (solid)\n"
          "4th order has steeper roll-off — cleaner band separation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2)
plt.ylim(-80, 5)
plt.legend(loc='upper right', fontsize=8)
plt.grid()
plt.tight_layout()
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nSaved Files Summary")
print("=" * 55)
for preset_name, filename in saved_files:
    print(f"  {preset_name:<15} → {filename}")