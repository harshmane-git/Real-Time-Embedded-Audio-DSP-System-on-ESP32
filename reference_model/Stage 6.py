# Stage 6 — Full DSP Pipeline with Limiter
# Reference Model — ESP32 Audio DSP Project
# Pipeline: Mic (I2S) → Gain → 3-Band IIR EQ → Delay → Limiter → Speaker (I2S)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, freqz
from math import gcd
import sounddevice as sd
import time

# ── Audio Loader with Resampling ──────────────────────────────────────────────
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

# ── Stage 2: Gain ─────────────────────────────────────────────────────────────
def apply_gain(signal, gain):
    return gain * signal

# ── Stage 4: 3-Band IIR EQ (Biquad) ──────────────────────────────────────────
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
    return g_low*low + g_mid*mid + g_high*high

# ── Stage 5: Delay (Circular Buffer) ─────────────────────────────────────────
class CircularBuffer:
    def __init__(self, size):
        self.buffer = np.zeros(size)
        self.size   = size
        self.index  = 0

    def read(self):
        return self.buffer[self.index]

    def write(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size

def apply_delay(signal, delay_samples, feedback=0.5, mix=0.5):
    N      = len(signal)
    output = np.zeros(N)
    buf    = CircularBuffer(delay_samples)
    for n in range(N):
        x_n       = signal[n]
        delayed   = buf.read()
        output[n] = (1 - mix) * x_n + mix * delayed
        buf.write(x_n + feedback * delayed)
    return output

# ── Stage 6: Limiter ──────────────────────────────────────────────────────────
def apply_limiter(signal, threshold=0.9):
    return np.clip(signal, -threshold, threshold)

# ── EQ Presets (GPIO Switch 1) ────────────────────────────────────────────────
EQ_PRESETS = {
    "Flat"        : {"g_low": 1.0, "g_mid": 1.0, "g_high": 1.0},
    "Bass Boost"  : {"g_low": 1.5, "g_mid": 1.0, "g_high": 0.8},
    "Treble Boost": {"g_low": 0.8, "g_mid": 1.0, "g_high": 1.5},
    "Voice Boost" : {"g_low": 0.7, "g_mid": 1.5, "g_high": 0.9},
    "Loudness"    : {"g_low": 1.5, "g_mid": 0.8, "g_high": 1.5},
}

# ── Delay Presets (GPIO Switch 2) ─────────────────────────────────────────────
DELAY_PRESETS = {
    "Short" : {"delay_ms": 125, "feedback": 0.3, "mix": 0.4},
    "Medium": {"delay_ms": 250, "feedback": 0.4, "mix": 0.5},
    "Long"  : {"delay_ms": 500, "feedback": 0.5, "mix": 0.5},
    "Speech" : {"delay_ms": 200, "feedback": 0.2, "mix": 0.2},
}

# ── Load File ─────────────────────────────────────────────────────────────────
filepath        = "581109__realdavidfloat__120-bpm-kick-snare-thumper-double-claps.wav"   # <-- change to your file
signal, fs      = load_audio(filepath, target_fs=16000)
t               = np.linspace(0, len(signal) / fs, len(signal), endpoint=False)

# ── Pipeline Configuration ────────────────────────────────────────────────────
block_size      = 256
gain            = 2.0
eq_preset       = EQ_PRESETS["Loudness"]
delay_preset    = DELAY_PRESETS["Medium"]
threshold       = 0.9

delay_ms        = delay_preset["delay_ms"]
feedback        = delay_preset["feedback"]
mix             = delay_preset["mix"]
delay_samples   = int((delay_ms / 1000) * fs)

low_coeffs      = lowpass_biquad_coeffs(300,  fs)
mid_coeffs      = bandpass_biquad_coeffs(1000, fs)
high_coeffs     = highpass_biquad_coeffs(3000, fs)

# ── Pipeline Execution ────────────────────────────────────────────────────────
stage_gain      = apply_gain(signal, gain)
stage_eq        = apply_eq(stage_gain, low_coeffs, mid_coeffs, high_coeffs, **eq_preset)
stage_delay     = apply_delay(stage_eq, delay_samples, feedback, mix)
stage_limiter   = apply_limiter(stage_delay, threshold)

# ── Console Output ────────────────────────────────────────────────────────────
clipped         = np.sum(np.abs(stage_delay) > threshold)

print("ESP32 Audio DSP — Reference Model")
print("=" * 55)
print(f"  File              : {filepath}")
print(f"  Sample rate       : {fs} Hz")
print(f"  Block size        : {block_size} samples")
print(f"  Duration          : {len(signal)/fs:.2f} seconds")
print(f"  EQ Preset         : Loudness")
print(f"  Delay Preset      : Medium ({delay_ms}ms)")
print("─" * 55)
print(f"  Input  peak       : {np.max(np.abs(signal)):.4f}")
print(f"  After Gain        : {np.max(np.abs(stage_gain)):.4f}")
print(f"  After EQ          : {np.max(np.abs(stage_eq)):.4f}")
print(f"  After Delay       : {np.max(np.abs(stage_delay)):.4f}")
print(f"  After Limiter     : {np.max(np.abs(stage_limiter)):.4f}  ✔")
print("─" * 55)
print(f"  Clipped samples   : {clipped} ({clipped/len(signal)*100:.2f}%)")
print("─" * 55)
print(f"  Theoretical latency        : {block_size/fs*1000:.2f} ms")
print(f"  Multiplications/sample     : 15  (5 per Biquad × 3 bands)")
print(f"  Multiplications/block      : {15 * block_size}")
print(f"  Delay buffer memory        : {delay_samples * 4} bytes")

# ── Save ──────────────────────────────────────────────────────────────────────
wavfile.write("pipeline_output.wav", fs, np.int16(stage_limiter * 32767))
print(f"\n  Saved             : pipeline_output.wav")

# ── Playback ──────────────────────────────────────────────────────────────────
print("\n▶  Playing: ORIGINAL")
sd.play(signal.astype(np.float32), fs)
sd.wait()
time.sleep(0.5)

print("▶  Playing: PIPELINE OUTPUT")
sd.play(stage_limiter.astype(np.float32), fs)
sd.wait()
print("✔  Playback complete")

# ── Plot 1: All Pipeline Stages ───────────────────────────────────────────────
view_end = min(3.0, len(signal) / fs)
stages   = [
    (signal,        "Input",                   'black'),
    (stage_gain,    f"Gain  ×{gain}",          'blue'),
    (stage_eq,      "EQ  Loudness",            'orange'),
    (stage_delay,   f"Delay  {delay_ms}ms",    'purple'),
    (stage_limiter, f"Limiter  ±{threshold}",  'green'),
]

fig, axes = plt.subplots(len(stages), 1, figsize=(13, 3 * len(stages)))
fig.suptitle("ESP32 Audio DSP — Reference Model: All Pipeline Stages  |  fs = 16000 Hz",
             fontweight='bold')

for i, (data, label, color) in enumerate(stages):
    axes[i].plot(t, data, color=color, linewidth=0.6)
    axes[i].axhline(y= threshold, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[i].axhline(y=-threshold, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[i].set_title(f"{label}  |  peak = {np.max(np.abs(data)):.4f}")
    axes[i].set_xlim(0, view_end)
    axes[i].set_ylim(-2.2, 2.2)
    axes[i].set_ylabel("Amplitude")
    axes[i].grid()

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()

# ── Plot 2: Limiter — Before vs After ─────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle("ESP32 Audio DSP — Limiter: Before vs After", fontweight='bold')

axes[0].plot(t, stage_delay, color='purple', linewidth=0.6)
axes[0].axhline(y= threshold, color='red', linestyle='--', linewidth=1.2, label=f'±{threshold}')
axes[0].axhline(y=-threshold, color='red', linestyle='--', linewidth=1.2)
axes[0].set_title(f"Before Limiter  |  peak = {np.max(np.abs(stage_delay)):.4f}")
axes[0].set_xlim(0, view_end)
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid()

axes[1].plot(t, stage_limiter, color='green', linewidth=0.6)
axes[1].axhline(y= threshold, color='red', linestyle='--', linewidth=1.2, label=f'±{threshold}')
axes[1].axhline(y=-threshold, color='red', linestyle='--', linewidth=1.2)
axes[1].set_title(f"After Limiter  |  peak = {np.max(np.abs(stage_limiter)):.4f}  |  clipped = {clipped} samples")
axes[1].set_xlim(0, view_end)
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (seconds)")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# ── Plot 3: FFT — Input vs Output ─────────────────────────────────────────────
freqs      = np.fft.rfftfreq(len(signal), d=1/fs)
fft_input  = np.abs(np.fft.rfft(signal))
fft_output = np.abs(np.fft.rfft(stage_limiter))

plt.figure(figsize=(13, 5))
plt.plot(freqs, 20*np.log10(fft_input  + 1e-10),
         color='black', alpha=0.8, linewidth=1.2, label='Input')
plt.plot(freqs, 20*np.log10(fft_output + 1e-10),
         color='green', alpha=0.8, linewidth=1.2, label='Pipeline Output')
plt.axvline(x=300,  color='gray', linestyle=':',  alpha=0.6, label='fc_low  = 300Hz')
plt.axvline(x=3000, color='gray', linestyle='--', alpha=0.6, label='fc_high = 3000Hz')
plt.title("ESP32 Audio DSP — Frequency Spectrum: Input vs Pipeline Output")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.ylim(-80, 20)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ── Plot 4: Biquad Filter Frequency Responses ─────────────────────────────────
def plot_filter_response(coeffs, label, color, fs):
    b = [coeffs[0], coeffs[1], coeffs[2]]
    a = [coeffs[3], coeffs[4], coeffs[5]]
    w, h = freqz(b, a, worN=8000, fs=fs)
    plt.plot(w, 20*np.log10(np.abs(h) + 1e-10),
             color=color, linewidth=1.5, label=label)

plt.figure(figsize=(13, 5))
plot_filter_response(low_coeffs,  "Low-Pass  300Hz  (Bass)",  'blue',   fs)
plot_filter_response(mid_coeffs,  "Band-Pass 1000Hz (Mid)",   'orange', fs)
plot_filter_response(high_coeffs, "High-Pass 3000Hz (Treble)",'green',  fs)
plt.axvline(x=300,  color='blue',  linestyle='--', alpha=0.4)
plt.axvline(x=1000, color='orange',linestyle='--', alpha=0.4)
plt.axvline(x=3000, color='green', linestyle='--', alpha=0.4)
plt.title("ESP32 Audio DSP — Biquad Filter Frequency Responses  |  fs = 16000 Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.ylim(-40, 5)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ── Performance Analysis ───────────────────────────────────────────────────────
print("\nPerformance Analysis")
print("=" * 55)
print(f"  Sampling rate              : {fs} Hz")
print(f"  Block size                 : {block_size} samples")
print(f"  Theoretical latency        : {block_size/fs*1000:.2f} ms")
print(f"  Multiplications/sample     : 15  (5 per Biquad × 3 bands)")
print(f"  Multiplications/block      : {15 * block_size}")
print(f"  Delay buffer memory        : {delay_samples * 4} bytes (float32)")
print(f"  EQ preset active           : Loudness")
print(f"  Delay preset active        : Medium ({delay_ms}ms)")

# Input → Gain (×2) → EQ (Loudness) → Delay (250ms) → Limiter (±0.9) → Output