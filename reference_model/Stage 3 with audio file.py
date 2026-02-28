# Stage 3: BIQUAD FILTER — No Input File Needed
# Generates a rich multi-frequency signal to simulate real audio

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import freqz

# ── Sampling setup ────────────────────────────────────────────────────────────
fs = 44100        # CD quality — better for audio playback
duration = 3      # seconds
t = np.linspace(0, duration, fs * duration, endpoint=False)

# ── Stage 1: Generate a complex "real-like" signal ───────────────────────────
# Mix multiple sine waves at different frequencies and amplitudes
# This simulates a real audio signal with bass, mids, and highs

signal_raw = (
    0.5  * np.sin(2 * np.pi * 120  * t) +   # Bass / low rumble
    0.4  * np.sin(2 * np.pi * 440  * t) +   # Musical A4 note (mid)
    0.3  * np.sin(2 * np.pi * 1000 * t) +   # Mid presence
    0.2  * np.sin(2 * np.pi * 3000 * t) +   # Upper mid
    0.15 * np.sin(2 * np.pi * 6000 * t) +   # High frequency
    0.1  * np.sin(2 * np.pi * 10000* t)     # Bright air / treble
)

# Normalize to [-1.0, 1.0]
signal_raw = signal_raw / np.max(np.abs(signal_raw))

# ── Stage 2: Apply Gain ───────────────────────────────────────────────────────
def apply_gain(signal, gain):
    return gain * signal

gain = 2
signal_gained = apply_gain(signal_raw, gain)
signal_gained = np.clip(signal_gained, -1.0, 1.0)   # prevent clipping

# ── Biquad Coefficients ───────────────────────────────────────────────────────
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

# ── Biquad Filter ─────────────────────────────────────────────────────────────
def apply_biquad(signal, b0, b1, b2, a0, a1, a2):
    N      = len(signal)
    output = np.zeros(N)

    x_prev1 = x_prev2 = 0.0
    y_prev1 = y_prev2 = 0.0

    for n in range(N):
        x_n = signal[n]
        y_n = (b0 * x_n
             + b1 * x_prev1
             + b2 * x_prev2
             - a1 * y_prev1
             - a2 * y_prev2)

        output[n]          = y_n
        x_prev2, x_prev1   = x_prev1, x_n
        y_prev2, y_prev1   = y_prev1, y_n

    return output

# ── Apply the Filter ──────────────────────────────────────────────────────────
fc = 1000    # <-- change this to hear different effects!
Q  = 0.707

b0, b1, b2, a0, a1, a2 = lowpass_biquad_coeffs(fc, fs, Q)
signal_filtered = apply_biquad(signal_gained, b0, b1, b2, a0, a1, a2)

# ── Save WAV files so you can LISTEN to them ─────────────────────────────────
def save_wav(filename, signal, fs):
    out = np.int16(signal * 32767)
    wavfile.write(filename, fs, out)
    print(f"Saved: {filename}")

save_wav("original.wav",  signal_raw,      fs)
save_wav("gained.wav",    signal_gained,   fs)
save_wav("filtered.wav",  signal_filtered, fs)

# ── Plot 1: Time Domain ───────────────────────────────────────────────────────
view_end = 0.01   # first 10ms — enough to see waveform shape

fig, axes = plt.subplots(3, 1, figsize=(13, 8))

axes[0].plot(t, signal_raw,      color='blue')
axes[0].set_title("Stage 1 — Complex Multi-Frequency Signal (120Hz + 440Hz + 1kHz + 3kHz + 6kHz + 10kHz)")
axes[0].set_xlim(0, view_end); axes[0].set_ylabel("Amplitude"); axes[0].grid()

axes[1].plot(t, signal_gained,   color='orange')
axes[1].set_title(f"Stage 2 — After Gain (×{gain})")
axes[1].set_xlim(0, view_end); axes[1].set_ylabel("Amplitude"); axes[1].grid()

axes[2].plot(t, signal_filtered, color='green')
axes[2].set_title(f"Stage 3 — After Biquad Low-Pass Filter (fc={fc} Hz, Q={Q})")
axes[2].set_xlim(0, view_end); axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time (s)"); axes[2].grid()

plt.tight_layout()
plt.show()

# ── Plot 2: Frequency Spectrum — the most revealing plot ─────────────────────
N_fft  = len(signal_raw)
freqs  = np.fft.rfftfreq(N_fft, d=1/fs)

fft_original = np.abs(np.fft.rfft(signal_raw))
fft_filtered = np.abs(np.fft.rfft(signal_filtered))

plt.figure(figsize=(13, 5))
plt.plot(freqs, 20*np.log10(fft_original + 1e-10), color='blue',  alpha=0.8, label='Original (all freqs present)')
plt.plot(freqs, 20*np.log10(fft_filtered + 1e-10), color='green', alpha=0.8, label='Filtered (highs cut)')
plt.axvline(x=fc, color='red', linestyle='--', linewidth=2, label=f'Cutoff = {fc} Hz')

# Mark each frequency component
for freq, label in [(120,'Bass'),(440,'A4'),(1000,'Mid'),(3000,'Upper Mid'),(6000,'High'),(10000,'Treble')]:
    plt.axvline(x=freq, color='gray', linestyle=':', alpha=0.5)
    plt.text(freq, 10, label, rotation=90, fontsize=8, color='gray')

plt.title("Frequency Spectrum — Before vs After Biquad Low-Pass Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.ylim(-80, 20)
plt.legend()
plt.grid()
plt.show()