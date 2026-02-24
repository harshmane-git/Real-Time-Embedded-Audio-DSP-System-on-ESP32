# Stage 4

import numpy as np
import matplotlib.pyplot as plt

# ── Sampling Rate ─────────────────────────────────────────────────────────────
fs = 16000
duration = 1
t = np.linspace(0, duration, fs * duration, endpoint=False)

# ── Coefficient Functions ─────────────────────────────────────────────────────
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

# ── EQ Coefficients ───────────────────────────────────────────────────────────
low_coeffs  = lowpass_biquad_coeffs(300,  fs)
mid_coeffs  = bandpass_biquad_coeffs(1000, fs)
high_coeffs = highpass_biquad_coeffs(3000, fs)

# ── EQ Presets ────────────────────────────────────────────────────────────────
EQ_PRESETS = {
    "Flat"         : {"g_low": 1.0, "g_mid": 1.0, "g_high": 1.0},
    "Bass Boost"   : {"g_low": 1.5, "g_mid": 1.0, "g_high": 0.8},
    "Treble Boost" : {"g_low": 0.8, "g_mid": 1.0, "g_high": 1.5},
    "Voice Boost"  : {"g_low": 0.7, "g_mid": 1.5, "g_high": 0.9},
    "Loudness"     : {"g_low": 1.5, "g_mid": 0.8, "g_high": 1.5},
}

colors = ['blue', 'red', 'purple', 'green', 'orange']

# ════════════════════════════════════════════════════════════════════════════════
# STEP 9A — TEST WITH MIXED SINE SIGNAL
# Input: 100 Hz + 1000 Hz + 4000 Hz mixed
# Observe: how each preset shapes the tone differently
# ════════════════════════════════════════════════════════════════════════════════

print("Step 9A — Mixed Sine Signal Test")
print("=" * 55)

# Input signal
sine_signal = (
    np.sin(2 * np.pi * 100  * t) +   # Bass
    np.sin(2 * np.pi * 1000 * t) +   # Mid
    np.sin(2 * np.pi * 4000 * t)     # Treble
)
sine_signal = sine_signal / np.max(np.abs(sine_signal))

# ── Plot 1: Input vs Output — Time Domain (per preset) ───────────────────────
view_end    = 0.02
n_presets   = len(EQ_PRESETS)
preset_list = list(EQ_PRESETS.keys())

fig, axes = plt.subplots(n_presets + 1, 1, figsize=(13, 3 * (n_presets + 1)))
fig.suptitle("Step 9A — Input vs Output: Mixed Sine Signal (Time Domain)",
             fontsize=12, fontweight='bold')

# Row 0: Input signal
axes[0].plot(t, sine_signal, color='black', linewidth=0.8)
axes[0].set_title("INPUT — 100Hz + 1000Hz + 4000Hz")
axes[0].set_xlim(0, view_end)
axes[0].set_ylim(-1.1, 1.1)
axes[0].set_ylabel("Amplitude")
axes[0].grid()

# Rows 1–N: Each preset output
for i, preset_name in enumerate(preset_list):
    p = EQ_PRESETS[preset_name]
    _, _, _, mixed = apply_eq(
        sine_signal, low_coeffs, mid_coeffs, high_coeffs,
        g_low=p["g_low"], g_mid=p["g_mid"], g_high=p["g_high"]
    )
    axes[i+1].plot(t, mixed, color=colors[i], linewidth=0.8)
    axes[i+1].set_title(
        f"OUTPUT — {preset_name}  "
        f"(g_low={p['g_low']}  g_mid={p['g_mid']}  g_high={p['g_high']})"
    )
    axes[i+1].set_xlim(0, view_end)
    axes[i+1].set_ylim(-1.1, 1.1)
    axes[i+1].set_ylabel("Amplitude")
    axes[i+1].grid()

    print(f"  {preset_name:<15} → output peak: {np.max(np.abs(mixed)):.4f}")

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()

# ── Plot 2: FFT — Input vs All Preset Outputs ─────────────────────────────────
freqs     = np.fft.rfftfreq(len(sine_signal), d=1/fs)
fft_input = np.abs(np.fft.rfft(sine_signal))

plt.figure(figsize=(13, 5))
plt.plot(freqs,
         20*np.log10(fft_input + 1e-10),
         color='black', linewidth=2.5,
         alpha=0.9, label='INPUT', zorder=5)

for i, preset_name in enumerate(preset_list):
    p = EQ_PRESETS[preset_name]
    _, _, _, mixed = apply_eq(
        sine_signal, low_coeffs, mid_coeffs, high_coeffs,
        g_low=p["g_low"], g_mid=p["g_mid"], g_high=p["g_high"]
    )
    fft_out = np.abs(np.fft.rfft(mixed))
    plt.plot(freqs,
             20*np.log10(fft_out + 1e-10),
             color=colors[i], alpha=0.8,
             linewidth=1.5, label=preset_name)

plt.axvline(x=300,  color='gray', linestyle=':',  alpha=0.6, label='fc_low=300Hz')
plt.axvline(x=3000, color='gray', linestyle='--', alpha=0.6, label='fc_high=3000Hz')
plt.title("Step 9A — FFT: Input vs All Preset Outputs (Sine Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2)
plt.ylim(-80, 20)
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# STEP 9B — TEST WITH WHITE NOISE
# White noise contains ALL frequencies at EQUAL energy
# This is the best signal for seeing frequency shaping clearly
# After EQ: boosted bands rise, cut bands fall in the FFT
# ════════════════════════════════════════════════════════════════════════════════

print("\nStep 9B — White Noise Test")
print("=" * 55)
print("White noise = equal energy at ALL frequencies")
print("Perfect for visualizing how each preset shapes the spectrum")

# Generate white noise
np.random.seed(42)   # fixed seed for reproducibility
noise = np.random.uniform(-1.0, 1.0, fs * duration)
noise = noise / np.max(np.abs(noise))

# ── Plot 3: White Noise — Input vs Output Time Domain ────────────────────────
fig, axes = plt.subplots(n_presets + 1, 1, figsize=(13, 3 * (n_presets + 1)))
fig.suptitle("Step 9B — White Noise: Input vs Output (Time Domain)",
             fontsize=12, fontweight='bold')

axes[0].plot(t, noise, color='black', linewidth=0.5, alpha=0.8)
axes[0].set_title("INPUT — White Noise (all frequencies equal)")
axes[0].set_xlim(0, 0.05)
axes[0].set_ylim(-1.1, 1.1)
axes[0].set_ylabel("Amplitude")
axes[0].grid()

for i, preset_name in enumerate(preset_list):
    p = EQ_PRESETS[preset_name]
    _, _, _, mixed = apply_eq(
        noise, low_coeffs, mid_coeffs, high_coeffs,
        g_low=p["g_low"], g_mid=p["g_mid"], g_high=p["g_high"]
    )
    axes[i+1].plot(t, mixed, color=colors[i], linewidth=0.5, alpha=0.8)
    axes[i+1].set_title(
        f"OUTPUT — {preset_name}  "
        f"(g_low={p['g_low']}  g_mid={p['g_mid']}  g_high={p['g_high']})"
    )
    axes[i+1].set_xlim(0, 0.05)
    axes[i+1].set_ylim(-1.1, 1.1)
    axes[i+1].set_ylabel("Amplitude")
    axes[i+1].grid()

    print(f"  {preset_name:<15} → output peak: {np.max(np.abs(mixed)):.4f}")

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()

# ── Plot 4: White Noise FFT — THE most revealing plot ────────────────────────
# With white noise input, the output FFT directly shows the EQ curve shape
# Flat input → any bumps/dips in output = pure EQ effect

freqs      = np.fft.rfftfreq(len(noise), d=1/fs)
fft_noise  = np.abs(np.fft.rfft(noise))

plt.figure(figsize=(13, 6))
plt.plot(freqs,
         20*np.log10(fft_noise + 1e-10),
         color='black', linewidth=1.5,
         alpha=0.6, label='INPUT (White Noise)', zorder=5)

for i, preset_name in enumerate(preset_list):
    p = EQ_PRESETS[preset_name]
    _, _, _, mixed = apply_eq(
        noise, low_coeffs, mid_coeffs, high_coeffs,
        g_low=p["g_low"], g_mid=p["g_mid"], g_high=p["g_high"]
    )
    fft_out = np.abs(np.fft.rfft(mixed))
    plt.plot(freqs,
             20*np.log10(fft_out + 1e-10),
             color=colors[i], alpha=0.85,
             linewidth=2, label=preset_name)

plt.axvline(x=300,  color='gray', linestyle=':',  alpha=0.6, label='fc_low  = 300Hz')
plt.axvline(x=3000, color='gray', linestyle='--', alpha=0.6, label='fc_high = 3000Hz')
plt.title("Step 9B — White Noise FFT: EQ Frequency Shaping\n"
          "(Flat input → output shape = pure EQ curve)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2)
plt.ylim(-60, 20)
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()