# Stage 3: BIQUAD FILTER
# Objective:
# 1. Understand what a Biquad filter is
# 2. Implement it from scratch using the Direct Form I difference equation
# 3. Apply it to the gain-scaled signal from Stage 2
# 4. Visualize the filtered output vs input

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz  # only used for frequency response visualization

fs = 16000
duration = 1
t = np.linspace(0, duration, fs * duration, endpoint=False)
f = 500
signal = np.sin(2 * np.pi * f * t)

def apply_gain(signal, gain):
    return gain * signal

gain = 2
signal_gained = apply_gain(signal, gain)   # <-- this is our Stage 3 input

# ── What is a Biquad Filter?
# A Biquad ("bi-quadratic") filter is a second-order IIR (Infinite Impulse
# Response) filter. It is the fundamental building block of almost every
# audio/DSP processing chain.
#
# It is defined by this difference equation (Direct Form I):
#
#   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
#          - a1*y[n-1] - a2*y[n-2]
#
# Where:
#   x[n] = current input sample
#   y[n] = current output sample
#   b0, b1, b2 = feedforward (numerator)   coefficients
#   a1, a2     = feedback    (denominator) coefficients
#   (a0 is always normalised to 1.0)
#
# Typical filter types you can build with a Biquad:
#   - Low-pass  (passes low frequencies, cuts highs)
#   - High-pass (passes high frequencies, cuts lows)
#   - Band-pass
#   - Notch / Band-stop
#   - Peak / Shelf EQ
#
# We'll implement a LOW-PASS Biquad using the standard Audio EQ Cookbook
# formulas (Robert Bristow-Johnson).

# ── Coefficient calculation: Low-Pass Biquad
def lowpass_biquad_coeffs(fc, fs, Q=0.707):
    """
    Compute Biquad low-pass coefficients.

    Parameters
    ----------
    fc : float  — cutoff frequency in Hz
    fs : float  — sampling rate in Hz
    Q  : float  — quality factor (0.707 = Butterworth / maximally flat)

    Returns
    -------
    (b0, b1, b2, a0, a1, a2)
    """
    # Normalised angular frequency
    w0 = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * Q)

    b0 =  (1 - cos_w0) / 2
    b1 =   1 - cos_w0
    b2 =  (1 - cos_w0) / 2
    a0 =   1 + alpha          # normalisation factor
    a1 =  -2 * cos_w0
    a2 =   1 - alpha

    # Normalise everything by a0 so the difference equation uses a0 = 1
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0


# ── Core filter function ──────────────────────────────────────────────────────
def apply_biquad(signal, b0, b1, b2, a0, a1, a2):
    """
    Apply a Biquad filter sample-by-sample using Direct Form I.

    State variables:
        x_prev1, x_prev2  → x[n-1], x[n-2]
        y_prev1, y_prev2  → y[n-1], y[n-2]
    All initialised to 0 (no prior history).
    """
    N = len(signal)
    output = np.zeros(N)

    x_prev1, x_prev2 = 0.0, 0.0
    y_prev1, y_prev2 = 0.0, 0.0

    for n in range(N):
        x_n = signal[n]

        # Difference equation
        y_n = (b0 * x_n
             + b1 * x_prev1
             + b2 * x_prev2
             - a1 * y_prev1
             - a2 * y_prev2)

        output[n] = y_n

        # Shift state
        x_prev2, x_prev1 = x_prev1, x_n
        y_prev2, y_prev1 = y_prev1, y_n

    return output


# ── Apply the filter ──────────────────────────────────────────────────────────
fc = 800   # cutoff frequency in Hz — sits just above our 500 Hz tone
Q  = 0.707 # Butterworth (flattest passband)

b0, b1, b2, a0, a1, a2 = lowpass_biquad_coeffs(fc, fs, Q)

print("Biquad Coefficients (normalised):")
print(f"  b0={b0:.6f}  b1={b1:.6f}  b2={b2:.6f}")
print(f"  a0={a0:.6f}  a1={a1:.6f}  a2={a2:.6f}")

signal_filtered = apply_biquad(signal_gained, b0, b1, b2, a0, a1, a2)


# ── Plot 1: Time domain ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(t, signal)
axes[0].set_title("Stage 1 — Original 500 Hz Signal")
axes[0].set_xlim(0, 0.01)
axes[0].set_ylabel("Amplitude")
axes[0].grid()

axes[1].plot(t, signal_gained, color='orange')
axes[1].set_title(f"Stage 2 — After Gain (×{gain})")
axes[1].set_xlim(0, 0.01)
axes[1].set_ylabel("Amplitude")
axes[1].grid()

axes[2].plot(t, signal_filtered, color='green')
axes[2].set_title(f"Stage 3 — After Biquad Low-Pass Filter (fc={fc} Hz, Q={Q})")
axes[2].set_xlim(0, 0.01)
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time (s)")
axes[2].grid()

plt.tight_layout()
plt.show()


# ── Plot 2: Frequency response of the filter ──────────────────────────────────
# freqz computes H(e^jw) from the b/a coefficients
b_coeffs = [b0, b1, b2]
a_coeffs = [a0, a1, a2]

w, h = freqz(b_coeffs, a_coeffs, worN=8192, fs=fs)

plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10), color='purple')
plt.axvline(x=fc, color='red', linestyle='--', label=f'Cutoff = {fc} Hz')
plt.axvline(x=f,  color='blue', linestyle='--', label=f'Signal = {f} Hz')
plt.title("Biquad Low-Pass Filter — Frequency Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.ylim(-60, 5)
plt.legend()
plt.grid()
plt.show()