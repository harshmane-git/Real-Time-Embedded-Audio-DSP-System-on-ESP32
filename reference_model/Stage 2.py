# STAGE 2 — GAIN BLOCK
# Objective:
# 1. Implement a gain function
# 2. Applying the gain function to our signal
# 3. Verifying the amplitude scaling visually

import numpy as np
import matplotlib.pyplot as plt

# Stage 1
fs = 16000
duration = 1
t = np.linspace(0, duration, fs * duration, endpoint=False)
f = 500
signal = np.sin(2 * np.pi * f * t)

# Stage 2 Gain Block
def apply_gain(signal, gain):
    return gain * signal

gain = 2
signal_gain = apply_gain(signal, gain)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlim(0, 0.01)

plt.subplot(2, 1, 2)
plt.plot(t, signal_gain)
plt.title(f"Signal after Gain = {gain}")
plt.xlim(0, 0.01)

plt.tight_layout()
plt.show()