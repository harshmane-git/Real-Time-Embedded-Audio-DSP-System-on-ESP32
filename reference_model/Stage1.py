#Stage 1: Creation and visualization of a test signal
# Objective here: 1) Set sampling rate = 16000 Hz
      #           2) Generate a time array
      #           3) Create a 500 Hz sine wave
      #           4) Plotting it

import numpy as np
import matplotlib.pyplot as plt

# Defining sampling rate
fs = 16000 #Hz

# Time array in seconds
duration = 1
t = np.linspace(0, duration, fs * duration, endpoint=False)

# Generating a sine wave
# Why sine wave? --> SIne wave has no complexity. LIke a pure signal. Easy to generate and analyze. Represents reals audio systems.
f = 500  # Hz
signal = np.sin(2 * np.pi * f * t)

# Plotting the sine wave. A smooth sine wave oscillating at 500 Hz.
plt.figure(figsize=(10,4))
plt.plot(t, signal)
plt.title("500 Hz Sine Wave")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim(0, 0.01)
plt.grid()
plt.show()