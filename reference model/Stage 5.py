# Stage 5 — Delay Block (Echo Effect)

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
fs            = 16000
delay_ms      = 500
feedback      = 0.5
mix           = 0.5
delay_samples = int((delay_ms / 1000) * fs)

# ── Circular Buffer ───────────────────────────────────────────────────────────
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

    def read_write(self, value):
        delayed = self.read()
        self.write(value)
        return delayed

# ── Delay Block ───────────────────────────────────────────────────────────────
def apply_delay(signal, delay_samples, feedback=0.5, mix=0.5):
    N      = len(signal)
    output = np.zeros(N)
    buf    = CircularBuffer(delay_samples)

    for n in range(N):
        x_n     = signal[n]
        delayed = buf.read()
        y_n     = x_n + mix * delayed
        buf.write(x_n + feedback * delayed)
        output[n] = y_n

    return output

# ── Test Signals ──────────────────────────────────────────────────────────────
duration = 3
t        = np.linspace(0, duration, fs * duration, endpoint=False)

# Impulse
impulse        = np.zeros(fs * duration)
impulse[0]     = 1.0
impulse_out    = apply_delay(impulse, delay_samples, feedback, mix)

# Sine burst
sine_burst     = np.zeros(fs * duration)
n_burst        = int(0.2 * fs)
sine_burst[:n_burst] = np.sin(2 * np.pi * 440 * t[:n_burst])
sine_out       = apply_delay(sine_burst, delay_samples, feedback, mix)

# ── Verification ──────────────────────────────────────────────────────────────
echo_positions = np.where(np.abs(impulse_out) > 0.001)[0]

print("Stage 5 — Delay Block")
print("=" * 45)
print(f"  Delay          : {delay_ms} ms = {delay_samples} samples")
print(f"  Feedback       : {feedback}")
print(f"  Mix            : {mix}")
print(f"  Echo at sample : {echo_positions[1]}  (expected: {delay_samples}) ✔")

# ── Plot 1: Impulse Response ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle("Stage 5 — Impulse Response", fontweight='bold')

axes[0].stem(range(fs), impulse[:fs],
             linefmt='blue', markerfmt='bo', basefmt='k-')
axes[0].set_title("Input — Impulse")
axes[0].set_xlabel("Samples")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, fs)
axes[0].grid()

axes[1].stem(range(fs), impulse_out[:fs],
             linefmt='green', markerfmt='go', basefmt='k-')
axes[1].set_title(f"Output — Echo at n={delay_samples} ({delay_ms}ms), feedback={feedback}")
axes[1].set_xlabel("Samples")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlim(0, fs)

for i, pos in enumerate([0, delay_samples, 2 * delay_samples]):
    if pos < fs:
        axes[1].axvline(x=pos, color='red', linestyle='--', alpha=0.6,
                        label='Original' if i == 0 else f'Echo {i}')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# ── Plot 2: Sine Burst ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle("Stage 5 — Sine Burst Response", fontweight='bold')

axes[0].plot(t, sine_burst, color='blue', linewidth=0.8)
axes[0].set_title("Input — 440 Hz Sine Burst (200 ms)")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlabel("Time (seconds)")
axes[0].axvline(x=0.2, color='red', linestyle='--', alpha=0.6, label='Burst end')
axes[0].legend()
axes[0].grid()

axes[1].plot(t, sine_out, color='green', linewidth=0.8)
axes[1].set_title(f"Output — Echo at {delay_ms} ms, feedback={feedback}, mix={mix}")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (seconds)")

for i in range(4):
    echo_time = 0.2 + i * (delay_ms / 1000)
    if echo_time < duration:
        axes[1].axvline(x=echo_time, color='red', linestyle='--',
                        alpha=0.5, label=f'Echo {i + 1}' if i < 3 else '')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# ── Plot 3: Feedback Stability ────────────────────────────────────────────────
feedback_values = [0.3, 0.5, 0.7, 0.9]
colors          = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(13, 5))

for fb, color in zip(feedback_values, colors):
    out = apply_delay(impulse, delay_samples, feedback=fb, mix=1.0)
    plt.plot(t[:fs * 2], out[:fs * 2],
             color=color, alpha=0.8, linewidth=1.5, label=f'feedback={fb}')

plt.title("Feedback Stability — Echo Decay Rate vs Feedback Value")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim(0, 2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()