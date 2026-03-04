# Stage 5 — Delay Block (Echo Effect)
# Updated: Added delay presets, safe block-boundary switching, and static allocation pattern

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
fs         = 16000
BLOCK_SIZE = 256   # Must match ESP32 block size

# ── Delay Presets (Switch 2 cycles through these) ─────────────────────────────
# Each preset: (name, delay_ms, feedback, mix)
DELAY_PRESETS = [
    ("Short Echo",   150, 0.3, 0.5),   # Preset 0 — subtle short slap echo
    ("Medium Echo",  500, 0.5, 0.5),   # Preset 1 — standard room echo
    ("Long Echo",   800, 0.7, 0.6),   # Preset 2 — long hall echo
]

def get_preset(index):
    name, delay_ms, feedback, mix = DELAY_PRESETS[index]
    delay_samples = int((delay_ms / 1000) * fs)
    return name, delay_samples, feedback, mix

# ── Circular Buffer ───────────────────────────────────────────────────────────
# NOTE for C port: allocate buffer as a STATIC global array, never with malloc()
# Example C:  static float delay_buf[MAX_DELAY_SAMPLES] = {0};
MAX_DELAY_SAMPLES = int((max(p[1] for p in DELAY_PRESETS) / 1000) * fs)

class CircularBuffer:
    """
    Fixed-size circular buffer.
    C port: use a static float array + static int write_index.
    Buffer is allocated once at startup — never during processing.
    """
    def __init__(self, size):
        # Static allocation: size is fixed at construction (matches C static array)
        self.buffer = np.zeros(size)
        self.size   = size
        self.index  = 0

    def read_delayed(self, delay_samples):
        read_pos = (self.index - delay_samples) % self.size
        return self.buffer[read_pos]

    def write(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size

    def clear(self):
        """Called on preset switch to flush old echoes (at block boundary)."""
        self.buffer[:] = 0.0
        self.index = 0

# ── Delay Block (block-based, safe preset switching) ─────────────────────────
def apply_delay_block(signal, preset_index=1, switch_at_sample=None, next_preset_index=None):
    """
    Process signal in BLOCK_SIZE chunks — mirrors ESP32 DMA block structure.

    Preset switching happens ONLY at block boundaries (never mid-block),
    matching the requirement: 'Presets must change safely at block boundaries'.

    Parameters
    ----------
    signal           : input audio (mono float array)
    preset_index     : initial preset (0, 1, or 2)
    switch_at_sample : sample index at which to queue a preset switch
    next_preset_index: preset to switch to
    """
    N      = len(signal)
    output = np.zeros(N)
    buf    = CircularBuffer(MAX_DELAY_SAMPLES)

    _, delay_samples, feedback, mix = get_preset(preset_index)
    pending_switch = False

    for block_start in range(0, N, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, N)

        # ── Safe preset switch at block boundary ──────────────────────────
        if pending_switch:
            preset_index = next_preset_index
            _, delay_samples, feedback, mix = get_preset(preset_index)
            buf.clear()          # flush buffer to avoid artefacts from old echoes
            pending_switch = False

        # Queue switch if the trigger falls inside this block
        if (switch_at_sample is not None and
                next_preset_index is not None and
                block_start <= switch_at_sample < block_end):
            pending_switch = True   # will apply at START of next block

        # ── Process block sample-by-sample (difference equation) ──────────
        for n in range(block_start, block_end):
            x_n     = signal[n]
            delayed = buf.read_delayed(delay_samples)
            y_n     = x_n + mix * delayed
            buf.write(x_n + feedback * delayed)
            output[n] = y_n

    return output

# ── Test Signals ──────────────────────────────────────────────────────────────
duration = 3
t        = np.linspace(0, duration, fs * duration, endpoint=False)

# Impulse
impulse     = np.zeros(fs * duration)
impulse[0]  = 1.0

# Sine burst (440 Hz, 200 ms)
sine_burst  = np.zeros(fs * duration)
n_burst     = int(0.2 * fs)
sine_burst[:n_burst] = np.sin(2 * np.pi * 440 * t[:n_burst])

# ── Run all three presets on impulse ─────────────────────────────────────────
preset_outputs = []
for i in range(len(DELAY_PRESETS)):
    out = apply_delay_block(impulse, preset_index=i)
    preset_outputs.append(out)

# ── Preset-switch test: Preset 0 → Preset 2 at t=1.0 s ───────────────────────
switch_sample = int(1.0 * fs)
switched_out  = apply_delay_block(
    sine_burst,
    preset_index=0,
    switch_at_sample=switch_sample,
    next_preset_index=2
)

# ── Verification ──────────────────────────────────────────────────────────────
print("Stage 5 — Delay Block (Updated)")
print("=" * 55)
print(f"  Sample rate       : {fs} Hz")
print(f"  Block size        : {BLOCK_SIZE} samples")
print(f"  Max delay buffer  : {MAX_DELAY_SAMPLES} samples (static)")
print()
for i, (name, delay_ms, feedback, mix) in enumerate(DELAY_PRESETS):
    delay_samples = int((delay_ms / 1000) * fs)
    echo_pos = np.where(np.abs(preset_outputs[i]) > 0.001)[0]
    first_echo = echo_pos[1] if len(echo_pos) > 1 else -1
    status = "✔" if first_echo == delay_samples else "✗"
    print(f"  Preset {i} [{name:12s}]: delay={delay_ms}ms  feedback={feedback}  "
          f"echo@{first_echo} (expected {delay_samples}) {status}")
print()
print(f"  Preset switch test: Preset 0 → Preset 2 at sample {switch_sample}")
print(f"  Switch occurs at next block boundary: "
      f"sample {((switch_sample // BLOCK_SIZE) + 1) * BLOCK_SIZE}")

# ── Plot 1: All Presets — Impulse Response ────────────────────────────────────
colors = ['blue', 'green', 'red']
fig, axes = plt.subplots(len(DELAY_PRESETS), 1, figsize=(13, 9), sharex=True)
fig.suptitle("Stage 5 — Impulse Response for Each Preset", fontweight='bold')

view_samples = fs * 2   # show 2 seconds
for i, (ax, out) in enumerate(zip(axes, preset_outputs)):
    name, delay_ms, feedback, mix = DELAY_PRESETS[i]
    delay_samples = int((delay_ms / 1000) * fs)
    ax.plot(np.arange(view_samples) / fs, out[:view_samples],
            color=colors[i], linewidth=0.8)
    for echo_n in range(4):
        pos = delay_samples * (echo_n + 1)
        if pos < view_samples:
            ax.axvline(x=pos / fs, color='red', linestyle='--',
                       alpha=0.5, label=f'Echo {echo_n+1}' if echo_n == 0 else '')
    ax.set_title(f"Preset {i}: {name}  |  delay={delay_ms}ms, feedback={feedback}, mix={mix}")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right')
    ax.grid()

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()

# ── Plot 2: Safe Preset Switch ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle("Stage 5 — Safe Preset Switch at Block Boundary (Preset 0 → Preset 2)",
             fontweight='bold')

axes[0].plot(t, sine_burst, color='blue', linewidth=0.8)
axes[0].set_title("Input — 440 Hz Sine Burst (200 ms)")
axes[0].set_ylabel("Amplitude")
axes[0].axvline(x=0.2, color='gray', linestyle='--', alpha=0.6, label='Burst end')
axes[0].legend()
axes[0].grid()

axes[1].plot(t, switched_out, color='green', linewidth=0.8)
switch_boundary = ((switch_sample // BLOCK_SIZE) + 1) * BLOCK_SIZE / fs
axes[1].axvline(x=switch_sample / fs, color='orange', linestyle=':',
                alpha=0.8, label=f'Switch requested (t={switch_sample/fs:.2f}s)')
axes[1].axvline(x=switch_boundary, color='red', linestyle='--',
                alpha=0.8, label=f'Switch applied @ block boundary (t={switch_boundary:.3f}s)')
axes[1].set_title("Output — Echo character changes at block boundary (no mid-block glitch)")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (seconds)")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# ── Plot 3: Feedback Stability across all presets ─────────────────────────────
plt.figure(figsize=(13, 5))
for i, (name, delay_ms, fb, mx) in enumerate(DELAY_PRESETS):
    plt.plot(t[:fs * 2], preset_outputs[i][:fs * 2],
             color=colors[i], alpha=0.8, linewidth=1.2,
             label=f'Preset {i}: {name} (fb={fb})')
plt.title("Feedback Stability — All Presets")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim(0, 2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()