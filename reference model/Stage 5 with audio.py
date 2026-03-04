# Stage 5 — Delay Block with Real Audio File

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import time

# ── Audio Loader ──────────────────────────────────────────────────────────────
def load_audio(filepath):
    if filepath.endswith(".wav"):
        fs, data = wavfile.read(filepath)
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        else:
            data = data.astype(np.float64)
    elif filepath.endswith(".mp3"):
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(filepath)
        fs    = audio.frame_rate
        data  = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        if audio.channels == 2:
            data = data.reshape(-1, 2)
    else:
        raise ValueError("Unsupported format. Use .wav or .mp3")

    if data.ndim == 2:
        print("Stereo detected → converting to mono")
        data = np.mean(data, axis=1)

    data = data / np.max(np.abs(data) + 1e-10)
    return data, fs


# ── Circular Buffer (FIXED) ───────────────────────────────────────────────────
#
#  OLD (buggy):
#    read()  → always reads self.buffer[self.index]  ← same slot about to be overwritten
#    write() → writes then advances index
#    Problem: read and write happen at the SAME position, so you always get
#             the oldest sample in a full buffer — works only because buffer
#             size == delay_samples, masking the real bug.
#
#  NEW (correct):
#    write_index always points to the NEXT slot to write.
#    read_delayed(d) computes:  read_pos = (write_index - d) % size
#    This gives you exactly the sample written d steps ago.
#    Buffer size can now be MAX_DELAY_SAMPLES (one static allocation for all presets).
#
#  C equivalent (for ESP32 port):
#    static float  delay_buf[MAX_DELAY_SAMPLES] = {0};
#    static int    write_idx = 0;
#
#    float read_delayed(int d) {
#        int read_pos = (write_idx - d + MAX_DELAY_SAMPLES) % MAX_DELAY_SAMPLES;
#        return delay_buf[read_pos];
#    }
#    void write_sample(float val) {
#        delay_buf[write_idx] = val;
#        write_idx = (write_idx + 1) % MAX_DELAY_SAMPLES;
#    }

class CircularBuffer:
    def __init__(self, size):
        # size should be >= any delay_samples you will request
        # On ESP32: allocate once as static array of MAX_DELAY_SAMPLES
        self.buffer     = np.zeros(size)
        self.size       = size
        self.write_index = 0          # always points to NEXT slot to write

    def read_delayed(self, delay_samples):
        """Return the sample written exactly delay_samples steps ago."""
        assert delay_samples <= self.size, \
            f"delay_samples ({delay_samples}) exceeds buffer size ({self.size})"
        read_pos = (self.write_index - delay_samples) % self.size
        return self.buffer[read_pos]

    def write(self, value):
        """Write value at current write position, then advance."""
        self.buffer[self.write_index] = value
        self.write_index = (self.write_index + 1) % self.size

    def clear(self):
        """Flush buffer — call this on preset switch (at block boundary)."""
        self.buffer[:]   = 0.0
        self.write_index = 0


# ── Delay Block ───────────────────────────────────────────────────────────────
def apply_delay(signal, delay_samples, feedback=0.5, mix=0.5):
    """
    Difference equation:
        y[n] = x[n]  +  mix * x[n - D]_fed
        buf_write = x[n] + feedback * x[n - D]_fed

    Buffer is sized to delay_samples here for a single-preset run.
    For multi-preset (see Stage5 updated version), size to MAX_DELAY_SAMPLES.
    """
    N      = len(signal)
    output = np.zeros(N)
    buf    = CircularBuffer(delay_samples)   # size == delay_samples is valid & intentional

    for n in range(N):
        x_n     = signal[n]
        delayed = buf.read_delayed(delay_samples)   # <-- FIXED: explicit delay tap
        y_n     = x_n + mix * delayed
        buf.write(x_n + feedback * delayed)
        output[n] = y_n

    output = output / np.max(np.abs(output) + 1e-10)
    return output


# ── Load File ─────────────────────────────────────────────────────────────────
filepath   = "WhatsApp Audio 2026-02-26 at 22.03.34.wav"    # <-- change to your file
signal, fs = load_audio(filepath)
t = np.linspace(0, len(signal) / fs, len(signal), endpoint=False)

print("Stage 5 — Delay Block (Real Audio)")
print("=" * 45)
print(f"  File          : {filepath}")
print(f"  Sample rate   : {fs} Hz")
print(f"  Duration      : {len(signal)/fs:.2f} seconds")

# ── Delay Configuration ───────────────────────────────────────────────────────
delay_ms      = 500
feedback      = 0.5
mix           = 0.5
delay_samples = int((delay_ms / 1000) * fs)

print(f"  Delay         : {delay_ms} ms = {delay_samples} samples")
print(f"  Feedback      : {feedback}")
print(f"  Mix           : {mix}")

# ── Apply Delay ───────────────────────────────────────────────────────────────
output = apply_delay(signal, delay_samples, feedback, mix)

# ── Save Output ───────────────────────────────────────────────────────────────
out_filepath = "delay_output.wav"
wavfile.write(out_filepath, fs, np.int16(output * 32767))
print(f"\n  Saved         : {out_filepath}")

# ── Playback ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 45)
print("  PLAYBACK")
print("=" * 45)

print("\n▶  Playing: ORIGINAL")
sd.play(signal.astype(np.float32), fs)
sd.wait()
time.sleep(0.5)

print(f"▶  Playing: DELAY OUTPUT ({delay_ms}ms echo, feedback={feedback})")
sd.play(output.astype(np.float32), fs)
sd.wait()
print("\n✔  Playback complete")

# ── Plot 1: Waveform — Input vs Output ───────────────────────────────────────
view_end = min(3.0, len(signal) / fs)

fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle("Stage 5 — Delay Block: Input vs Output Waveform", fontweight='bold')

axes[0].plot(t, signal, color='blue', linewidth=0.6)
axes[0].set_title(f"Input — {filepath}")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, view_end)
axes[0].set_ylim(-1.1, 1.1)
axes[0].grid()

axes[1].plot(t, output, color='green', linewidth=0.6)
axes[1].set_title(f"Output — Delay {delay_ms}ms | Feedback {feedback} | Mix {mix}")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_xlim(0, view_end)
axes[1].set_ylim(-1.1, 1.1)
axes[1].grid()

plt.tight_layout()
plt.show()

# ── Plot 2: FFT — Input vs Output ─────────────────────────────────────────────
freqs      = np.fft.rfftfreq(len(signal), d=1/fs)
fft_input  = np.abs(np.fft.rfft(signal))
fft_output = np.abs(np.fft.rfft(output))

plt.figure(figsize=(13, 5))
plt.plot(freqs, 20*np.log10(fft_input  + 1e-10),
         color='blue',  alpha=0.8, linewidth=1.2, label='Input')
plt.plot(freqs, 20*np.log10(fft_output + 1e-10),
         color='green', alpha=0.8, linewidth=1.2, label='Delay Output')
plt.title("Stage 5 — Frequency Spectrum: Input vs Delay Output")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()