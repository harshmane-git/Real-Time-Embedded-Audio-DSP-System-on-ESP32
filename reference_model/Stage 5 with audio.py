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

# ── Delay Block ───────────────────────────────────────────────────────────────
def apply_delay(signal, delay_samples, feedback=0.5, mix=0.5):
    N      = len(signal)
    output = np.zeros(N)
    buf    = CircularBuffer(delay_samples)

    for n in range(N):
        x_n       = signal[n]
        delayed   = buf.read()
        y_n       = x_n + mix * delayed
        buf.write(x_n + feedback * delayed)
        output[n] = y_n

    output = output / np.max(np.abs(output) + 1e-10)
    return output

# ── Load File ─────────────────────────────────────────────────────────────────
filepath = "filtered_audio.wav"    # <-- change to your file
signal, fs = load_audio(filepath)
t = np.linspace(0, len(signal) / fs, len(signal), endpoint=False)

print("Stage 5 — Delay Block")
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