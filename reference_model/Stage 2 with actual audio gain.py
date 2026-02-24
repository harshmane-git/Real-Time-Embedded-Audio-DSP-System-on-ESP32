# STAGE 2 — GAIN BLOCK (Real Audio File + Playback Comparison)
# Plays ORIGINAL first, then GAINED so you can hear the difference

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd   # pip install sounddevice
import time

# ── Load WAV File ─────────────────────────────────────────────────────────────
input_file = "kalman_filtered_audio.wav"   # <-- change to your file
fs, signal_raw = wavfile.read(input_file)

print(f"Sample Rate : {fs} Hz")
print(f"Duration    : {len(signal_raw)/fs:.2f} seconds")
print(f"Channels    : {'Stereo' if signal_raw.ndim == 2 else 'Mono'}")

# ── Normalize to float [-1.0, 1.0] ───────────────────────────────────────────
if signal_raw.dtype == np.int16:
    signal = signal_raw / 32768.0
elif signal_raw.dtype == np.int32:
    signal = signal_raw / 2147483648.0
else:
    signal = signal_raw.astype(np.float64)

# If stereo, keep both channels for playback but use left for plotting
if signal.ndim == 2:
    signal_plot = signal[:, 0]
else:
    signal_plot = signal

# ── Time Array ────────────────────────────────────────────────────────────────
t = np.linspace(0, len(signal_plot) / fs, len(signal_plot), endpoint=False)

# ── Gain Block ────────────────────────────────────────────────────────────────
def apply_gain(signal, gain):
    return np.clip(gain * signal, -1.0, 1.0)

gain = 3   # <-- adjust this

signal_gained      = apply_gain(signal,      gain)   # for playback (keeps stereo if present)
signal_gained_plot = apply_gain(signal_plot, gain)   # for plotting (mono)

# ── Playback Function ─────────────────────────────────────────────────────────
def play_audio(audio, fs, label):
    duration = len(audio) / fs
    print(f"\n▶  Playing: {label}  ({duration:.1f} seconds)")
    print("   Listen carefully to the volume/loudness...")
    sd.play(audio.astype(np.float32), fs)
    sd.wait()   # block until playback finishes
    print(f"✓  Done: {label}")

# ── Play Original First, Then Gained ─────────────────────────────────────────
print("\n" + "="*50)
print("  AUDIO COMPARISON")
print("="*50)

play_audio(signal, fs, "ORIGINAL (before gain)")

print("\n   Get ready for the gained version...")
time.sleep(1)   # 1 second pause between playbacks

play_audio(signal_gained, fs, f"GAINED  (x{gain})")

print("\n" + "="*50)
print("  PLAYBACK COMPLETE")
print("="*50)

# ── Save Output ───────────────────────────────────────────────────────────────
wavfile.write("gained_output.wav", fs, np.int16(signal_gained_plot * 32767))
print("\nSaved: gained_output.wav")

# ── Plot: Time Domain Comparison ──────────────────────────────────────────────
view_end = min(0.05, len(signal_plot) / fs)

plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.plot(t, signal_plot, color='blue', linewidth=0.7)
plt.title("▶  ORIGINAL Signal (what you heard first)")
plt.ylabel("Amplitude")
plt.xlim(0, view_end)
plt.ylim(-1.1, 1.1)
plt.axhline(y=1.0,  color='red', linestyle='--', alpha=0.4, label='Clip limit ±1.0')
plt.axhline(y=-1.0, color='red', linestyle='--', alpha=0.4)
plt.legend(fontsize=8)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, signal_gained_plot, color='orange', linewidth=0.7)
plt.title(f"▶  GAINED Signal x{gain} (what you heard second)")
plt.ylabel("Amplitude")
plt.xlabel("Time (seconds)")
plt.xlim(0, view_end)
plt.ylim(-1.1, 1.1)
plt.axhline(y=1.0,  color='red', linestyle='--', alpha=0.4, label='Clip limit ±1.0')
plt.axhline(y=-1.0, color='red', linestyle='--', alpha=0.4)
plt.legend(fontsize=8)
plt.grid()

plt.tight_layout()
plt.show()