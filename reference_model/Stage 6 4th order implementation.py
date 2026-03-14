# Stage 6 — Full DSP Pipeline with Limiter
# ESP32 Audio DSP Reference Model
# Pipeline: Mic (I2S) → Gain → 3-Band IIR EQ (4th Order) → Delay → Limiter → Speaker (I2S)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, freqz
from math import gcd

# ── Audio Loader ──────────────────────────────────────────────────────────────
def load_audio(filepath, target_fs=16000):
    if not filepath.endswith(".wav"):
        raise ValueError("Only .wav files are supported.")
    fs, data = wavfile.read(filepath)
    if data.dtype == np.int16:   data = data / 32768.0
    elif data.dtype == np.int32: data = data / 2147483648.0
    else:                        data = data.astype(np.float64)
    if data.ndim == 2:
        print("Stereo → mono")
        data = np.mean(data, axis=1)
    if fs != target_fs:
        g = gcd(target_fs, fs)
        data = resample_poly(data, target_fs // g, fs // g)
        print(f"Resampled: {fs} Hz → {target_fs} Hz")
        fs = target_fs
    return data / np.max(np.abs(data) + 1e-10), fs

# ── 2nd Order Biquad Coefficient Functions ────────────────────────────────────
def lowpass_biquad_coeffs(fc, fs, Q=0.707):
    w0=2*np.pi*fc/fs; c=np.cos(w0); s=np.sin(w0); a=s/(2*Q)
    b0=(1-c)/2; b1=1-c; b2=(1-c)/2; a0=1+a; a1=-2*c; a2=1-a
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

def highpass_biquad_coeffs(fc, fs, Q=0.707):
    w0=2*np.pi*fc/fs; c=np.cos(w0); s=np.sin(w0); a=s/(2*Q)
    b0=(1+c)/2; b1=-(1+c); b2=(1+c)/2; a0=1+a; a1=-2*c; a2=1-a
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

def bandpass_biquad_coeffs(fc, fs, Q=1.0):
    w0=2*np.pi*fc/fs; c=np.cos(w0); s=np.sin(w0); a=s/(2*Q)
    b0=a; b1=0; b2=-a; a0=1+a; a1=-2*c; a2=1-a
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0

# ── 4th Order Coefficient Generators ─────────────────────────────────────────
# Butterworth 4th order Q values:
# Section 1: Q = 0.5412  (Butterworth pole at pi/8)
# Section 2: Q = 1.3066  (Butterworth pole at 3*pi/8)
Q1 = 0.5412
Q2 = 1.3066

def lowpass_4th_order(fc, fs):
    return lowpass_biquad_coeffs(fc, fs, Q=Q1), lowpass_biquad_coeffs(fc, fs, Q=Q2)

def highpass_4th_order(fc, fs):
    return highpass_biquad_coeffs(fc, fs, Q=Q1), highpass_biquad_coeffs(fc, fs, Q=Q2)

def bandpass_4th_order(fc, fs):
    return bandpass_biquad_coeffs(fc, fs, Q=Q1*2), bandpass_biquad_coeffs(fc, fs, Q=Q2*2)

# ── apply_biquad() — used only for Plot 2 (EQ band breakdown) ─────────────────
def apply_biquad(signal, b0, b1, b2, a0, a1, a2):
    N=len(signal); out=np.zeros(N); x1=x2=y1=y2=0.0
    for n in range(N):
        x=signal[n]; y=b0*x+b1*x1+b2*x2-a1*y1-a2*y2
        out[n]=y; x2,x1=x1,x; y2,y1=y1,y
    return out

# ── Circular Buffer ───────────────────────────────────────────────────────────
class CircularBuffer:
    def __init__(self, size):
        self.buf=np.zeros(size); self.size=size; self.wi=0
    def read_delayed(self, d):
        return self.buf[(self.wi - d) % self.size]
    def write(self, v):
        self.buf[self.wi]=v; self.wi=(self.wi+1) % self.size
    def clear(self):
        self.buf[:]=0.0; self.wi=0

# ── Presets ───────────────────────────────────────────────────────────────────
EQ_PRESETS = {
    "Flat"        : {"g_low": 1.0, "g_mid": 1.0, "g_high": 1.0},
    "Bass Boost"  : {"g_low": 1.5, "g_mid": 1.0, "g_high": 0.8},
    "Treble Boost": {"g_low": 0.8, "g_mid": 1.0, "g_high": 1.5},
    "Voice Boost" : {"g_low": 0.7, "g_mid": 1.5, "g_high": 0.9},
    "Loudness"    : {"g_low": 1.5, "g_mid": 0.8, "g_high": 1.5},
}
DELAY_PRESETS = {
    "Short" : {"delay_ms": 125, "feedback": 0.3, "mix": 0.4},
    "Medium": {"delay_ms": 250, "feedback": 0.4, "mix": 0.5},
    "Long"  : {"delay_ms": 500, "feedback": 0.5, "mix": 0.5},
    "Speech": {"delay_ms": 200, "feedback": 0.2, "mix": 0.2},
}
MAX_DELAY_SAMPLES = int(max(p["delay_ms"] for p in DELAY_PRESETS.values()) / 1000 * 16000)

# ── Block-Based Pipeline — 4th Order EQ ──────────────────────────────────────
# Each band uses two cascaded Biquad sections (Section 1 → Section 2)
# State variables: 8 per band (x1,x2,y1,y2 for each section)
# Roll-off: -80 dB/decade per band
def run_pipeline(signal, fs, BLOCK_SIZE, gain,
                 lp_s1, lp_s2,
                 bp_s1, bp_s2,
                 hp_s1, hp_s2,
                 eq_gains,
                 delay_samples, feedback, mix, threshold):

    N = len(signal)
    output    = np.zeros(N)
    out_gain  = np.zeros(N)
    out_eq    = np.zeros(N)
    out_delay = np.zeros(N)

    # LP section 1 state
    lp1_x1=lp1_x2=lp1_y1=lp1_y2=0.0
    # LP section 2 state
    lp2_x1=lp2_x2=lp2_y1=lp2_y2=0.0
    # BP section 1 state
    bp1_x1=bp1_x2=bp1_y1=bp1_y2=0.0
    # BP section 2 state
    bp2_x1=bp2_x2=bp2_y1=bp2_y2=0.0
    # HP section 1 state
    hp1_x1=hp1_x2=hp1_y1=hp1_y2=0.0
    # HP section 2 state
    hp2_x1=hp2_x2=hp2_y1=hp2_y2=0.0

    buf = CircularBuffer(MAX_DELAY_SAMPLES)
    g_low, g_mid, g_high = eq_gains["g_low"], eq_gains["g_mid"], eq_gains["g_high"]
    d_samp, d_fb, d_mix  = delay_samples, feedback, mix

    for bs in range(0, N, BLOCK_SIZE):
        be = min(bs + BLOCK_SIZE, N)

        for n in range(bs, be):
            x = signal[n]

            # ── Gain ──────────────────────────────────────────────────────────
            x_g = gain * x
            out_gain[n] = x_g

            # ── EQ Bass — LP 4th Order (Section 1 → Section 2) ───────────────
            b0,b1,b2,_,a1,a2 = lp_s1
            lp1_y = b0*x_g  + b1*lp1_x1 + b2*lp1_x2 - a1*lp1_y1 - a2*lp1_y2
            lp1_x2,lp1_x1   = lp1_x1,x_g
            lp1_y2,lp1_y1   = lp1_y1,lp1_y

            b0,b1,b2,_,a1,a2 = lp_s2
            lp2_y = b0*lp1_y + b1*lp2_x1 + b2*lp2_x2 - a1*lp2_y1 - a2*lp2_y2
            lp2_x2,lp2_x1   = lp2_x1,lp1_y
            lp2_y2,lp2_y1   = lp2_y1,lp2_y

            # ── EQ Mid — BP 4th Order (Section 1 → Section 2) ────────────────
            b0,b1,b2,_,a1,a2 = bp_s1
            bp1_y = b0*x_g  + b1*bp1_x1 + b2*bp1_x2 - a1*bp1_y1 - a2*bp1_y2
            bp1_x2,bp1_x1   = bp1_x1,x_g
            bp1_y2,bp1_y1   = bp1_y1,bp1_y

            b0,b1,b2,_,a1,a2 = bp_s2
            bp2_y = b0*bp1_y + b1*bp2_x1 + b2*bp2_x2 - a1*bp2_y1 - a2*bp2_y2
            bp2_x2,bp2_x1   = bp2_x1,bp1_y
            bp2_y2,bp2_y1   = bp2_y1,bp2_y

            # ── EQ Treble — HP 4th Order (Section 1 → Section 2) ─────────────
            b0,b1,b2,_,a1,a2 = hp_s1
            hp1_y = b0*x_g  + b1*hp1_x1 + b2*hp1_x2 - a1*hp1_y1 - a2*hp1_y2
            hp1_x2,hp1_x1   = hp1_x1,x_g
            hp1_y2,hp1_y1   = hp1_y1,hp1_y

            b0,b1,b2,_,a1,a2 = hp_s2
            hp2_y = b0*hp1_y + b1*hp2_x1 + b2*hp2_x2 - a1*hp2_y1 - a2*hp2_y2
            hp2_x2,hp2_x1   = hp2_x1,hp1_y
            hp2_y2,hp2_y1   = hp2_y1,hp2_y

            # ── EQ Mix ────────────────────────────────────────────────────────
            x_eq = g_low*lp2_y + g_mid*bp2_y + g_high*hp2_y
            out_eq[n] = x_eq

            # ── Delay ─────────────────────────────────────────────────────────
            delayed = buf.read_delayed(d_samp)
            x_dly   = x_eq + d_mix * delayed
            buf.write(x_eq + d_fb * delayed)
            out_delay[n] = x_dly

            # ── Limiter ───────────────────────────────────────────────────────
            output[n] = np.clip(x_dly, -threshold, threshold)

    return output, out_gain, out_eq, out_delay

# ── Configuration ─────────────────────────────────────────────────────────────
filepath      = "WhatsApp Audio 2026-02-26 at 22.03.34.wav"
signal, fs    = load_audio(filepath, target_fs=16000)
t             = np.linspace(0, len(signal)/fs, len(signal), endpoint=False)
BLOCK_SIZE    = 256
gain          = 2.0
eq_preset     = EQ_PRESETS["Voice Boost"]
delay_preset  = DELAY_PRESETS["Speech"]
threshold     = 0.9
delay_ms      = delay_preset["delay_ms"]
feedback      = delay_preset["feedback"]
mix           = delay_preset["mix"]
delay_samples = int((delay_ms/1000)*fs)
fc_low, fc_mid, fc_high = 300, 800, 2000

lp_s1, lp_s2 = lowpass_4th_order(fc_low,  fs)
bp_s1, bp_s2 = bandpass_4th_order(fc_mid,  fs)
hp_s1, hp_s2 = highpass_4th_order(fc_high, fs)

# ── Run Pipeline ──────────────────────────────────────────────────────────────
stage_limiter, stage_gain, stage_eq, stage_delay = run_pipeline(
    signal, fs, BLOCK_SIZE, gain,
    lp_s1, lp_s2,
    bp_s1, bp_s2,
    hp_s1, hp_s2,
    eq_preset,
    delay_samples, feedback, mix, threshold
)
clipped = np.sum(np.abs(stage_delay) > threshold)

print("ESP32 Audio DSP — Stage 6 (4th Order EQ)")
print("=" * 55)
print(f"  File          : {filepath}")
print(f"  Sample rate   : {fs} Hz  |  Block size: {BLOCK_SIZE}")
print(f"  EQ Preset     : Voice Boost  ({fc_low}/{fc_mid}/{fc_high} Hz)")
print(f"  EQ Order      : 4th order (2 cascaded Biquad sections per band)")
print(f"  Delay Preset  : Speech ({delay_ms}ms, fb={feedback}, mix={mix})")
print("─" * 55)
print(f"  Input  peak   : {np.max(np.abs(signal)):.4f}")
print(f"  After Gain    : {np.max(np.abs(stage_gain)):.4f}")
print(f"  After EQ      : {np.max(np.abs(stage_eq)):.4f}")
print(f"  After Delay   : {np.max(np.abs(stage_delay)):.4f}")
print(f"  After Limiter : {np.max(np.abs(stage_limiter)):.4f}  ✔")
print(f"  Clipped       : {clipped} samples ({clipped/len(signal)*100:.2f}%)")

wavfile.write("pipeline_output_fourth_order.wav", fs, np.int16(stage_limiter*32767))
print(f"\n  Saved: pipeline_output_fourth_order.wav")

# ── Plot 1: All Pipeline Stages ───────────────────────────────────────────────
view_end = min(3.0, len(signal)/fs)
stages = [
    (signal,        "Input",                                            'black'),
    (stage_gain,    f"Gain ×{gain}",                                   'blue'),
    (stage_eq,      f"EQ Voice Boost ({fc_low}/{fc_mid}/{fc_high}Hz) — 4th Order", 'orange'),
    (stage_delay,   f"Delay {delay_ms}ms fb={feedback}",               'purple'),
    (stage_limiter, f"Limiter ±{threshold}",                           'green'),
]
fig, axes = plt.subplots(len(stages), 1, figsize=(13, 3*len(stages)))
fig.suptitle("ESP32 DSP — Full Pipeline (4th Order EQ)  |  fs=16000 Hz", fontweight='bold')
for i, (data, label, color) in enumerate(stages):
    axes[i].plot(t, data, color=color, linewidth=0.6)
    axes[i].axhline( threshold, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[i].axhline(-threshold, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[i].set_title(f"{label}  |  peak={np.max(np.abs(data)):.4f}")
    axes[i].set_xlim(0, view_end); axes[i].set_ylim(-2.2, 2.2)
    axes[i].set_ylabel("Amplitude"); axes[i].grid()
axes[-1].set_xlabel("Time (s)")
plt.tight_layout(); plt.show()

# ── Plot 2: EQ Band Breakdown ─────────────────────────────────────────────────
g_low, g_mid, g_high = eq_preset["g_low"], eq_preset["g_mid"], eq_preset["g_high"]

# Run each band through both sections for breakdown plot
low_band  = apply_biquad(apply_biquad(stage_gain, *lp_s1), *lp_s2)
mid_band  = apply_biquad(apply_biquad(stage_gain, *bp_s1), *bp_s2)
high_band = apply_biquad(apply_biquad(stage_gain, *hp_s1), *hp_s2)

fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
fig.suptitle(f"EQ Band Breakdown — Voice Boost | {fc_low}/{fc_mid}/{fc_high} Hz | 4th Order",
             fontweight='bold')
for ax, (data, label, color) in zip(axes, [
    (low_band *g_low,  f"Bass ×{g_low}  (LP {fc_low}Hz  4th order)",    'blue'),
    (mid_band *g_mid,  f"Mid  ×{g_mid}  (BP {fc_mid}Hz  4th order)",    'orange'),
    (high_band*g_high, f"Treble ×{g_high} (HP {fc_high}Hz 4th order)",  'green'),
    (stage_eq,         "Mixed EQ Output",                                'red'),
]):
    ax.plot(t, data, color=color, linewidth=0.6)
    ax.set_title(f"{label}  |  peak={np.max(np.abs(data)):.4f}")
    ax.set_xlim(0, view_end); ax.set_ylabel("Amplitude"); ax.grid()
axes[-1].set_xlabel("Time (s)")
plt.tight_layout(); plt.show()

# ── Plot 3: Limiter Before vs After ───────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle("Limiter: Before vs After", fontweight='bold')
for ax, (data, label, color) in zip(axes, [
    (stage_delay,   f"Before  peak={np.max(np.abs(stage_delay)):.4f}", 'purple'),
    (stage_limiter, f"After   peak={np.max(np.abs(stage_limiter)):.4f}  clipped={clipped}", 'green'),
]):
    ax.plot(t, data, color=color, linewidth=0.6)
    ax.axhline( threshold, color='red', linestyle='--', linewidth=1.2, label=f'±{threshold}')
    ax.axhline(-threshold, color='red', linestyle='--', linewidth=1.2)
    ax.set_title(label); ax.set_xlim(0, view_end)
    ax.set_ylabel("Amplitude"); ax.legend(); ax.grid()
axes[-1].set_xlabel("Time (s)")
plt.tight_layout(); plt.show()

# ── Plot 4: FFT — Input vs Output ─────────────────────────────────────────────
freqs = np.fft.rfftfreq(len(signal), d=1/fs)
plt.figure(figsize=(13, 5))
plt.plot(freqs, 20*np.log10(np.abs(np.fft.rfft(signal))        +1e-10), color='black', linewidth=1.2, label='Input')
plt.plot(freqs, 20*np.log10(np.abs(np.fft.rfft(stage_limiter)) +1e-10), color='green', linewidth=1.2, label='Output')
for fc, color, ls in [(fc_low,'blue',':'),(fc_mid,'orange',':'),(fc_high,'green','--')]:
    plt.axvline(x=fc, color=color, linestyle=ls, alpha=0.6, label=f'{fc}Hz')
plt.title("FFT: Input vs Pipeline Output  (4th Order EQ)")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2); plt.ylim(-80, 20); plt.legend(); plt.grid()
plt.tight_layout(); plt.show()

# ── Plot 5: Biquad Filter Responses — 4th Order ───────────────────────────────
plt.figure(figsize=(13, 5))
for (s1, s2), label, color in [
    ((lp_s1, lp_s2), f"LP {fc_low}Hz  4th order",  'blue'),
    ((bp_s1, bp_s2), f"BP {fc_mid}Hz  4th order",  'orange'),
    ((hp_s1, hp_s2), f"HP {fc_high}Hz 4th order",  'green'),
]:
    w, h1 = freqz(list(s1[:3]), [s1[3], s1[4], s1[5]], worN=8000, fs=fs)
    w, h2 = freqz(list(s2[:3]), [s2[3], s2[4], s2[5]], worN=8000, fs=fs)
    plt.plot(w, 20*np.log10(np.abs(h1*h2)+1e-10), color=color, linewidth=1.5, label=label)
plt.title(f"4th Order Biquad Filter Responses  |  fs={fs} Hz  |  -80 dB/decade")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2); plt.ylim(-80, 5); plt.legend(); plt.grid()
plt.tight_layout(); plt.show()

# ── Performance Analysis (Section 7) ──────────────────────────────────────────
def measure_latency(x, y, fs):
    win=min(4096, len(x))
    corr=np.correlate(y[:win], x[:win], mode='full')
    lag=np.argmax(np.abs(corr))-(win-1)
    return lag, lag/fs*1000

# 4th order: 10 mults + 8 adds per band (two sections of 5+4)
# 3 bands + gain + limiter
mults_ps        = 10*3 + 2
adds_ps         = 8*3  + 2
total_ops_block = (mults_ps + adds_ps) * BLOCK_SIZE
block_ms        = BLOCK_SIZE / fs * 1000
esp32_cycles    = block_ms / 1000 * 240e6
cycles_needed   = total_ops_block * 4
headroom        = (1 - cycles_needed/esp32_cycles) * 100
total_mem       = delay_samples*4 + 3*2*4*4 + BLOCK_SIZE*4
lat_samp, lat_ms = measure_latency(signal, stage_limiter, fs)

print("\nPerformance Analysis")
print("=" * 55)
print(f"  Theoretical latency  : {block_ms:.2f} ms")
print(f"  Measured latency     : {lat_ms:.2f} ms  ({lat_samp} samples)")
print(f"  EQ order             : 4th (2 sections × 3 bands)")
print(f"  Mults/sample         : {mults_ps}  |  Adds/sample: {adds_ps}")
print(f"  Total ops/block      : {total_ops_block}")
print(f"  Cycles needed        : {cycles_needed:,.0f}  |  Available: {esp32_cycles:,.0f}")
print(f"  CPU headroom         : {headroom:.1f}%  {'✔' if headroom>0 else '✗'}")
print(f"  Total DSP memory     : {total_mem} bytes  ({total_mem/1024:.2f} KB)")