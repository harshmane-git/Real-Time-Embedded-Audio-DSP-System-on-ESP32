#What is to be expected:-
# Test Vectors test the filters in various ways which are listed below
# We feed some signals into the filters and filters behave in a certain way
#Now when we will implement the same filter in C, we should get same output as the Python reference
#This will ensure the filters are working properly while implementation and bugs can be found

import numpy as np
import os
from datetime import datetime

# Biquad coefficient functions


def lowpass_biquad_coeffs(fc, fs, Q=0.707):
    w0     = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha  = sin_w0 / (2 * Q)

    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2

    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0


def highpass_biquad_coeffs(fc, fs, Q=0.707):
    w0     = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha  = sin_w0 / (2 * Q)

    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2

    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0


def bandpass_biquad_coeffs(fc, fs, Q=1.0):
    w0     = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha  = sin_w0 / (2 * Q)

    b0 = alpha
    b1 = 0
    b2 = -alpha

    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0
# Biquad filter


def biquad_filter(x, coeffs):

    b0,b1,b2,a0,a1,a2 = coeffs

    y = np.zeros_like(x)

    x1 = x2 = 0
    y1 = y2 = 0

    for n in range(len(x)):

        y[n] = (
            b0*x[n]
            + b1*x1
            + b2*x2
            - a1*y1
            - a2*y2
        )

        x2 = x1
        x1 = x[n]

        y2 = y1
        y1 = y[n]

    return y
#Create unique folder- helps it keep organized


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder = f"test_vectors_{timestamp}"
os.makedirs(folder)

print("Saving test vectors in:", folder)

# Filter setup


fs = 16000

lp = lowpass_biquad_coeffs(300, fs)
bp = bandpass_biquad_coeffs(1000, fs)
hp = highpass_biquad_coeffs(2000, fs)

def save_test(name, signal):

    lp_out = biquad_filter(signal, lp)
    bp_out = biquad_filter(signal, bp)
    hp_out = biquad_filter(signal, hp)

    np.savetxt(f"{folder}/input_{name}.csv", signal, delimiter=",")
    np.savetxt(f"{folder}/output_lp_{name}.csv", lp_out, delimiter=",")
    np.savetxt(f"{folder}/output_bp_{name}.csv", bp_out, delimiter=",")
    np.savetxt(f"{folder}/output_hp_{name}.csv", hp_out, delimiter=",")

    print(name, "saved")

#Impulse Test- gives impulse response of the filter

impulse = np.zeros(256)
impulse[0] = 1

save_test("impulse", impulse)

# #  Step Test- A constant signal (all samples high(1)) [How steady it is + divergence/oscillation]

step = np.ones(256)

save_test("step", step)

# Multitone Test- How well can equaliser separate frequencies


t = np.linspace(0,0.02,int(fs*0.02),endpoint=False)

multitone = (
    np.sin(2*np.pi*100*t) +
    np.sin(2*np.pi*500*t) +
    np.sin(2*np.pi*2000*t)
)

save_test("multitone", multitone)
# Single Frequency Test-How it processes a specific frequency

sine = np.sin(2*np.pi*1000*t)

save_test("sine1000", sine)

# Random Signal Test-Stress testing the specific frequency [if its stable with arbitrary signals or not]

np.random.seed(0)
noise = np.random.randn(256)
save_test("random", noise)


print("\nAll 5 test vectors generated successfully.")
