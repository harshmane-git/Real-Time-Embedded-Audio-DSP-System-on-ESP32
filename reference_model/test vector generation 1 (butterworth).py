import numpy as np
import scipy.signal as sp
import os
# Sampling parameters

fs = 16000

# filters will go here--
def filters_constant(fs):

    low_b, low_a = sp.butter(4, 300, btype='low', fs=fs)
    mid_b, mid_a = sp.butter(4, [300,2000], btype='bandpass', fs=fs)
    high_b, high_a = sp.butter(4, 2000, btype='high', fs=fs)

    return low_b, low_a, mid_b, mid_a, high_b, high_a

# Apply filters and reconstruct signal

def filter_diff_eqn(x, filters, gains):

    low_b, low_a, mid_b, mid_a, high_b, high_a = filters
    G_low, G_mid, G_high = gains

    low  = sp.lfilter(low_b, low_a, x)
    mid  = sp.lfilter(mid_b, mid_a, x)
    high = sp.lfilter(high_b, high_a, x)

    reconstructed = (
        G_low*low +
        G_mid*mid +
        G_high*high
    )

    return reconstructed

# Save test vectors

def save_test_vector(name, input_signal):

    output_signal = filter_diff_eqn(input_signal, filters, gains)

    np.savetxt(
        f"test_vectors/input_{name}.txt",
        input_signal,
        fmt="%.10f"
    )

    np.savetxt(
        f"test_vectors/output_{name}.txt",
        output_signal,
        fmt="%.10f"
    )

    print(f"{name} test vector generated")


# Create folder

os.makedirs("test_vectors", exist_ok=True)



# Initialize filters and gains

filters = filters_constant(fs)

gains = (1.0, 1.0, 1.0)

#  Impulse Test- gives impulse response of the filter

impulse = np.zeros(256)
impulse[0] = 1

save_test_vector("impulse", impulse)

#  Step Test- A constant signal (all samples high(1)) [How steady it is + divergence/oscillation]

step = np.ones(256)
save_test_vector("step", step)

# 3. Multitone Test- How well can equaliser separate frequencies

t = np.linspace(0,0.02,int(fs*0.02),endpoint=False)

multitone = (
    np.sin(2*np.pi*100*t) +
    np.sin(2*np.pi*500*t) +
    np.sin(2*np.pi*2000*t)
)

save_test_vector("multitone", multitone)

# Single Frequency Test-How it processes a specific frequency

sine1000 = np.sin(2*np.pi*1000*t)

save_test_vector("sine1000", sine1000)

# Random Signal Test-Stress testing the specific frequency [if its stable with arbitrary signals or not]

np.random.seed(0)

random_signal = np.random.randn(256)

save_test_vector("random", random_signal)


print("\nAll test vectors generated successfully.")
