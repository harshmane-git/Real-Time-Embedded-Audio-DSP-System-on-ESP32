#ifndef CONFIG_H
#define CONFIG_H

// =====================================================
//               DSP CONFIGURATION FILE
// =====================================================
// This file contains ONLY user-tunable parameters.
// DSP internals such as buffers, coefficients,
// pointers, and filter states must NOT be added here.
// =====================================================


// ================= ENABLE FLAGS =================

// Enable / Disable Equalizer
#define ENABLE_EQ              1

// Enable / Disable Delay
#define ENABLE_DELAY           1

// Enable / Disable Limiter
#define ENABLE_LIMITER         1


// ================= MASTER GAIN =================

// Overall output gain in dB
#define GLOBAL_GAIN_DB         3.0f


// ================= EQUALIZER SETTINGS =================

// Bass Gain (Low Frequencies)
#define LOW_GAIN_DB            2.0f

// Midrange Gain
#define MID_GAIN_DB           -1.0f

// Treble Gain (High Frequencies)
#define HIGH_GAIN_DB           1.5f


// ================= DELAY SETTINGS =================

// Delay time in seconds
#define DELAY_SECONDS          0.064f


// ================= LIMITER SETTINGS =================

// Maximum allowed output amplitude
#define LIMITER_THRESHOLD      0.9f


#endif
