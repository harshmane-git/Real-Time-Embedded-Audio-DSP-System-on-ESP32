#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

// Core Mathematical Parameters (Mandatory for DSP Math Blocks)
#define AUDIO_SAMPLE_RATE     16000     // Required for biquad frequency scaling
#define AUDIO_BLOCK_SIZE      256       // Processing loop buffer block chunk size
#define GAIN_STEPS            4         // Number of master log steps
#define EQ_NUM_PRESETS        4         // Total equalizer presets

#endif