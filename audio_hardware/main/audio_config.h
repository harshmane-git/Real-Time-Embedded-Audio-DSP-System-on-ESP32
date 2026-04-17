#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

// ============================================================================
// Audio Pipeline Configuration - Centralized Parameters
// ============================================================================
// Change these values here to affect entire pipeline
// ============================================================================

// Audio Processing Parameters
#define AUDIO_SAMPLE_RATE       16000   // Hz - Sample rate for I2S and DSP
#define AUDIO_BLOCK_SIZE        256     // Samples - Processing block size

// Ring Buffer - TRANSPORT LAYER (decoupling, NOT DSP effects)
// Size must accommodate: I2S jitter + DSP processing latency (EQ + Limiter + Delay)
// 8 slots × 256 samples = 2048 samples = 128ms @ 16kHz
#define AUDIO_RB_SLOTS          8
#define AUDIO_RB_SAMPLES_PER_SLOT 256
#define AUDIO_RB_SIZE           (AUDIO_RB_SLOTS * AUDIO_RB_SAMPLES_PER_SLOT)  // 2048

// DSP Configuration - Presets
#define NUM_PRESETS             3
#define PRESET_GAIN_0           1.0f    // Default
#define PRESET_GAIN_1           0.5f    // -6dB
#define PRESET_GAIN_2           2.0f    // +6dB

// Bounds checking
#define PRESET_MIN              0
#define PRESET_MAX              (NUM_PRESETS - 1)

// I2S Configuration - Microphone (RX)
#define MIC_I2S_PORT            I2S_NUM_0
#define MIC_GPIO_BCLK           26
#define MIC_GPIO_WS             25
#define MIC_GPIO_DIN            33

// I2S Configuration - Amplifier (TX)
#define AMP_I2S_PORT            I2S_NUM_1
#define AMP_GPIO_BCLK           27
#define AMP_GPIO_WS             14
#define AMP_GPIO_DOUT           22

// GPIO Configuration - Switches
#define GPIO_SWITCH1            32      // Preset 1
#define GPIO_SWITCH2            34      // Preset 2

#endif