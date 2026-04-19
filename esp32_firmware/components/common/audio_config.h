#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

// ============================================================================
// Audio Pipeline Configuration - Centralized Parameters
// ============================================================================

// Audio Processing Parameters
#define AUDIO_SAMPLE_RATE           16000
#define AUDIO_BLOCK_SIZE            256

// Ring Buffer
// 4 slots × 256 samples = 1024 samples = 64ms @ 16kHz
#define AUDIO_RB_SLOTS              4
#define AUDIO_RB_SAMPLES_PER_SLOT   256
#define AUDIO_RB_SIZE               (AUDIO_RB_SLOTS * AUDIO_RB_SAMPLES_PER_SLOT)  // 1024

// DSP Configuration - Presets
#define NUM_PRESETS             3
#define PRESET_GAIN_0           1.0f
#define PRESET_GAIN_1           0.5f
#define PRESET_GAIN_2           2.0f

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
#define GPIO_SWITCH1            32
#define GPIO_SWITCH2            34

#endif