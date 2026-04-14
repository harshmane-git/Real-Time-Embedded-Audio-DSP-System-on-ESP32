#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include "common_types.h"
#include "audio_config.h"
#include <stdint.h>

// ============================================================================
// Ring Buffer - Firmware TRANSPORT LAYER (Not DSP)
// ============================================================================
// Purpose: 
//   - Decouple I2S DMA timing from DSP processing loop
//   - Buffer for EQ, Limiter, Delay processing latency
//   - NOT a delay effect - delay effects are DSP layer responsibility
//
// Architecture:
//   Mic (I2S) → rb_write() → rb_read() → DSP (Gain/EQ/Limiter/Delay) → Amp (I2S)
//   └─ Transport ─┘         └──── DSP Layer ────┘
//
// Size: 8 slots × 256 samples = 2048 samples = 128ms @ 16kHz
//   - Provides margin for DSP processing (EQ ~2ms, Limiter ~5ms, Delay ~50ms)
//   - Handles I2S DMA jitter and context switches
//
// Overflow Policy: CIRCULAR OVERWRITE
//   - When buffer full (all 8 slots available): oldest slot is overwritten
//   - Read pointer auto-advances to maintain synchronization
//   - Prevents underflow but indicates processing can't keep up (warning condition)
//
// Startup Behavior: 
//   - First read will fail (STATUS_NOT_OK) until first slot is written
//   - This is expected - buffer must be primed before audio output
// ============================================================================

typedef struct
{
    uint32_t size;  // Ring buffer size in samples (for future flexibility)
} rb_config;

typedef struct
{
    float *buffer;              // Data buffer: AUDIO_RB_SIZE samples
    uint32_t write_slot;        // Current write slot (0 to AUDIO_RB_SLOTS-1)
    uint32_t read_slot;         // Current read slot (0 to AUDIO_RB_SLOTS-1)
    uint32_t slots_available;   // Number of complete slots available [0, AUDIO_RB_SLOTS]
} rb_hdl;

STATUS rb_Open(uint32_t *size);
STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg);
STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples);
STATUS rb_Reset(rb_hdl *hdl);  // Safe reset function - use this instead of manual manipulation
STATUS rb_Close(rb_hdl *hdl);

#endif