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
//   Mic (I2S) → rb_write() → rb_read() → DSP (EQ → Gain → Delay → Limiter) → Amp (I2S)
//   └─ Transport ─┘         └──────────── DSP Layer ────────────────────────┘
//
// Size: AUDIO_RB_SLOTS × AUDIO_RB_SAMPLES_PER_SLOT samples
//   - samples_per_slot is derived from sample_rate: (sample_rate / 1000) * BLOCK_DURATION_MS
//   - Provides margin for DSP processing (EQ ~2ms, Limiter ~5ms, Delay ~50ms)
//   - Handles I2S DMA jitter and context switches
//
// Overflow Policy: CIRCULAR OVERWRITE
//   - When buffer full (all slots available): oldest slot is overwritten
//   - Read pointer auto-advances to maintain synchronization
//   - Prevents underflow but indicates processing can't keep up (warning condition)
//
// Startup Behavior:
//   - First read will fail (STATUS_NOT_OK) until first slot is written
//   - This is expected - buffer must be primed before audio output
//
// Usage:
//   uint32_t buf_size = 0;
//   rb_Open(AUDIO_SAMPLE_RATE, &buf_size);   // query required buffer size in samples
//   rb_Initialize(&hdl, &cfg);               // allocate and prepare
// ============================================================================

typedef struct
{
    uint32_t size;          // Ring buffer size in samples (slots × samples_per_slot)
    uint32_t sample_rate;   // Sample rate in Hz — used to derive block duration
} rb_config;

typedef struct
{
    float    *buffer;           // Data buffer: size = slots × samples_per_slot
    uint32_t  write_slot;       // Current write slot [0, AUDIO_RB_SLOTS-1]
    uint32_t  read_slot;        // Current read slot  [0, AUDIO_RB_SLOTS-1]
    uint32_t  slots_available;  // Number of complete slots available [0, AUDIO_RB_SLOTS]
    uint32_t  samples_per_slot; // Derived from sample_rate at Open time
} rb_hdl;

// ----------------------------------------------------------------------------
// rb_Open
// ----------------------------------------------------------------------------
// Call this BEFORE rb_Initialize to query how many samples the buffer needs.
// Caller uses the returned buffer_size_out to understand memory requirements.
//
// @param sample_rate      : Audio sample rate in Hz (e.g. 16000)
// @param buffer_size_out  : OUTPUT — total buffer size in samples
//                           = AUDIO_RB_SLOTS × samples_per_slot
// @return STATUS_OK always (validates sample_rate > 0)
// ----------------------------------------------------------------------------
STATUS rb_Open(uint32_t sample_rate, uint32_t *buffer_size_out);

STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg);
STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples);
STATUS rb_Reset(rb_hdl *hdl);
STATUS rb_Close(rb_hdl *hdl);

#endif