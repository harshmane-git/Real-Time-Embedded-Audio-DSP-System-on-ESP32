#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include "common_types.h"
#include "audio_config.h"
#include "mic.h"
#include "amp.h"
#include "ring_buffer.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"

// ============================================================================
// Audio Pipeline
// ============================================================================
// Signal flow:
//
//   Mic (I2S)
//     → Ring Buffer          [transport / I2S decoupling]
//     → LP | BP | HP         [EQ — parallel biquad filters]
//     → Weighted sum
//     → Master Gain          [applied AFTER EQ sum]
//     → Delay                [TODO]
//     → Limiter              [TODO]
//     → Amp (I2S)
//
// Preset change triggers full system restart:
//   mic_Close → amp_Close → rb_Reset → mic_Init → amp_Init → filter_init
//   This ensures no stale audio from the old preset leaks through.
//
// preset_request is a volatile flag set by GPIO ISR or switch task:
//   preset_request = 1;  // request preset 1
//   preset_request = 2;  // request preset 2
//   preset_request = 0;  // no request (default)
// The pipeline reads and clears this flag at each block boundary.
// ============================================================================

typedef struct
{
    mic_hdl *mic;
    amp_hdl *amp;
    rb_hdl  *rb;
} audio_hdl;

// Set by GPIO ISR / switch task to request a preset change.
// Pipeline reads this at block boundary and performs full system restart.
extern volatile int preset_request;

STATUS audio_Open(audio_hdl *hdl);
STATUS audio_Initialize(audio_hdl *hdl);
STATUS audio_Process(audio_hdl *hdl);
STATUS audio_Close(audio_hdl *hdl);

#endif