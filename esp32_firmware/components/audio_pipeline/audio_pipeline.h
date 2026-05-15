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
#include "equalizer.h"

// ============================================================================
// Signal flow:
//   Mic → Ring Buffer
//       → LP + BP + HP (hardcoded biquad, always ON)
//       → Sum
//       → EQ           (SW1 cycles 4 presets: flat/bass/mid/treble)
//       → Gain         (SW2 cycles {0.5, 1.0, 1.5, 2.0})
//       → Delay        (SW3 — passthrough until C_implementation delivers)
//       → Amp
//
// SW1, SW2, SW3 are independent and stackable.
// All effects OFF at startup — unity gain, flat response.
// ============================================================================

typedef struct
{
    mic_hdl *mic;
    amp_hdl *amp;
    rb_hdl  *rb;
} audio_hdl;

// Set by main.c GPIO reads every loop — read by audio_pipeline each block
extern volatile int sw1_level;
extern volatile int sw2_level;
extern volatile int sw3_level;

STATUS audio_Open(audio_hdl *hdl);
STATUS audio_Initialize(audio_hdl *hdl);
STATUS audio_Process(audio_hdl *hdl);
STATUS audio_Close(audio_hdl *hdl);

void audio_apply_pending(void);

#endif