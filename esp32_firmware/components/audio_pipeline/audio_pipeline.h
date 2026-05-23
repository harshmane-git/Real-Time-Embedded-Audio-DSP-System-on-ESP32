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
#include "gain.h"
#include "delay.h"

// ============================================================================
// Signal flow:
//   Mic → Ring Buffer
//       → LP + BP + HP  (hardcoded, always ON)
//       → Sum
//       → EQ            (SW1 cycles 4 presets)
//       → Master Gain   (SW2 cycles gain steps)
//       → Delay         (SW3 toggles ON/OFF)
//       → Amp
// ============================================================================

typedef struct
{
    mic_hdl *mic;
    amp_hdl *amp;
    rb_hdl  *rb;
} audio_hdl;

// EQ state — shared with main.c
extern int eq_preset;
extern int eq_preset_pending;

STATUS audio_Open(audio_hdl *hdl);
STATUS audio_Initialize(audio_hdl *hdl);
STATUS audio_Process(audio_hdl *hdl);
STATUS audio_Close(audio_hdl *hdl);

// Called from main.c BEFORE audio_Process — block boundary safe zone
void audio_apply_pending(void);

// Called from main.c on SW2/SW3 press
void audio_set_gain(int index);
void audio_toggle_delay(void);

#endif