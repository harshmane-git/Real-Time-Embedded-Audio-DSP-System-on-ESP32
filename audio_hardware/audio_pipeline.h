#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include "common_types.h"
#include "audio_config.h"
#include "mic.h"
#include "amp.h"
#include "ring_buffer.h"

typedef struct
{
    mic_hdl *mic;
    amp_hdl *amp;
    rb_hdl *rb;
} audio_hdl;

// Error codes for audio processing
typedef enum
{
    AUDIO_ERR_NONE = 0,
    AUDIO_ERR_INVALID_PRESET = 1,
    AUDIO_ERR_RB_UNDERFLOW = 2,
    AUDIO_ERR_MIC_FAILED = 3,
    AUDIO_ERR_AMP_FAILED = 4,
} audio_error_t;

STATUS audio_Open(audio_hdl *hdl);
STATUS audio_Initialize(audio_hdl *hdl);
STATUS audio_Process(audio_hdl *hdl);
STATUS audio_Close(audio_hdl *hdl);

#endif