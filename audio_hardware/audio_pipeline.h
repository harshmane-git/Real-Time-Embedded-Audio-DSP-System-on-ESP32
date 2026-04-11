#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include "common_types.h"
#include "mic.h"
#include "amp.h"
#include "ring_buffer.h"

typedef struct
{
    mic_hdl *mic;
    amp_hdl *amp;
    rb_hdl *rb;
} audio_hdl;

STATUS audio_Open(audio_hdl *hdl);
STATUS audio_Initialize(audio_hdl *hdl);
STATUS audio_Process(audio_hdl *hdl);
STATUS audio_Close(audio_hdl *hdl);

#endif