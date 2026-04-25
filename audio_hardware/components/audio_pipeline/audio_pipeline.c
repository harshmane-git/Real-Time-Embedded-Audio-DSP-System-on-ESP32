#include "audio_pipeline.h"
#include <stdlib.h>

#define BLOCK_SIZE 256

static float input_block[BLOCK_SIZE];

STATUS audio_Open(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    hdl->mic = malloc(sizeof(mic_hdl));
    hdl->amp = malloc(sizeof(amp_hdl));
    hdl->rb  = NULL;   // not used

    if (!hdl->mic || !hdl->amp)
        return STATUS_NOT_OK;

    return STATUS_OK;
}

STATUS audio_Initialize(audio_hdl *hdl)
{
    mic_config mic_cfg = { .sample_rate = 16000 };
    amp_config amp_cfg = { .sample_rate = 16000 };

    mic_Initialize(hdl->mic, &mic_cfg);
    amp_Initialize(hdl->amp, &amp_cfg);

    return STATUS_OK;
}

STATUS audio_Process(audio_hdl *hdl)
{
    mic_Process(hdl->mic, input_block, BLOCK_SIZE);

    // 🔥 direct pass-through
    amp_Process(hdl->amp, input_block, BLOCK_SIZE);

    return STATUS_OK;
}

STATUS audio_Close(audio_hdl *hdl)
{
    mic_Close(hdl->mic);

    free(hdl->mic);
    free(hdl->amp);

    return STATUS_OK;
}