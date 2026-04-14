#include "audio_pipeline.h"
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 256

static float input_block[BLOCK_SIZE];
static float output_block[BLOCK_SIZE];

volatile int preset_request = 0;
static int current_preset = 0;

typedef struct
{
    float gain;
} preset_config;

static preset_config presets[] =
{
    {1.0f},
    {0.5f},
    {2.0f}
};

STATUS audio_Open(audio_hdl *hdl)
{
    hdl->mic = malloc(sizeof(mic_hdl));
    hdl->amp = malloc(sizeof(amp_hdl));
    hdl->rb  = malloc(sizeof(rb_hdl));

    if (!hdl->mic || !hdl->amp || !hdl->rb)
    {
        if (hdl->mic) free(hdl->mic);
        if (hdl->amp) free(hdl->amp);
        if (hdl->rb)  free(hdl->rb);

        return STATUS_NOT_OK;
    }

    return STATUS_OK;
}

STATUS audio_Initialize(audio_hdl *hdl)
{
    mic_config mic_cfg = {16000};
    amp_config amp_cfg = {16000};
    rb_config rb_cfg = {0};

    if (mic_Initialize(hdl->mic, &mic_cfg) != STATUS_OK)
        return STATUS_NOT_OK;

    if (amp_Initialize(hdl->amp, &amp_cfg) != STATUS_OK)
        return STATUS_NOT_OK;

    if (rb_Initialize(hdl->rb, &rb_cfg) != STATUS_OK)
        return STATUS_NOT_OK;

    return STATUS_OK;
}

STATUS audio_Process(audio_hdl *hdl)
{
    mic_Process(hdl->mic, input_block, BLOCK_SIZE);

    if (preset_request != 0)
    {
        current_preset = preset_request;
        preset_request = 0;
        printf("Preset changed: %d\n", current_preset);
    }

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        input_block[i] *= presets[current_preset].gain;
    }

    if (rb_Process(hdl->rb, input_block, output_block, BLOCK_SIZE) == STATUS_OK)
    {
        amp_Process(hdl->amp, output_block, BLOCK_SIZE);
    }

    return STATUS_OK;
}

STATUS audio_Close(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    if (hdl->rb)
    {
        rb_Close(hdl->rb);
        free(hdl->rb);
    }

    if (hdl->mic)
    {
        mic_Close(hdl->mic);
        free(hdl->mic);
    }

    if (hdl->amp)
    {
        amp_Close(hdl->amp);
        free(hdl->amp);
    }

    return STATUS_OK;
}