#include "audio_pipeline.h"
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 256

static float input_block[BLOCK_SIZE];
static float output_block[BLOCK_SIZE];

static volatile int preset_request = 0;
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