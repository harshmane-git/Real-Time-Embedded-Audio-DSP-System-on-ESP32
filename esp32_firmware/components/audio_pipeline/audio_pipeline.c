#include "audio_pipeline.h"
#include "audio_config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Stage 2: mic → ring buffer → amp
// ============================================================================

static float input_block[AUDIO_BLOCK_SIZE];   // mic → rb
static float rb_out_block[AUDIO_BLOCK_SIZE];  // rb  → amp

STATUS audio_Open(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    hdl->mic = malloc(sizeof(mic_hdl));
    hdl->amp = malloc(sizeof(amp_hdl));
    hdl->rb  = malloc(sizeof(rb_hdl));

    if (!hdl->mic || !hdl->amp || !hdl->rb)
    {
        printf("[ERROR] audio_Open: malloc failed\n");
        return STATUS_NOT_OK;
    }

    return STATUS_OK;
}

STATUS audio_Initialize(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    // --- Mic ---
    mic_config mic_cfg = { .sample_rate = AUDIO_SAMPLE_RATE };
    if (mic_Initialize(hdl->mic, &mic_cfg) != STATUS_OK)
    {
        printf("[ERROR] mic_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    // --- Amp ---
    amp_config amp_cfg = { .sample_rate = AUDIO_SAMPLE_RATE };
    if (amp_Initialize(hdl->amp, &amp_cfg) != STATUS_OK)
    {
        printf("[ERROR] amp_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    // --- Ring Buffer ---
    uint32_t rb_size = 0;
    if (rb_Open(AUDIO_SAMPLE_RATE, &rb_size) != STATUS_OK)
    {
        printf("[ERROR] rb_Open failed\n");
        return STATUS_NOT_OK;
    }

    rb_config rb_cfg = {
        .size        = rb_size,
        .sample_rate = AUDIO_SAMPLE_RATE
    };

    if (rb_Initialize(hdl->rb, &rb_cfg) != STATUS_OK)
    {
        printf("[ERROR] rb_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    printf("[INIT] Stage 2 ready — mic → rb (%lu samples) → amp\n",
           (unsigned long)rb_size);

    return STATUS_OK;
}

STATUS audio_Process(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    // 1. Read from mic
    mic_Process(hdl->mic, input_block, AUDIO_BLOCK_SIZE);

    // 2. Push through ring buffer
    STATUS rb_status = rb_Process(hdl->rb, input_block, rb_out_block, AUDIO_BLOCK_SIZE);

    if (rb_status != STATUS_OK)
    {
        // Underflow on first few blocks — pass silence to amp, don't stall
        memset(rb_out_block, 0, sizeof(rb_out_block));
    }

    // 3. Send to amp
    amp_Process(hdl->amp, rb_out_block, AUDIO_BLOCK_SIZE);

    return STATUS_OK;
}

STATUS audio_Close(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    mic_Close(hdl->mic);
    amp_Close(hdl->amp);
    rb_Close(hdl->rb);

    free(hdl->mic);
    free(hdl->amp);
    free(hdl->rb);

    return STATUS_OK;
}