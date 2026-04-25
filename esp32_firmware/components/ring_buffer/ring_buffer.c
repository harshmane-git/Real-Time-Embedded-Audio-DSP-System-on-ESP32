#include "ring_buffer.h"
#include "audio_config.h"
#include <stdlib.h>
#include <string.h>

// rb_Open: size = AUDIO_RB_SLOTS × AUDIO_BLOCK_SIZE, derived from sample_rate.
// Currently sample_rate is always 16000 Hz — kept as a parameter so the
// pipeline can call it uniformly without knowing the formula.
STATUS rb_Open(uint32_t sample_rate, uint32_t *pui32Size)
{
    if (pui32Size == NULL) return STATUS_NOT_OK;
    (void)sample_rate;  // reserved — formula uses compile-time constants for now
    *pui32Size = AUDIO_RB_SIZE;  // 4 slots × 256 = 1024 samples
    return STATUS_OK;
}

STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg)
{
    if (hdl == NULL || cfg == NULL) return STATUS_NOT_OK;

    hdl->size   = cfg->size;
    hdl->buffer = malloc(sizeof(float) * hdl->size);

    if (hdl->buffer == NULL) return STATUS_NOT_OK;

    hdl->write = 0;
    hdl->read  = 0;
    hdl->count = 0;

    return STATUS_OK;
}

STATUS rb_Reset(rb_hdl *hdl)
{
    if (hdl == NULL || hdl->buffer == NULL) return STATUS_NOT_OK;

    // Zero the buffer and reset all pointers — discards any stale audio
    memset(hdl->buffer, 0, sizeof(float) * hdl->size);
    hdl->write = 0;
    hdl->read  = 0;
    hdl->count = 0;

    return STATUS_OK;
}

STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples)
{
    if (hdl == NULL || input == NULL || output == NULL) return STATUS_NOT_OK;

    // Write new samples — overwrite oldest if full (streaming, non-blocking)
    for (uint32_t i = 0; i < samples; i++)
    {
        if (hdl->count == hdl->size)
        {
            // Buffer full — advance read to drop oldest sample
            hdl->read = (hdl->read + 1) % hdl->size;
            hdl->count--;
        }

        hdl->buffer[hdl->write] = input[i];
        hdl->write = (hdl->write + 1) % hdl->size;
        hdl->count++;
    }

    // Underflow guard — not enough samples to fill output block
    if (hdl->count < samples) return STATUS_NOT_OK;

    // Read back one block
    for (uint32_t i = 0; i < samples; i++)
    {
        output[i]  = hdl->buffer[hdl->read];
        hdl->read  = (hdl->read + 1) % hdl->size;
        hdl->count--;
    }

    return STATUS_OK;
}

STATUS rb_Close(rb_hdl *hdl)
{
    if (hdl == NULL) return STATUS_NOT_OK;
    free(hdl->buffer);
    hdl->buffer = NULL;
    return STATUS_OK;
}