#include "ring_buffer.h"
#include <stdlib.h>

STATUS rb_Open(uint32_t *size)
{
    *size = sizeof(rb_hdl);
    return STATUS_OK;
}

STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg)
{
    hdl->size = cfg->size;
    hdl->buffer = malloc(sizeof(float) * hdl->size);

    hdl->write = 0;
    hdl->read = 0;
    hdl->count = 0;

    return STATUS_OK;
}

STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples)
{
    // write
    for (uint32_t i = 0; i < samples; i++)
    {
        if (hdl->count < hdl->size)
        {
            hdl->buffer[hdl->write] = input[i];
            hdl->write = (hdl->write + 1) % hdl->size;
            hdl->count++;
        }
        else
        {
            return STATUS_NOT_OK; //buffer full flag
        }
    }

    // read
    if (hdl->count < samples) return STATUS_NOT_OK;

    for (uint32_t i = 0; i < samples; i++)
    {
        output[i] = hdl->buffer[hdl->read];
        hdl->read = (hdl->read + 1) % hdl->size;
        hdl->count--;
    }

    return STATUS_OK;
}

STATUS rb_Close(rb_hdl *hdl)
{
    free(hdl->buffer);
    return STATUS_OK;
}