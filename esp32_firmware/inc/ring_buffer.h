#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include "common_types.h"
#include <stdint.h>

typedef struct
{
    uint32_t size;          // total number of float samples in the buffer
    uint32_t sample_rate;   // used to derive buffer size in rb_Open
} rb_config;

typedef struct
{
    float    *buffer;
    uint32_t  size;
    uint32_t  write;
    uint32_t  read;
    uint32_t  count;
} rb_hdl;

// rb_Open: derives required buffer size from sample_rate and writes it to
// *pui32Size. audio_pipeline passes this value into rb_config.size so the
// caller never hard-codes the buffer length.
STATUS rb_Open(uint32_t sample_rate, uint32_t *pui32Size);

STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg);
STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples);

// rb_Reset: flushes all buffered audio and resets read/write pointers.
// Called during preset change to discard stale audio before re-init.
STATUS rb_Reset(rb_hdl *hdl);

STATUS rb_Close(rb_hdl *hdl);

#endif