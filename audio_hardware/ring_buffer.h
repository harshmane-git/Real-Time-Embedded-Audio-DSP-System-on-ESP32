#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include "common_types.h"
#include <stdint.h>

typedef struct
{
    uint32_t size;
} rb_config;

typedef struct
{
    float *buffer;
    uint32_t size;
    uint32_t write;
    uint32_t read;
    uint32_t count;
} rb_hdl;

STATUS rb_Open(uint32_t *size);
STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg);
STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples);
STATUS rb_Close(rb_hdl *hdl);

#endif