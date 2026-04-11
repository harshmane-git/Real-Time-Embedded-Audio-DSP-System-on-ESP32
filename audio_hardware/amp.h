#ifndef AMP_H
#define AMP_H

#include "common_types.h"
#include <stdint.h>

typedef struct
{
    uint32_t sample_rate;
} amp_config;

typedef struct
{
    int dummy;
} amp_hdl;

STATUS amp_Open(uint32_t *size);
STATUS amp_Initialize(amp_hdl *hdl, const amp_config *cfg);
STATUS amp_Process(amp_hdl *hdl, const float *input, uint32_t samples);
STATUS amp_Close(amp_hdl *hdl);

#endif