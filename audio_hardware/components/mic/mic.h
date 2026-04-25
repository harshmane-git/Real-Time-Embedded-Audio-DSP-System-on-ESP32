#ifndef MIC_H
#define MIC_H

#include "common_types.h"
#include <stdint.h>
#include "driver/i2s_std.h"

typedef struct
{
    uint32_t sample_rate;
} mic_config;

typedef struct
{
    i2s_chan_handle_t rx_handle;
} mic_hdl;

STATUS mic_Open(uint32_t *size);
STATUS mic_Initialize(mic_hdl *hdl, const mic_config *cfg);
STATUS mic_Process(mic_hdl *hdl, float *output, uint32_t samples);
STATUS mic_Close(mic_hdl *hdl);

#endif