#ifndef LIMITER_H
#define LIMITER_H

#include <stdint.h>
#include "common_types.h"

typedef struct {
    float fThreshold;
} limiter_config_t;

typedef struct {
    float fThreshold;
} limiter_hdl_t;

STATUS limiter_open(uint32_t *pui32Size);
STATUS limiter_init(limiter_hdl_t *phdl, const limiter_config_t *psConfig);
STATUS limiter_process(limiter_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS limiter_close(limiter_hdl_t *phdl);

#endif