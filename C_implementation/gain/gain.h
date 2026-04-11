#ifndef GAIN_H
#define GAIN_H

#include <stdint.h>
#include "common_types.h"

typedef struct {
    float gain_linear;     // Linear gain (e.g. 2.0 = 2x amplitude, 1.0 = original)
} gain_config_t;

typedef struct {
    float gain_linear;
} gain_hdl_t;

STATUS gain_open(uint32_t *pui32Size);
STATUS gain_init(gain_hdl_t *phdl, const gain_config_t *psConfig);
STATUS gain_process(gain_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS gain_close(gain_hdl_t *phdl);

#endif