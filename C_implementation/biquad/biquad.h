#ifndef BIQUAD_H
#define BIQUAD_H

#include <stdint.h>
#include "common_types.h"

typedef struct {
    float w[2];     // filter state
    float b[3];     // b0, b1, b2
    float a[2];     // a1, a2
} biquad_t;

STATUS biquad_open(uint32_t *pui32Size);
STATUS biquad_init(biquad_t *phdl, const float coeffs[5]);
STATUS biquad_process_block(biquad_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS biquad_close(biquad_t *phdl);

#endif