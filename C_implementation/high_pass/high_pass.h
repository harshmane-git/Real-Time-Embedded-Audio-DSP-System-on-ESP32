#ifndef HIGH_PASS_H
#define HIGH_PASS_H

#include <stdint.h>
#include "common_types.h"
#include "biquad.h"

typedef struct {
    float s1[5];
    float s2[5];
} high_pass_config_t;

typedef struct {
    biquad_t bq1;
    biquad_t bq2;
} high_pass_hdl_t;

STATUS high_pass_open(uint32_t *pui32Size);
STATUS high_pass_init(high_pass_hdl_t *phdl, const high_pass_config_t *psConfig);
STATUS high_pass_process(high_pass_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS high_pass_close(high_pass_hdl_t *phdl);

#endif