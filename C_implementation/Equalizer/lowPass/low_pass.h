#ifndef LOW_PASS_H
#define LOW_PASS_H

#include <stdint.h>
#include "common_types.h"
#include "biquad.h"

typedef struct {
    float s1[5];
    float s2[5];
} low_pass_config_t;

typedef struct {
    biquad_t bq1;
    biquad_t bq2;
} low_pass_hdl_t;

STATUS low_pass_open(uint32_t *pui32Size);
STATUS low_pass_init(low_pass_hdl_t *phdl, const low_pass_config_t *psConfig);
STATUS low_pass_process(low_pass_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS low_pass_close(low_pass_hdl_t *phdl);

#endif