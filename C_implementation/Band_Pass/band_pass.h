#ifndef BAND_PASS_H
#define BAND_PASS_H

#include <stdint.h>
#include "common_types.h"
#include "biquad.h"

typedef struct {
    float s1[5];
    float s2[5];
} band_pass_config_t;

typedef struct {
    biquad_t bq1;
    biquad_t bq2;
} band_pass_hdl_t;

STATUS band_pass_open(uint32_t *pui32Size);
STATUS band_pass_init(band_pass_hdl_t *phdl, const band_pass_config_t *psConfig);
STATUS band_pass_process(band_pass_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS band_pass_close(band_pass_hdl_t *phdl);

#endif