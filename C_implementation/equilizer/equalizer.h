#ifndef EQUALIZER_H
#define EQUALIZER_H

#include <stdint.h>
#include "common_types.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"

typedef struct {
    low_pass_config_t  low;
    band_pass_config_t mid;
    high_pass_config_t high;
    float low_gain_db;
    float mid_gain_db;
    float high_gain_db;
} equalizer_config_t;

typedef struct {
    low_pass_hdl_t  low;
    band_pass_hdl_t mid;
    high_pass_hdl_t high;
    float low_gain_linear;
    float mid_gain_linear;
    float high_gain_linear;
} equalizer_hdl_t;

STATUS equalizer_open(uint32_t *pui32Size);
STATUS equalizer_init(equalizer_hdl_t *phdl, const equalizer_config_t *psConfig);
STATUS equalizer_process(equalizer_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS equalizer_close(equalizer_hdl_t *phdl);

#endif