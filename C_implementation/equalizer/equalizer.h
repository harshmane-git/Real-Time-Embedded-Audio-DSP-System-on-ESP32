/* ========================= equalizer.h ========================= */

#ifndef EQUALIZER_H
#define EQUALIZER_H

#include <stdint.h>
#include "common_types.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"

typedef struct
{
    low_pass_config_t  low;
    band_pass_config_t mid;
    high_pass_config_t high;
} equalizer_config_t;

typedef struct
{
    low_pass_hdl_t  low_hdl;
    band_pass_hdl_t mid_hdl;
    high_pass_hdl_t high_hdl;
} equalizer_hdl_t;

STATUS equalizer_open(uint32_t *pui32Size);

STATUS equalizer_init(equalizer_hdl_t *phdl,
                      const equalizer_config_t *psConfig);

STATUS equalizer_process(equalizer_hdl_t *phdl,
                         const float *pfInput,
                         float *pfLow,
                         float *pfMid,
                         float *pfHigh,
                         uint32_t ui32NumSamples);

STATUS equalizer_close(equalizer_hdl_t *phdl);

#endif