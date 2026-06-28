/* ========================= equalizer.h ========================= */

#ifndef EQUALIZER_H
#define EQUALIZER_H

#include <stdint.h>
#include "common_types.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"

//  group bundles the individual configuration data packages needed by our three independent filter blocks
typedef struct
{
    low_pass_config_t  low;
    band_pass_config_t mid;
    high_pass_config_t high;
} equalizer_config_t;

//  This master state handle acts as a combined memory block that contains the individual tracking handles for all three sub filter blocks
typedef struct
{
    low_pass_hdl_t  low_hdl;
    band_pass_hdl_t mid_hdl;
    high_pass_hdl_t high_hdl;
} equalizer_hdl_t;

//  This function calculates the total byte size required to allocate our master equalizer state structure
STATUS equalizer_open(uint32_t *pui32Size);

//  This function setups our sub filters by passing down their matching configuration parameters
STATUS equalizer_init(equalizer_hdl_t *phdl,
                      const equalizer_config_t *psConfig);

//  This splits our single raw input audio block into separate simultaneously calculated bass midrange and treble stream arrays
STATUS equalizer_process(equalizer_hdl_t *phdl,
                         const float *pfInput,
                         float *pfLow,
                         float *pfMid,
                         float *pfHigh,
                         uint32_t ui32NumSamples);

//  This executes cleanup routines across all sub filter blocks to safely complete the lifecycle teardown process
STATUS equalizer_close(equalizer_hdl_t *phdl);

#endif
