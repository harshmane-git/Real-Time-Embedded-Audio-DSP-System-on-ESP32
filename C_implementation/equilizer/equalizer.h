#ifndef EQUALIZER_H
#define EQUALIZER_H

#include <stdint.h>
#include "common_types.h"
/* ... existing includes ... */

/* IMPORTANT: psConfig must be static or heap-allocated. 
   Do NOT pass a stack-allocated equalizer_config_t on ESP32. */
STATUS equalizer_init(equalizer_hdl_t *phdl,
                      const equalizer_config_t *psConfig);

/* NOTE: This component is non-reentrant due to internal static buffers.
   It MUST be called from a single audio processing task only. */
STATUS equalizer_process(equalizer_hdl_t *phdl,
                         const float *pfInput,
                         float *pfLow,
                         float *pfMid,
                         float *pfHigh,
                         uint32_t ui32NumSamples);

/* NOTE: Update this if dynamic memory is added to nested handles in the future. */
STATUS equalizer_close(equalizer_hdl_t *phdl);

#endif
