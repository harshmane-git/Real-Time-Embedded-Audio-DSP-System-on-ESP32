/* ========================= gain.h ========================= */


#ifndef GAIN_H
#define GAIN_H

#include <stdint.h>
#include "common_types.h"

typedef struct
{
    float gain_db;
} gain_config_t;

/* EQ band gain config lives in gain block */
typedef struct
{
    float low_gain_db;
    float mid_gain_db;
    float high_gain_db;
} eq_gain_config_t;

typedef struct
{
    float gain_linear;
} gain_hdl_t;

typedef struct
{
    gain_hdl_t low;
    gain_hdl_t mid;
    gain_hdl_t high;
} eq_gain_hdl_t;

STATUS gain_open(uint32_t *pui32Size);

STATUS gain_init(gain_hdl_t *phdl,
                 const gain_config_t *psConfig);

STATUS gain_process(gain_hdl_t *phdl,
                    const float *pfInput,
                    float *pfOutput,
                    uint32_t ui32NumSamples);

STATUS gain_close(gain_hdl_t *phdl);

/* band gain helper functions */
STATUS eq_gain_open(uint32_t *pui32Size);

STATUS eq_gain_init(eq_gain_hdl_t *phdl,
                    const eq_gain_config_t *psConfig);

STATUS eq_gain_close(eq_gain_hdl_t *phdl);

#endif