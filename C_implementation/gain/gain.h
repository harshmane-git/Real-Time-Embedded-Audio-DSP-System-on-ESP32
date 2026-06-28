#ifndef GAIN_H
#define GAIN_H

#include <stdint.h>
#include "common_types.h"

//  This structural parameter holds the raw human readable decibel target for a single volume control track
typedef struct
{
    float gain_db;
} gain_config_t;

//  This configuration group packages the independent decibel settings for our separate bass midrange and treble channels
typedef struct
{
    float low_gain_db;
    float mid_gain_db;
    float high_gain_db;
} eq_gain_config_t;

//  This single handle structure holds our precalculated decimal multiplier used inside our audio scaling loop
typedef struct
{
    float gain_linear;
} gain_hdl_t;

//  This master equalizer handle contains three individual gain handles to track linear multipliers for separate bands simultaneously
typedef struct
{
    gain_hdl_t low;
    gain_hdl_t mid;
    gain_hdl_t high;
} eq_gain_hdl_t;

//  This function calculates the byte size needed to allocate memory for a standard single channel gain handle
STATUS gain_open(uint32_t *pui32Size);

//  This function converts decibels into a linear math multiplier and saves it inside our handle memory
STATUS gain_init(gain_hdl_t *phdl,
                 const gain_config_t *psConfig);

//  This function multiplies our linear factor against an array block of audio samples to scale the volume amplitude
STATUS gain_process(gain_hdl_t *phdl,
                    const float *pfInput,
                    float *pfOutput,
                    uint32_t ui32NumSamples);

//  This function securely finishes the runtime lifecycle for a single standard gain instance
STATUS gain_close(gain_hdl_t *phdl);

//  This function calculates the total byte footprint needed to allocate our nested three band equalizer gain structure
STATUS eq_gain_open(uint32_t *pui32Size);

//  This function initializes our three separate equalizer volume channels by passing down their configured parameters
STATUS eq_gain_init(eq_gain_hdl_t *phdl,
                    const eq_gain_config_t *psConfig);

STATUS eq_gain_close(eq_gain_hdl_t *phdl);

#endif
