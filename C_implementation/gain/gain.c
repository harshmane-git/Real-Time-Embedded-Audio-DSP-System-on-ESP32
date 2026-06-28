/* ========================= gain.c ========================= */

#include "gain.h"
#include <math.h>

STATUS gain_open(uint32_t *pui32Size)
{
    if (pui32Size == NULL)
        return STATUS_NOT_OK;

    *pui32Size = sizeof(gain_hdl_t);
    return STATUS_OK;
}

//  This converts human readable decibel parameters into a raw linear multiplier used directly by our processing loop
STATUS gain_init(gain_hdl_t *phdl,
                 const gain_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    // The decibel scale is logarithmic so we convert it to a linear factor using a base ten power formula
    phdl->gain_linear =
        powf(10.0f,
              psConfig->gain_db / 20.0f);

    return STATUS_OK;
}

//  This runs sample by sample multiplying our linear factor against incoming audio data to change the volume amplitude
STATUS gain_process(gain_hdl_t *phdl,
                    const float *pfInput,
                    float *pfOutput,
                    uint32_t ui32NumSamples)
{
    if (phdl == NULL ||
        pfInput == NULL ||
        pfOutput == NULL)
    {
        return STATUS_NOT_OK;
    }

    // Process our block by applying the pre calculated multiplier across every single incoming audio sample
    for (uint32_t i = 0;
         i < ui32NumSamples;
         i++)
    {
        //  gain scaling works sample by sample on independent variables
        
        // Multiply the raw sample data directly by the linear scale factor to generate the adjusted volume output
        pfOutput[i] =
            pfInput[i] *
            phdl->gain_linear;
    }

    return STATUS_OK;
}

//  This closes out our standard single gain context instance safely
STATUS gain_close(gain_hdl_t *phdl)
{
    if (phdl == NULL)
        return STATUS_NOT_OK;

    return STATUS_OK;
}

/* ================= BAND GAIN ================= */

//  This function figures out how much memory our nested three band equalizer gain structure needs to allocate
STATUS eq_gain_open(uint32_t *pui32Size)
{
    if (pui32Size == NULL)
        return STATUS_NOT_OK;

    *pui32Size = sizeof(eq_gain_hdl_t);
    return STATUS_OK;
}

//  This extracts individual bass midrange and treble decibel values and passes them down to initialize three separate standard gain tracks
STATUS eq_gain_init(eq_gain_hdl_t *phdl,
                    const eq_gain_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    // Package the separate band configurations into standard structure parameters
    gain_config_t low  = { psConfig->low_gain_db };
    gain_config_t mid  = { psConfig->mid_gain_db };
    gain_config_t high = { psConfig->high_gain_db };

    // Initialize the separate linear multipliers 
    gain_init(&phdl->low,  &low);
    gain_init(&phdl->mid,  &mid);
    gain_init(&phdl->high, &high);

    return STATUS_OK;
}

STATUS eq_gain_close(eq_gain_hdl_t *phdl)
{
    if (phdl == NULL)
        return STATUS_NOT_OK;

    return STATUS_OK;
}
