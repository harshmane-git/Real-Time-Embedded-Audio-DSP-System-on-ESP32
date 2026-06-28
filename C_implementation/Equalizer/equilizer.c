#include "equalizer.h"

STATUS equalizer_open(uint32_t *pui32Size)
{
    if (pui32Size == NULL)
        return STATUS_NOT_OK;

    *pui32Size = sizeof(equalizer_hdl_t);
    return STATUS_OK;
}

//  This passes configuration bundles down to our separate low pass band pass and high pass filters to initialize their coefficients and reset their math history
STATUS equalizer_init(equalizer_hdl_t *phdl,
                      const equalizer_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    // Send coefficient parameters down to initialize the low pass bass filter stage
    low_pass_init(&phdl->low_hdl,
                  &psConfig->low);

    // Send coefficient parameters down to initialize the band pass mid range filter stage
    band_pass_init(&phdl->mid_hdl,
                   &psConfig->mid);

    // Send coefficient parameters down to initialize the high pass treble filter stage
    high_pass_init(&phdl->high_hdl,
                   &psConfig->high);

    return STATUS_OK;
}

// Basic information: This splits our single raw incoming audio stream into three distinct simultaneous frequency paths for bass mid range and treble separation
STATUS equalizer_process(equalizer_hdl_t *phdl,
                         const float *pfInput,
                         float *pfLow,
                         float *pfMid,
                         float *pfHigh,
                         uint32_t ui32NumSamples)
{
    if (phdl == NULL ||
        pfInput == NULL ||
        pfLow == NULL ||
        pfMid == NULL ||
        pfHigh == NULL)
    {
        return STATUS_NOT_OK;
    }

    // Logic: Checking for empty and full states is not actually implementing boundary checks here because this master equalizer relies on sub filters that update memory arrays 

    // Run our raw input audio through the low pass block
    low_pass_process(&phdl->low_hdl,
                     pfInput,
                     pfLow,
                     ui32NumSamples);

    // Run the exact same raw input through the band pass block 
    band_pass_process(&phdl->mid_hdl,
                      pfInput,
                      pfMid,
                      ui32NumSamples);

    // Run the exact same raw input through the high pass block 
    high_pass_process(&phdl->high_hdl,
                      pfInput,
                      pfHigh,
                      ui32NumSamples);

    return STATUS_OK;
}

STATUS equalizer_close(equalizer_hdl_t *phdl)
{
    if (phdl == NULL)
        return STATUS_NOT_OK;

    low_pass_close(&phdl->low_hdl);
    
    band_pass_close(&phdl->mid_hdl);
    
    high_pass_close(&phdl->high_hdl);

    return STATUS_OK;
}
