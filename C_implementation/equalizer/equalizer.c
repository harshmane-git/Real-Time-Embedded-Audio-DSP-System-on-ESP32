/* ========================= equalizer.c ========================= */

#include "equalizer.h"

STATUS equalizer_open(uint32_t *pui32Size)
{
    if (pui32Size == NULL)
        return STATUS_NOT_OK;

    *pui32Size = sizeof(equalizer_hdl_t);
    return STATUS_OK;
}

STATUS equalizer_init(equalizer_hdl_t *phdl,
                      const equalizer_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    low_pass_init(&phdl->low_hdl,
                  &psConfig->low);

    band_pass_init(&phdl->mid_hdl,
                   &psConfig->mid);

    high_pass_init(&phdl->high_hdl,
                   &psConfig->high);

    return STATUS_OK;
}

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

    low_pass_process(&phdl->low_hdl,
                     pfInput,
                     pfLow,
                     ui32NumSamples);

    band_pass_process(&phdl->mid_hdl,
                      pfInput,
                      pfMid,
                      ui32NumSamples);

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