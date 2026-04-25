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

STATUS gain_init(gain_hdl_t *phdl,
                 const gain_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    phdl->gain_linear =
        powf(10.0f,
              psConfig->gain_db / 20.0f);

    return STATUS_OK;
}

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

    for (uint32_t i = 0;
         i < ui32NumSamples;
         i++)
    {
        pfOutput[i] =
            pfInput[i] *
            phdl->gain_linear;
    }

    return STATUS_OK;
}

STATUS gain_close(gain_hdl_t *phdl)
{
    if (phdl == NULL)
        return STATUS_NOT_OK;

    return STATUS_OK;
}

/* ================= BAND GAIN ================= */

STATUS eq_gain_open(uint32_t *pui32Size)
{
    if (pui32Size == NULL)
        return STATUS_NOT_OK;

    *pui32Size = sizeof(eq_gain_hdl_t);
    return STATUS_OK;
}

STATUS eq_gain_init(eq_gain_hdl_t *phdl,
                    const eq_gain_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    gain_config_t low  = { psConfig->low_gain_db };
    gain_config_t mid  = { psConfig->mid_gain_db };
    gain_config_t high = { psConfig->high_gain_db };

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