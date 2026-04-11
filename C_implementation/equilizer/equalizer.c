#include "equalizer.h"
#include <math.h>

STATUS equalizer_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(equalizer_hdl_t);
    return STATUS_OK;
}

STATUS equalizer_init(equalizer_hdl_t *phdl, const equalizer_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;

    low_pass_init(&phdl->low, &psConfig->low);
    band_pass_init(&phdl->mid, &psConfig->mid);
    high_pass_init(&phdl->high, &psConfig->high);

    phdl->low_gain_linear  = powf(10.0f, psConfig->low_gain_db / 20.0f);
    phdl->mid_gain_linear  = powf(10.0f, psConfig->mid_gain_db / 20.0f);
    phdl->high_gain_linear = powf(10.0f, psConfig->high_gain_db / 20.0f);

    return STATUS_OK;
}

STATUS equalizer_process(equalizer_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;

    float low_out[256], mid_out[256], high_out[256];

    low_pass_process(&phdl->low, pfInput, low_out, ui32NumSamples);
    band_pass_process(&phdl->mid, pfInput, mid_out, ui32NumSamples);
    high_pass_process(&phdl->high, pfInput, high_out, ui32NumSamples);

    for (uint32_t i = 0; i < ui32NumSamples; i++) {
        low_out[i]  *= phdl->low_gain_linear;
        mid_out[i]  *= phdl->mid_gain_linear;
        high_out[i] *= phdl->high_gain_linear;
        pfOutput[i]  = low_out[i] + mid_out[i] + high_out[i];
    }

    return STATUS_OK;
}

STATUS equalizer_close(equalizer_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}