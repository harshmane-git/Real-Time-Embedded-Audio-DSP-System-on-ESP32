#include "band_pass.h"

STATUS band_pass_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(band_pass_hdl_t);
    return STATUS_OK;
}

STATUS band_pass_init(band_pass_hdl_t *phdl, const band_pass_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;
    biquad_init(&phdl->bq1, psConfig->s1);
    biquad_init(&phdl->bq2, psConfig->s2);
    return STATUS_OK;
}

STATUS band_pass_process(band_pass_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;
    float temp[256];
    biquad_process_block(&phdl->bq1, pfInput, temp, ui32NumSamples);
    biquad_process_block(&phdl->bq2, temp, pfOutput, ui32NumSamples);
    return STATUS_OK;
}

STATUS band_pass_close(band_pass_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}