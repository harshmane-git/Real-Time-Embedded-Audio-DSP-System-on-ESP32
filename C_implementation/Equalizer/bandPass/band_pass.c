#include "band_pass.h"

// Basic information: Opens the band pass module and returns the memory size required for the handle structure
STATUS band_pass_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(band_pass_hdl_t);
    return STATUS_OK;
}

// Basic information: Initializes the dual internal biquad filter structures sequentially using provided configurations
STATUS band_pass_init(band_pass_hdl_t *phdl, const band_pass_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;
    biquad_init(&phdl->bq1, psConfig->s1);
    biquad_init(&phdl->bq2, psConfig->s2);
    return STATUS_OK;
}

// Basic information: Processes an audio block by streaming samples through a cascaded two stage biquad filter array
STATUS band_pass_process(band_pass_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;
    static float temp[256];
    biquad_process_block(&phdl->bq1, pfInput, temp, ui32NumSamples);
    biquad_process_block(&phdl->bq2, temp, pfOutput, ui32NumSamples);
    return STATUS_OK;
}

// Basic information: Finalizes component tracking contexts to complete lifecycle cleanup operations safely
STATUS band_pass_close(band_pass_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}
