#include "high_pass.h"

//  Opens the high pass module and returns the memory size required for the handle structure
STATUS high_pass_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(high_pass_hdl_t);
    return STATUS_OK;
}

//  Initializes the dual internal biquad filter structures sequentially using provided configurations
//  configurations are present in main_test.c
STATUS high_pass_init(high_pass_hdl_t *phdl, const high_pass_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;
    biquad_init(&phdl->bq1, psConfig->s1);
    biquad_init(&phdl->bq2, psConfig->s2);
    return STATUS_OK;
}

//  Processes an audio block by streaming samples through a cascaded two stage biquad filter array
STATUS high_pass_process(high_pass_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;
    static float temp[256];
    biquad_process_block(&phdl->bq1, pfInput, temp, ui32NumSamples);
    biquad_process_block(&phdl->bq2, temp, pfOutput, ui32NumSamples);
    return STATUS_OK;
}

//  Finalizes component tracking contexts to complete lifecycle cleanup operations safely
STATUS high_pass_close(high_pass_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}
