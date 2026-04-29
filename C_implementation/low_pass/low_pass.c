#include "low_pass.h"
#include <stddef.h>

STATUS low_pass_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(low_pass_hdl_t);
    return STATUS_OK;
}

STATUS low_pass_init(low_pass_hdl_t *phdl, const low_pass_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;
    
    /* Biquad handles stability checks internally */
    if (biquad_init(&phdl->bq1, psConfig->s1) != STATUS_OK) return STATUS_NOT_OK;
    if (biquad_init(&phdl->bq2, psConfig->s2) != STATUS_OK) return STATUS_NOT_OK;
    
    return STATUS_OK;
}

STATUS low_pass_process(low_pass_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;

    /* 'static' prevents 1KB stack consumption per call on ESP32 */
    /* Note: Non-reentrant; call from one task only */
    static float temp[256]; 
    
    biquad_process_block(&phdl->bq1, pfInput, temp, ui32NumSamples);
    biquad_process_block(&phdl->bq2, temp, pfOutput, ui32NumSamples);
    
    return STATUS_OK;
}

STATUS low_pass_close(low_pass_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}
