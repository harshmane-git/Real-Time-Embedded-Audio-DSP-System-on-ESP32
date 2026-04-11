#include "gain.h"

STATUS gain_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(gain_hdl_t);
    return STATUS_OK;
}

STATUS gain_init(gain_hdl_t *phdl, const gain_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;
    
    phdl->gain_linear = psConfig->gain_linear;   // Direct linear gain, no dB conversion
    return STATUS_OK;
}

STATUS gain_process(gain_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;
    
    for (uint32_t i = 0; i < ui32NumSamples; i++) {
        pfOutput[i] = pfInput[i] * phdl->gain_linear;
    }
    return STATUS_OK;
}

STATUS gain_close(gain_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}