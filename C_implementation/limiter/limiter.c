#include "limiter.h"

STATUS limiter_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(limiter_hdl_t);
    return STATUS_OK;
}

STATUS limiter_init(limiter_hdl_t *phdl, const limiter_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;
    phdl->fThreshold = psConfig->fThreshold;
    return STATUS_OK;
}

STATUS limiter_process(limiter_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;
    for (uint32_t i = 0; i < ui32NumSamples; i++) {
        float val = pfInput[i];
        if (val > phdl->fThreshold)      val = phdl->fThreshold;
        else if (val < -phdl->fThreshold) val = -phdl->fThreshold;
        pfOutput[i] = val;
    }
    return STATUS_OK;
}

STATUS limiter_close(limiter_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}