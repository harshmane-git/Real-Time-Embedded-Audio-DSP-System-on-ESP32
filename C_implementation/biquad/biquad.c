#include "biquad.h"

STATUS biquad_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(biquad_t);
    return STATUS_OK;
}

STATUS biquad_init(biquad_t *phdl, const float coeffs[5]) {
    if (phdl == NULL || coeffs == NULL) return STATUS_NOT_OK;

    phdl->b[0] = coeffs[0];
    phdl->b[1] = coeffs[1];
    phdl->b[2] = coeffs[2];
    phdl->a[0] = coeffs[3];
    phdl->a[1] = coeffs[4];

    phdl->w[0] = 0.0f;
    phdl->w[1] = 0.0f;

    return STATUS_OK;
}

STATUS biquad_process_block(biquad_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;

    for (uint32_t i = 0; i < ui32NumSamples; i++) {
        float x = pfInput[i];
        float w0 = x - phdl->a[0] * phdl->w[0] - phdl->a[1] * phdl->w[1];
        pfOutput[i] = phdl->b[0] * w0 + phdl->b[1] * phdl->w[0] + phdl->b[2] * phdl->w[1];
        phdl->w[1] = phdl->w[0];
        phdl->w[0] = w0;
    }
    return STATUS_OK;
}

STATUS biquad_close(biquad_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}