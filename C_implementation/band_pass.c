#include "band_pass.h"

// Aayush’s Band-Pass 1000 Hz coefficients
static const float bp_coeffs[5] = {
    0.16061027f, 0.00000000f, -0.16061027f, -1.55098998f, 0.67877946f
};

void band_pass_init(band_pass_t *bp) {
    biquad_init(&bp->bq, bp_coeffs);
}

void band_pass_process_block(band_pass_t *bp, const float *input, float *output, int len) {
    biquad_process_block(&bp->bq, input, output, len);
}