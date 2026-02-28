#include "low_pass.h"

// Aayush’s Low-Pass 300 Hz coefficients
static const float lp_coeffs[5] = {
    0.00319979f, 0.00639958f, 0.00319979f, -1.83371141f, 0.84651057f
};

void low_pass_init(low_pass_t *lp) {
    biquad_init(&lp->bq, lp_coeffs);
}

void low_pass_process_block(low_pass_t *lp, const float *input, float *output, int len) {
    biquad_process_block(&lp->bq, input, output, len);
}