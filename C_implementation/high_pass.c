#include "high_pass.h"

// Aayush’s High-Pass 3000 Hz coefficients
static const float hp_coeffs[5] = {
    0.41813839f, -0.83627678f, 0.41813839f, -0.46291040f, 0.20964317f
};

void high_pass_init(high_pass_t *hp) {
    biquad_init(&hp->bq, hp_coeffs);
}

void high_pass_process_block(high_pass_t *hp, const float *input, float *output, int len) {
    biquad_process_block(&hp->bq, input, output, len);
}