#include "../inc/biquad.h"     // or "/biquad.h" if you still have path issues

void biquad_init(biquad_t *bq, const float coeffs[5]) {
    bq->b0 = coeffs[0];
    bq->b1 = coeffs[1];
    bq->b2 = coeffs[2];
    bq->a1 = coeffs[3];
    bq->a2 = coeffs[4];
    bq->w1 = 0.0f;
    bq->w2 = 0.0f;
}

float biquad_process_sample(biquad_t *bq, float x) {
    float w0 = x - bq->a1 * bq->w1 - bq->a2 * bq->w2;
    float y = bq->b0 * w0 + bq->b1 * bq->w1 + bq->b2 * bq->w2;
    bq->w2 = bq->w1;
    bq->w1 = w0;
    return y;
}

void biquad_process_block(biquad_t *bq, const float *input, float *output, int len) {
    for (int i = 0; i < len; i++) {
        output[i] = biquad_process_sample(bq, input[i]);
    }
}