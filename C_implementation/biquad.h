#ifndef BIQUAD_H
#define BIQUAD_H

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float w1, w2;
} biquad_t;

void biquad_init(biquad_t *bq, const float coeffs[5]);
float biquad_process_sample(biquad_t *bq, float input);
void biquad_process_block(biquad_t *bq, const float *input, float *output, int len);

#endif