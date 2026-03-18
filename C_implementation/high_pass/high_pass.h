#ifndef HIGH_PASS_H
#define HIGH_PASS_H

#include "biquad.h"

// ───────────────────────────────────────────────
// Coefficients for High-Pass 2000 Hz (4th order)
// fs = 16000 Hz | Bark-Aligned | 4th Order Butterworth
// ───────────────────────────────────────────────
static const float hp_s1[5] = {0.51627979f, -1.03255959f, 0.51627979f, -0.85540037f, 0.20971880f};
static const float hp_s2[5] = {0.67177700f, -1.34355400f, 0.67177700f, -1.11303657f, 0.57407142f};

typedef struct {
    biquad_t bq1;
    biquad_t bq2;
} high_pass_t;

void high_pass_init(high_pass_t *hp);
void high_pass_process_block(high_pass_t *hp, const float *input, float *output, int len);

#endif