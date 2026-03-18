#ifndef LOW_PASS_H
#define LOW_PASS_H

#include "biquad.h"

// ───────────────────────────────────────────────
// Coefficients for Low-Pass 300 Hz (4th order)
// fs = 16000 Hz | Bark-Aligned | 4th Order Butterworth
// Signal passes: input → s1 → s2 → output
// ───────────────────────────────────────────────
static const float lp_s1[5] = {0.00312629f, 0.00625258f, 0.00312629f, -1.79158896f, 0.80409412f};
static const float lp_s2[5] = {0.00331660f, 0.00663319f, 0.00331660f, -1.90064888f, 0.91391527f};

typedef struct {
    biquad_t bq1;
    biquad_t bq2;
} low_pass_t;

void low_pass_init(low_pass_t *lp);
void low_pass_process_block(low_pass_t *lp, const float *input, float *output, int len);

#endif