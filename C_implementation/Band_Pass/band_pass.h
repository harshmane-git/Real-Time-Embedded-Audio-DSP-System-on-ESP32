#ifndef BAND_PASS_H
#define BAND_PASS_H

#include "biquad.h"

// ───────────────────────────────────────────────
// Coefficients for Band-Pass 800 Hz (4th order)
// fs = 16000 Hz | Bark-Aligned | 4th Order Butterworth
// ───────────────────────────────────────────────
static const float bp_s1[5] = {0.12491506f, 0.00000000f, -0.12491506f, -1.66451047f, 0.75016988f};
static const float bp_s2[5] = {0.05582542f, 0.00000000f, -0.05582542f, -1.79592677f, 0.88834915f};

typedef struct {
    biquad_t bq1;
    biquad_t bq2;
} band_pass_t;

void band_pass_init(band_pass_t *bp);
void band_pass_process_block(band_pass_t *bp, const float *input, float *output, int len);

#endif