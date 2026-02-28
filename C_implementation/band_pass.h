#ifndef BAND_PASS_H
#define BAND_PASS_H

#include "biquad.h"

typedef struct {
    biquad_t bq;   // Single biquad for 2nd-order band-pass
} band_pass_t;

void band_pass_init(band_pass_t *bp);
void band_pass_process_block(band_pass_t *bp, const float *input, float *output, int len);

#endif