#ifndef HIGH_PASS_H
#define HIGH_PASS_H

#include "biquad.h"

typedef struct {
    biquad_t bq;   // Single biquad for 2nd-order high-pass
} high_pass_t;

void high_pass_init(high_pass_t *hp);
void high_pass_process_block(high_pass_t *hp, const float *input, float *output, int len);

#endif