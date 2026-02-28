#ifndef LOW_PASS_H
#define LOW_PASS_H

#include "biquad.h"

typedef struct {
    biquad_t bq;   // Single biquad for 2nd-order low-pass
} low_pass_t;

void low_pass_init(low_pass_t *lp);
void low_pass_process_block(low_pass_t *lp, const float *input, float *output, int len);

#endif