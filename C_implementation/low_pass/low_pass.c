#include "low_pass.h"

void low_pass_init(low_pass_t *lp) {
    biquad_init(&lp->bq1, lp_s1);
    biquad_init(&lp->bq2, lp_s2);
}

void low_pass_process_block(low_pass_t *lp, const float *input, float *output, int len) {
    float temp[256];
    biquad_process_block(&lp->bq1, input, temp, len);
    biquad_process_block(&lp->bq2, temp, output, len);
}