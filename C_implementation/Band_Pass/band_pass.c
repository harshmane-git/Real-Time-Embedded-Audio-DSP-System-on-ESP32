#include "band_pass.h"

void band_pass_init(band_pass_t *bp) {
    biquad_init(&bp->bq1, bp_s1);
    biquad_init(&bp->bq2, bp_s2);
}

void band_pass_process_block(band_pass_t *bp, const float *input, float *output, int len) {
    float temp[256];
    biquad_process_block(&bp->bq1, input, temp, len);
    biquad_process_block(&bp->bq2, temp, output, len);
}