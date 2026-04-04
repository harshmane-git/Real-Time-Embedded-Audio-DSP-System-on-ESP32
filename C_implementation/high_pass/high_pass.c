#include "high_pass.h"

void high_pass_init(high_pass_t *hp) {
    biquad_init(&hp->bq1, hp_s1);
    biquad_init(&hp->bq2, hp_s2);
}

void high_pass_process_block(high_pass_t *hp, const float *input, float *output, int len) {
    float temp[4096];
    biquad_process_block(&hp->bq1, input, temp, len);
    biquad_process_block(&hp->bq2, temp, output, len);
}