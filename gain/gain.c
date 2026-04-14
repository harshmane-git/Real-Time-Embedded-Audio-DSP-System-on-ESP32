#include "gain.h"
#include <math.h>

void apply_gain(float *samples, int len, float gain) {
    if (gain == 1.0f) return;
    for (int i = 0; i < len; i++) {
        samples[i] *= gain;
    }
}

void apply_gain_db(float *samples, int len, float gain_db) {
    float linear = powf(10.0f, gain_db / 20.0f);
    apply_gain(samples, len, linear);
}