#include "gain_module.h"
#include "gain.h"

void apply_eq_gains(
    const float *low_in, float *low_out,
    const float *mid_in, float *mid_out,
    const float *high_in, float *high_out,
    int len,
    const gain_config_t *config
) {
    // Copy inputs to outputs
    for (int i = 0; i < len; i++) {
        low_out[i] = low_in[i];
        mid_out[i] = mid_in[i];
        high_out[i] = high_in[i];
    }

    // Apply per-band gain
    apply_gain_db(low_out, len, config->low_db);
    apply_gain_db(mid_out, len, config->mid_db);
    apply_gain_db(high_out, len, config->high_db);

    // Simple mixer: sum the three bands
    for (int i = 0; i < len; i++) {
        low_out[i] = low_out[i] + mid_out[i] + high_out[i];
    }
}