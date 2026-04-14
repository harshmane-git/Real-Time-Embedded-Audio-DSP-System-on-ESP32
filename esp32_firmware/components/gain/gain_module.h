#ifndef GAIN_MODULE_H
#define GAIN_MODULE_H

// Per-band gain only (low / mid / high)
typedef struct {
    float low_db;
    float mid_db;
    float high_db;
} gain_config_t;

// Apply per-band gains and mix the three bands
void apply_eq_gains(
    const float *low_in, float *low_out,
    const float *mid_in, float *mid_out,
    const float *high_in, float *high_out,
    int len,
    const gain_config_t *config
);
                    
#endif