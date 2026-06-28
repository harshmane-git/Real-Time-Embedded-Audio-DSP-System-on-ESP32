#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

typedef struct
{
    // ================= ENABLE FLAGS =================

    int enable_eq;
    int enable_delay;
    int enable_limiter;

    // ================= MASTER GAIN =================

    float global_gain_db;

    // ================= EQUALIZER =================

    float low_gain_db;
    float mid_gain_db;
    float high_gain_db;

    // ================= DELAY =================

    float delay_seconds;

    // ================= LIMITER =================

    float limiter_threshold;

} audio_config_t;

#endif
