#include <stdio.h>
#include <math.h>

#include "audio_config.h"
#include "config_loader.h"
#include "config.h"

#define FLOAT_TOLERANCE 0.0001f

int main(void)
{
    printf("=========================================\n");
    printf("CONFIG LOADER UNIT TEST\n");
    printf("=========================================\n\n");

    audio_config_t config;

    if(config_load(&config) != 0)
    {
        printf("FAIL : Could not load config.bin\n");
        return -1;
    }

    printf("PASS : Configuration loaded successfully\n\n");

    /* ================= ENABLE FLAGS ================= */

    printf("ENABLE_EQ         : %s\n",
           (config.enable_eq == ENABLE_EQ) ? "PASS" : "FAIL");

    printf("ENABLE_DELAY      : %s\n",
           (config.enable_delay == ENABLE_DELAY) ? "PASS" : "FAIL");

    printf("ENABLE_LIMITER    : %s\n",
           (config.enable_limiter == ENABLE_LIMITER) ? "PASS" : "FAIL");

    /* ================= MASTER GAIN ================= */

    printf("GLOBAL_GAIN_DB    : %s\n",
           (fabs(config.global_gain_db - GLOBAL_GAIN_DB) < FLOAT_TOLERANCE)
           ? "PASS" : "FAIL");

    /* ================= EQUALIZER ================= */

    printf("LOW_GAIN_DB       : %s\n",
           (fabs(config.low_gain_db - LOW_GAIN_DB) < FLOAT_TOLERANCE)
           ? "PASS" : "FAIL");

    printf("MID_GAIN_DB       : %s\n",
           (fabs(config.mid_gain_db - MID_GAIN_DB) < FLOAT_TOLERANCE)
           ? "PASS" : "FAIL");

    printf("HIGH_GAIN_DB      : %s\n",
           (fabs(config.high_gain_db - HIGH_GAIN_DB) < FLOAT_TOLERANCE)
           ? "PASS" : "FAIL");

    /* ================= DELAY ================= */

    printf("DELAY_SECONDS     : %s\n",
           (fabs(config.delay_seconds - DELAY_SECONDS) < FLOAT_TOLERANCE)
           ? "PASS" : "FAIL");

    /* ================= LIMITER ================= */

    printf("LIMITER_THRESHOLD : %s\n",
           (fabs(config.limiter_threshold - LIMITER_THRESHOLD) < FLOAT_TOLERANCE)
           ? "PASS" : "FAIL");

    printf("\n=========================================\n");
    printf("LOADER UNIT TEST COMPLETED\n");
    printf("=========================================\n");

    return 0;
}
