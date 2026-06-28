#include <stdio.h>
#include <math.h>

#include "audio_config.h"
#include "config_loader.h"
#include "config.h"

#define FLOAT_TOLERANCE 0.0001f

/* Function from config_generator.c */
int generate_config(void);

int main(void)
{
    printf("=========================================\n");
    printf("END TO END UNIT TEST\n");
    printf("=========================================\n\n");

    /* Step 1 : Generate encrypted configuration */

    if(generate_config() != 0)
    {
        printf("FAIL : Could not generate config.bin\n");
        return -1;
    }

    printf("PASS : config.bin generated\n");

    /* Step 2 : Load and decrypt configuration */

    audio_config_t config;

    if(config_load(&config) != 0)
    {
        printf("FAIL : Could not load config.bin\n");
        return -1;
    }

    printf("PASS : Configuration loaded\n\n");

    int all_pass = 1;

    /* ================= ENABLE FLAGS ================= */

    if(config.enable_eq != ENABLE_EQ)
    {
        printf("FAIL : enable_eq\n");
        all_pass = 0;
    }

    if(config.enable_delay != ENABLE_DELAY)
    {
        printf("FAIL : enable_delay\n");
        all_pass = 0;
    }

    if(config.enable_limiter != ENABLE_LIMITER)
    {
        printf("FAIL : enable_limiter\n");
        all_pass = 0;
    }

    /* ================= MASTER GAIN ================= */

    if(fabs(config.global_gain_db - GLOBAL_GAIN_DB) > FLOAT_TOLERANCE)
    {
        printf("FAIL : global_gain_db\n");
        all_pass = 0;
    }

    /* ================= EQUALIZER ================= */

    if(fabs(config.low_gain_db - LOW_GAIN_DB) > FLOAT_TOLERANCE)
    {
        printf("FAIL : low_gain_db\n");
        all_pass = 0;
    }

    if(fabs(config.mid_gain_db - MID_GAIN_DB) > FLOAT_TOLERANCE)
    {
        printf("FAIL : mid_gain_db\n");
        all_pass = 0;
    }

    if(fabs(config.high_gain_db - HIGH_GAIN_DB) > FLOAT_TOLERANCE)
    {
        printf("FAIL : high_gain_db\n");
        all_pass = 0;
    }

    /* ================= DELAY ================= */

    if(fabs(config.delay_seconds - DELAY_SECONDS) > FLOAT_TOLERANCE)
    {
        printf("FAIL : delay_seconds\n");
        all_pass = 0;
    }

    /* ================= LIMITER ================= */

    if(fabs(config.limiter_threshold - LIMITER_THRESHOLD) > FLOAT_TOLERANCE)
    {
        printf("FAIL : limiter_threshold\n");
        all_pass = 0;
    }

    printf("\n");

    if(all_pass)
    {
        printf("=========================================\n");
        printf("ALL END TO END TESTS PASSED\n");
        printf("=========================================\n");
    }
    else
    {
        printf("=========================================\n");
        printf("END TO END TEST FAILED\n");
        printf("=========================================\n");
    }

    return 0;
}
