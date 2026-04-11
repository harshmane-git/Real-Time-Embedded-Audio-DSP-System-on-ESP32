#include <stdio.h>
#include <sndfile.h>
#include <stdbool.h>
#include "common_types.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"
#include "gain.h"
#include "delay.h"
#include "limiter.h"

#define BLOCK_SIZE 256

int main(void) {
    const char* input_file = "C:/Users/Umesh/Downloads/Recording.wav";
    const char* output_file = "delay.wav";

    // ====================== TEST CONFIGURATION FOR DELAY ======================
    bool enable_bypass = false;
    bool enable_mute = false;

    bool  enable_gain = false;
    float global_gain_linear = 1.0f;

    bool enable_eq = false;      // Disabled for clean delay test

    bool  enable_delay = true;
    float delay_seconds = 0.25f;      // ? Change this (0.1, 0.2, 0.3, 0.4)

    bool  enable_limiter = false;
    float limiter_threshold = 0.9f;

    // ====================== COEFFICIENTS ======================
    low_pass_config_t lp_config = { .s1 = {0.00312629f, 0.00625258f, 0.00312629f, -1.79158896f, 0.80409412f},
                                    .s2 = {0.00331660f, 0.00663319f, 0.00331660f, -1.90064888f, 0.91391527f} };

    band_pass_config_t bp_config = { .s1 = {0.12491506f, 0.00000000f, -0.12491506f, -1.66451047f, 0.75016988f},
                                     .s2 = {0.05582542f, 0.00000000f, -0.05582542f, -1.79592677f, 0.88834915f} };

    high_pass_config_t hp_config = { .s1 = {0.51627979f, -1.03255959f, 0.51627979f, -0.85540037f, 0.20971880f},
                                     .s2 = {0.67177700f, -1.34355400f, 0.67177700f, -1.11303657f, 0.57407142f} };

    gain_config_t gain_config = { .gain_linear = global_gain_linear };
    delay_config_t delay_config = { .delay_seconds = delay_seconds };
    limiter_config_t limiter_config = { .fThreshold = limiter_threshold };

    // ... (rest of the code remains same as before - handles, processing, etc.)

    // At the end, print clear status
    printf("\n=== Delay Test Status ===\n");
    printf("Delay           : %s (%.2f seconds)\n", enable_delay ? "ENABLED" : "DISABLED", delay_seconds);
    printf("EQ              : %s\n", enable_eq ? "ENABLED" : "DISABLED");
    printf("Gain            : %s\n", enable_gain ? "ENABLED" : "DISABLED");
    printf("Limiter         : %s\n", enable_limiter ? "ENABLED" : "DISABLED");
    printf("Output saved as : %s\n", output_file);
    printf("Processing Complete!\n");

    return 0;
}