#include <stdio.h>
#include <stdint.h>

#include "config.h"
#include "audio_config.h"

// =====================================================
// SIMPLE XOR ENCRYPTION
// =====================================================

void encrypt_data(uint16_t *data, size_t num_words)
{
    uint16_t key = 0xA5F3;

    for (size_t i = 0; i < num_words; i++)
    {
        data[i] ^= key;
    }
}

// =====================================================
// MAIN
// =====================================================

int main()
{
    // Create config structure
    audio_config_t config;

    // ================= ENABLE FLAGS =================

    config.enable_eq       = ENABLE_EQ;
    config.enable_delay    = ENABLE_DELAY;
    config.enable_limiter  = ENABLE_LIMITER;

    // ================= MASTER GAIN =================

    config.global_gain_db  = GLOBAL_GAIN_DB;

    // ================= EQUALIZER =================

    config.low_gain_db     = LOW_GAIN_DB;
    config.mid_gain_db     = MID_GAIN_DB;
    config.high_gain_db    = HIGH_GAIN_DB;

    // ================= DELAY =================

    config.delay_seconds   = DELAY_SECONDS;

    // ================= LIMITER =================

    config.limiter_threshold = LIMITER_THRESHOLD;

    // =====================================================
    // ENCRYPT CONFIG DATA
    // =====================================================

    encrypt_data((uint16_t *)&config,
             sizeof(audio_config_t) / sizeof(uint16_t));

    // =====================================================
    // CREATE BINARY FILE
    // =====================================================

    FILE *fp = fopen("config.bin", "wb");

    if (fp == NULL)
    {
        printf("Error creating config.bin\n");
        return -1;
    }

    fwrite(&config,
           sizeof(audio_config_t),
           1,
           fp);

    fclose(fp);

    printf("Encrypted config.bin generated successfully!\n");

    return 0;
}
