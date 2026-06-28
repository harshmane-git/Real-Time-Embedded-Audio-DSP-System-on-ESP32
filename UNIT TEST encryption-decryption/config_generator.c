#include <stdio.h>
#include <stdint.h>

#include "audio_config.h"
#include "config.h"
#include "config_generator.h"

/* =====================================================
 * 16-bit XOR Encryption
 * ===================================================== */

void encrypt_data(uint16_t *data, size_t num_words)
{
    uint16_t key = 0xA5F3;

    for(size_t i = 0; i < num_words; i++)
    {
        data[i] ^= key;
    }
}

/* =====================================================
 * Generate Encrypted Configuration File
 * ===================================================== */

int generate_config(void)
{
    audio_config_t config;

    /* ================= ENABLE FLAGS ================= */

    config.enable_eq       = ENABLE_EQ;
    config.enable_delay    = ENABLE_DELAY;
    config.enable_limiter  = ENABLE_LIMITER;

    /* ================= MASTER GAIN ================= */

    config.global_gain_db = GLOBAL_GAIN_DB;

    /* ================= EQUALIZER ================= */

    config.low_gain_db  = LOW_GAIN_DB;
    config.mid_gain_db  = MID_GAIN_DB;
    config.high_gain_db = HIGH_GAIN_DB;

    /* ================= DELAY ================= */

    config.delay_seconds = DELAY_SECONDS;

    /* ================= LIMITER ================= */

    config.limiter_threshold = LIMITER_THRESHOLD;

    /* Encrypt the structure */

    encrypt_data((uint16_t *)&config,
                 sizeof(audio_config_t) / sizeof(uint16_t));

    /* Write encrypted data */

    FILE *fp = fopen("config.bin", "wb");

    if(fp == NULL)
    {
        return -1;
    }

    fwrite(&config,
           sizeof(audio_config_t),
           1,
           fp);

    fclose(fp);

    return 0;
}
