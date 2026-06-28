#include <stdio.h>
#include <stdint.h>

#include "audio_config.h"
#include "config_loader.h"

/* =====================================================
 * 16-bit XOR Decryption
 * ===================================================== */

void decrypt_data(uint16_t *data, size_t num_words)
{
    uint16_t key = 0xA5F3;

    for(size_t i = 0; i < num_words; i++)
    {
        data[i] ^= key;
    }
}

/* =====================================================
 * Load + Decrypt Configuration
 * ===================================================== */

int config_load(audio_config_t *config)
{
    FILE *fp = fopen("config.bin", "rb");

    if(fp == NULL)
    {
        return -1;
    }

    fread(config,
          sizeof(audio_config_t),
          1,
          fp);

    fclose(fp);

    decrypt_data((uint16_t *)config,
                 sizeof(audio_config_t) / sizeof(uint16_t));

    return 0;
}
