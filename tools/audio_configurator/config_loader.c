#include <stdio.h>
#include <stdint.h>

#include "audio_config.h"

void decrypt_data(uint8_t *data, size_t size)
{
    uint8_t key = 0xAA;

    for (size_t i = 0; i < size; i++)
    {
        data[i] ^= key;
    }
}

int main()
{
    FILE *fp = fopen("config.bin", "rb");

    if (fp == NULL)
    {
        printf("Could not open config.bin\n");
        return -1;
    }

    printf("config.bin opened successfully\n");

    audio_config_t config;

    fread(&config,
          sizeof(audio_config_t),
          1,
          fp);

    printf("Encrypted bytes loaded into RAM\n");

    fclose(fp);

    decrypt_data((uint8_t *)&config,
             sizeof(audio_config_t));

printf("Configuration decrypted successfully\n");

    printf("\n===== RECOVERED CONFIGURATION =====\n");

printf("ENABLE_EQ         : %d\n", config.enable_eq);
printf("ENABLE_DELAY      : %d\n", config.enable_delay);
printf("ENABLE_LIMITER    : %d\n", config.enable_limiter);

printf("GLOBAL_GAIN_DB    : %.2f\n",
       config.global_gain_db);

printf("LOW_GAIN_DB       : %.2f\n",
       config.low_gain_db);

printf("MID_GAIN_DB       : %.2f\n",
       config.mid_gain_db);

printf("HIGH_GAIN_DB      : %.2f\n",
       config.high_gain_db);

printf("DELAY_SECONDS     : %.3f\n",
       config.delay_seconds);

printf("LIMITER_THRESHOLD : %.2f\n",
       config.limiter_threshold);

    return 0;
}
