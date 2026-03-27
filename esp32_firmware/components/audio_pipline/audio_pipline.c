#include "audio_pipeline.h"
#include "mic.h"
#include "amp.h"
#include "ring_buffer.h"
#include "esp_timer.h"
#include <stdio.h>

#define BLOCK_SIZE 256

static float input_block[BLOCK_SIZE];
static float process_block[BLOCK_SIZE];

void audio_init(void)
{
    mic_init();
    amp_init();
    rb_init();
}

void audio_process(void)
{
    // Step 1: Get audio block (simulated or real)
    mic_read_block(input_block, BLOCK_SIZE);

    // Step 2: Write to ring buffer
    rb_write(input_block, BLOCK_SIZE);

    // Step 3: Process only full block → ensures real-time safety
    if(rb_available() >= BLOCK_SIZE)
    {
        int64_t start = esp_timer_get_time();

        rb_read(process_block, BLOCK_SIZE);

        int64_t end = esp_timer_get_time();

        // Debug only (disable in final real-time system)
        printf("Block time: %lld us\n", end - start);

        // DSP hook (team will replace this section)
        for(int i = 0; i < BLOCK_SIZE; i++)
        {
            process_block[i] = process_block[i];
        }

        // Step 4: Send to output
        amp_write_block(process_block, BLOCK_SIZE);
    }
}
