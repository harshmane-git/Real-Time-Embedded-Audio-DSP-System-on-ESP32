#include "audio_pipeline.h"
#include "mic.h"
#include "ring_buffer.h"
#include "amp.h"
#include <stdio.h>

#define BLOCK_SIZE 256

static float input_block[BLOCK_SIZE];
static float process_block[BLOCK_SIZE];

void audio_init(void)
{
    mic_init();
    rb_init();
    amp_init();
}

void audio_process(void)
{
    // 1. Generate sine (mic simulation)
    mic_process(input_block, BLOCK_SIZE);

    // 2. Push into ring buffer
    rb_write(input_block, BLOCK_SIZE);


    // 3. If enough samples → process
    if(rb_read(process_block, BLOCK_SIZE) == 0)
    {
        printf("Block processed\n");
        // 4. Send to speaker
        amp_write_block(process_block, BLOCK_SIZE);

    }
}