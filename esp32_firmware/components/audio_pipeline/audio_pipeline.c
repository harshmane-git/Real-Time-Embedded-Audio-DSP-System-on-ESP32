#include "audio_pipeline.h"
#include "mic.h"
#include "amp.h"
#include "ring_buffer.h"
#include "esp_timer.h"
#include <stdio.h>

#define BLOCK_SIZE RB_SAMPLES_PER_SLOT      // 256

// Ring buffer instance
static ring_buffer_t rb;

// float buffers for mic/amp (their native type)
static float    input_float[BLOCK_SIZE];
static float    output_float[BLOCK_SIZE];

// int32_t buffers for ring buffer (its native type)
static int32_t  input_int[BLOCK_SIZE];
static int32_t  process_int[BLOCK_SIZE];

// Conversion helpers
static void float_to_int32(float *src, int32_t *dst, int len)
{
    for(int i = 0; i < len; i++)
        dst[i] = (int32_t)(src[i] * 2147483647.0f);  // scale to INT32_MAX
}

static void int32_to_float(int32_t *src, float *dst, int len)
{
    for(int i = 0; i < len; i++)
        dst[i] = (float)src[i] / 2147483647.0f;      // normalize back to [-1.0, 1.0]
}

void audio_init(void)
{
    mic_init();
    amp_init();
    rb_init(&rb);
}

void audio_process(void)
{
    // Step 1: Read from mic (float)
    mic_read_block(input_float, BLOCK_SIZE);

    // Step 2: Convert float → int32_t for ring buffer
    float_to_int32(input_float, input_int, BLOCK_SIZE);

    // Step 3: Write into ring buffer
    if(!I2S_write(&rb, input_int, BLOCK_SIZE))
    {
        printf("[audio_pipeline] WARNING: Ring buffer overflow!\n");
        return;
    }

    // Step 4: Wait until write is 2 slots ahead
    if(rb_slots_available(&rb) < 3)
    {
        return;
    }

    int64_t start = esp_timer_get_time();

    // Step 5: Read from ring buffer
    if(!DMA_read(&rb, process_int))
    {
        printf("[audio_pipeline] WARNING: DMA read failed!\n");
        return;
    }

    int64_t end = esp_timer_get_time();
    printf("Block time: %lld us\n", end - start);

    // Step 6: Convert int32_t → float for DSP
    int32_to_float(process_int, output_float, BLOCK_SIZE);

    // DSP hook (team will replace this section)
    for(int i = 0; i < BLOCK_SIZE; i++)
    {
        output_float[i] = output_float[i];   // passthrough for now
    }

    // Step 7: Send to amp (float)
    amp_write_block(output_float, BLOCK_SIZE);
}