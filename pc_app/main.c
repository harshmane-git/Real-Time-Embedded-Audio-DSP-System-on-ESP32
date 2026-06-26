#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sndfile.h>

// Core type systems from your implementation team
#include "dsp_types.h"
#include "common_types.h"
#include "audio_config.h" 

// Component Headers
#include "biquad.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"
#include "equalizer.h"
#include "gain.h"
#include "delay.h"

#define BLOCK_SIZE 256

// Enforce File Scope Rule for all Config Structs to avoid stack exhaustion
static gain_config_t     g_gain_config;
static equalizer_config_t g_eq_config;
static delay_config_t     g_delay_config;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.wav> <output.wav>\n", argv[0]);
        return 1;
    }

    SNDFILE *infile = NULL, *outfile = NULL;
    SF_INFO sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));

    // Open Input File
    infile = sf_open(argv[1], SFM_READ, &sfinfo);
    if (!infile) {
        printf("Error: Could not open input file '%s'\n", argv[1]);
        return 1;
    }

    if (sfinfo.samplerate != 16000) {
        printf("Warning: Sample rate is %d Hz. Pipeline expects 16000 Hz!\n", sfinfo.samplerate);
    }
    if (sfinfo.channels != 1) {
        printf("Error: Input file must be mono (1 channel).\n");
        sf_close(infile);
        return 1;
    }

    // Open Output File
    outfile = sf_open(argv[2], SFM_WRITE, &sfinfo);
    if (!outfile) {
        printf("Error: Could not open output file '%s'\n", argv[2]);
        sf_close(infile);
        return 1;
    }

    // 3. CORRECTED TYPE DECLARATIONS FROM COMPILER PROMPTS
    gain_hdl_t      gain_hdl;
    equalizer_hdl_t eq_hdl;      // Fixed from equalizer_t to equalizer_hdl_t
    delay_hdl_t     delay_hdl;

    // 4. INITIALIZATION ROUTINES PASSED WITH CONFIG OBJECT POINTERS
    gain_init(&gain_hdl, &g_gain_config);
    equalizer_init(&eq_hdl, &g_eq_config); 
    delay_init(&delay_hdl, &g_delay_config);

    float input_buffer[BLOCK_SIZE];
    float processing_buffer[BLOCK_SIZE];
    sf_count_t read_count;
    long total_blocks_processed = 0;

    printf("Processing audio data streams via zero-copy pointer chain...\n");

    // 5. Execution Processing Loop
    while ((read_count = sf_readf_float(infile, input_buffer, BLOCK_SIZE)) > 0) {
        if (read_count < BLOCK_SIZE) {
            memset(input_buffer + read_count, 0, (BLOCK_SIZE - read_count) * sizeof(float));
        }

        memcpy(processing_buffer, input_buffer, sizeof(input_buffer));
        float *dsp_out = processing_buffer;

        // Allocate temporary cross-over bands to receive split frequencies
        float low_band[BLOCK_SIZE];
        float mid_band[BLOCK_SIZE];
        float high_band[BLOCK_SIZE];

        // --- Execute Pipeline Chain ---
        
        // 1. Equalizer splits into Low, Mid, and High arrays
        equalizer_process(&eq_hdl, dsp_out, low_band, mid_band, high_band, BLOCK_SIZE);

        // 2. Sum the bands back together into our pointer destination
        for (int i = 0; i < BLOCK_SIZE; i++) {
            dsp_out[i] = low_band[i] + mid_band[i] + high_band[i];
        }

        // 3. Pass the combined audio stream to the next downstream processing blocks
        gain_process(&gain_hdl, dsp_out, dsp_out, BLOCK_SIZE);
        delay_process(&delay_hdl, dsp_out, dsp_out, BLOCK_SIZE);

        sf_writef_float(outfile, dsp_out, BLOCK_SIZE);
        total_blocks_processed++;
    }

    sf_close(infile);
    sf_close(outfile);

    printf("Success! Processed %ld blocks cleanly. Saved to: %s\n", total_blocks_processed, argv[2]);
    return 0;
}