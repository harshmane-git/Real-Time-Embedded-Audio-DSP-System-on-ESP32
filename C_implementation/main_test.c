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
#include "equalizer.h"

#define BLOCK_SIZE 256

int main(void)
{
    const char *input_file =
        "C:/Users/Umesh/Downloads/Recording.wav";

    const char *output_file =
        "C:/Users/Umesh/Downloads/final_test1.wav";

    /* ====================== CONFIGURATION ====================== */

    bool enable_bypass = false;
    bool enable_mute   = false;

    bool  enable_gain        = false;
    float global_gain_linear = 2.0f;

    bool enable_eq = true;

    bool  enable_delay  = true;
    float delay_seconds = 0.25f;

    bool  enable_limiter    = false;
    float limiter_threshold = 0.9f;

    /* EQ gains now handled by gain block */
    eq_gain_config_t eq_gain_cfg =
    {
        .low_gain_db  = -2.0f,
        .mid_gain_db  = 3.0f,
        .high_gain_db = 1.0f
    };

    /* ====================== COEFFICIENTS ====================== */

    low_pass_config_t lp_config = {
        .s1 = {0.00312629f,0.00625258f,0.00312629f,-1.79158896f,0.80409412f},
        .s2 = {0.00331660f,0.00663319f,0.00331660f,-1.90064888f,0.91391527f}
    };

    band_pass_config_t bp_config = {
        .s1 = {0.12491506f,0.00000000f,-0.12491506f,-1.66451047f,0.75016988f},
        .s2 = {0.05582542f,0.00000000f,-0.05582542f,-1.79592677f,0.88834915f}
    };

    high_pass_config_t hp_config = {
        .s1 = {0.51627979f,-1.03255959f,0.51627979f,-0.85540037f,0.20971880f},
        .s2 = {0.67177700f,-1.34355400f,0.67177700f,-1.11303657f,0.57407142f}
    };

    equalizer_config_t eq_config =
    {
        .low  = lp_config,
        .mid  = bp_config,
        .high = hp_config
    };

    gain_config_t gain_config = {
        .gain_db = 6.0f
    };

    delay_config_t delay_config = {
        .delay_seconds = delay_seconds
    };

    limiter_config_t limiter_config = {
        .fThreshold = limiter_threshold
    };

    /* ====================== HANDLES ====================== */

    low_pass_hdl_t   lp_hdl;
    band_pass_hdl_t  bp_hdl;
    high_pass_hdl_t  hp_hdl;
    gain_hdl_t       gain_hdl;
    delay_hdl_t      delay_hdl;
    limiter_hdl_t    limiter_hdl;

    equalizer_hdl_t  eq_hdl;
    eq_gain_hdl_t    eq_gain_hdl;

    uint32_t size;
    STATUS st_lp, st_bp, st_hp, st_gn, st_dl, st_lm;

    /* ====================== OPEN + INIT ====================== */

    low_pass_open(&size);
    st_lp = low_pass_init(&lp_hdl, &lp_config);

    band_pass_open(&size);
    st_bp = band_pass_init(&bp_hdl, &bp_config);

    high_pass_open(&size);
    st_hp = high_pass_init(&hp_hdl, &hp_config);

    gain_open(&size);
    st_gn = gain_init(&gain_hdl, &gain_config);

    delay_open(&size);
    st_dl = delay_init(&delay_hdl, &delay_config);

    limiter_open(&size);
    st_lm = limiter_init(&limiter_hdl, &limiter_config);

    equalizer_open(&size);
    equalizer_init(&eq_hdl, &eq_config);

    eq_gain_open(&size);
    eq_gain_init(&eq_gain_hdl, &eq_gain_cfg);

    /* ====================== STATUS ====================== */

    printf("\n=== Module Initialization Status ===\n");

    printf("Low Pass   : %s\n",
           (st_lp == STATUS_OK) ? "SUCCESS" : "FAILED");

    printf("Band Pass  : %s\n",
           (st_bp == STATUS_OK) ? "SUCCESS" : "FAILED");

    printf("High Pass  : %s\n",
           (st_hp == STATUS_OK) ? "SUCCESS" : "FAILED");

    printf("Gain       : %s\n",
           (st_gn == STATUS_OK) ? "SUCCESS" : "FAILED");

    printf("Delay      : %s\n",
           (st_dl == STATUS_OK) ? "SUCCESS" : "FAILED");

    printf("Limiter    : %s\n",
           (st_lm == STATUS_OK) ? "SUCCESS" : "FAILED");

    printf("====================================\n\n");

    /* ====================== FILE OPEN ====================== */

    SF_INFO sfinfo = {0};

    SNDFILE *infile =
        sf_open(input_file, SFM_READ, &sfinfo);

    if (infile == NULL)
    {
        printf("Input file open failed\n");
        return 1;
    }

    SNDFILE *outfile =
        sf_open(output_file, SFM_WRITE, &sfinfo);

    if (outfile == NULL)
    {
        printf("Output file open failed\n");
        sf_close(infile);
        return 1;
    }

    /* ====================== BUFFERS ====================== */

    float buffer[BLOCK_SIZE];
    float temp[BLOCK_SIZE];
    float low_out[BLOCK_SIZE];
    float mid_out[BLOCK_SIZE];
    float high_out[BLOCK_SIZE];
    float final_out[BLOCK_SIZE];

    sf_count_t read_count;

    /* ====================== PROCESS LOOP ====================== */

    while ((read_count =
        sf_read_float(infile,
                      buffer,
                      BLOCK_SIZE)) > 0)
    {
        if (enable_bypass)
        {
            for (uint32_t i = 0; i < read_count; i++)
                final_out[i] = buffer[i];
        }
        else if (enable_mute)
        {
            for (uint32_t i = 0; i < read_count; i++)
                final_out[i] = 0.0f;
        }
        else
        {
            for (uint32_t i = 0; i < read_count; i++)
                temp[i] = buffer[i];

            if (enable_eq)
            {
                equalizer_process(&eq_hdl,
                                  temp,
                                  low_out,
                                  mid_out,
                                  high_out,
                                  (uint32_t)read_count);

                gain_process(&eq_gain_hdl.low,
                             low_out,
                             low_out,
                             (uint32_t)read_count);

                gain_process(&eq_gain_hdl.mid,
                             mid_out,
                             mid_out,
                             (uint32_t)read_count);

                gain_process(&eq_gain_hdl.high,
                             high_out,
                             high_out,
                             (uint32_t)read_count);

                for (uint32_t i = 0; i < read_count; i++)
                {
                    temp[i] =
                        low_out[i] +
                        mid_out[i] +
                        high_out[i];
                }
            }

            if (enable_delay)
            {
                delay_process(&delay_hdl,
                              temp,
                              temp,
                              (uint32_t)read_count);
            }

            if (enable_limiter)
            {
                limiter_process(&limiter_hdl,
                                temp,
                                temp,
                                (uint32_t)read_count);
            }

            if (enable_gain)
            {
                gain_process(&gain_hdl,
                             temp,
                             final_out,
                             (uint32_t)read_count);
            }
            else
            {
                for (uint32_t i = 0; i < read_count; i++)
                    final_out[i] = temp[i];
            }
        }

        sf_write_float(outfile,
                       final_out,
                       read_count);
    }

    /* ====================== DELAY TAIL ====================== */

    if (enable_delay)
    {
        uint32_t remain =
            delay_hdl.delay_samples;

        while (remain > 0)
        {
            uint32_t block =
                (remain > BLOCK_SIZE) ?
                BLOCK_SIZE :
                remain;

            delay_process(&delay_hdl,
                          NULL,
                          final_out,
                          block);

            sf_write_float(outfile,
                           final_out,
                           block);

            remain -= block;
        }
    }

    /* ====================== CLOSE ====================== */

    low_pass_close(&lp_hdl);
    band_pass_close(&bp_hdl);
    high_pass_close(&hp_hdl);
    gain_close(&gain_hdl);
    delay_close(&delay_hdl);
    limiter_close(&limiter_hdl);

    equalizer_close(&eq_hdl);
    eq_gain_close(&eq_gain_hdl);

    sf_close(infile);
    sf_close(outfile);

    printf("Processing Complete! Output saved as: %s\n",
           output_file);

    return 0;
}