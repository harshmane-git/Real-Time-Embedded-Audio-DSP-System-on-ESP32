#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rms_error.h"

#define FRAME_SIZE 256 //samples processed at a time

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s file1.csv file2.csv\n", argv[0]);
        return 1;
    }

    FILE *f1 = fopen(argv[1], "r");
    FILE *f2 = fopen(argv[2], "r");

    if (!f1 || !f2)
    {
        printf("Error opening files\n");
        return 1;
    }

    rms_handle_t h;
    rms_init(&h);

    double x[FRAME_SIZE];
    double y[FRAME_SIZE];

    int frame_count = 0;

    double max_frame_rms = 0.0;
    int worst_frame = -1;

    while (1)
    {
        int i;

        for (i = 0; i < FRAME_SIZE; i++)
        {
            if (fscanf(f1, "%lf", &x[i]) != 1 ||
                fscanf(f2, "%lf", &y[i]) != 1)
            {
                break;
            }
        }

        if (i == 0)
            break;

        // Global RMS accumulation
        rms_process(&h, x, y, i);

        // Frame RMS calculation
        double frame_sum_sq = 0.0;

        for (int j = 0; j < i; j++)
        {
            double diff = x[j] - y[j];
            frame_sum_sq += diff * diff;
        }

        double frame_rms = sqrt(frame_sum_sq / i);

        printf("Frame %d (samples %d–%d) RMS: %e\n",
               frame_count,
               frame_count * FRAME_SIZE,
               frame_count * FRAME_SIZE + i - 1,
               frame_rms);

        // 🔹 Track worst frame
        if (frame_rms > max_frame_rms)
        {
            max_frame_rms = frame_rms;
            worst_frame = frame_count;
        }

        frame_count++;
    }

    fclose(f1);
    fclose(f2);

    //  Final global metrics
    double rms = rms_get(&h);

    printf("\n========== FINAL REPORT ==========\n");
    printf("Total Frames: %d\n", frame_count);
    printf("RMS Error: %e\n", rms);
    printf("Avg Error: %e\n", rms_avg(&h));
    printf("Max Error: %e\n", rms_max(&h));
    printf("Min Error: %e\n", rms_min(&h));

    printf("\nWorst Frame: %d\n", worst_frame);
    printf("Worst Frame RMS: %e\n", max_frame_rms);

    // Pass/Fail
    if (rms < 1e-6)
        printf("✅ PASS (Perfect match)\n");
    else if (rms < 1e-5)
        printf("⚠️ PASS (Minor differences)\n");
    else
        printf("❌ FAIL (Significant mismatch)\n");

    return 0;
}
