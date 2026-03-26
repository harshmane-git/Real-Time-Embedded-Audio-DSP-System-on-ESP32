#include "rms_error.h"
#include <math.h>

void rms_init(rms_handle_t *h)
{
    h->sum_sq = 0.0;
    h->sum_err = 0.0;
    h->count = 0;
    h->max_err = 0.0;
    h->min_err = 1e9;
}

void rms_process(rms_handle_t *h, double *x, double *y, int N)
{
    for (int i = 0; i < N; i++)
    {
        double diff = x[i] - y[i];
        double abs_diff = fabs(diff);

        h->sum_sq += diff * diff;
        h->sum_err += abs_diff;

        if (abs_diff > h->max_err)
            h->max_err = abs_diff;

        if (abs_diff < h->min_err)
            h->min_err = abs_diff;

        h->count++;
    }
}

double rms_get(rms_handle_t *h)
{
    if (h->count == 0) return -1;
    return sqrt(h->sum_sq / h->count);
}

double rms_avg(rms_handle_t *h)
{
    if (h->count == 0) return -1;
    return h->sum_err / h->count;
}

double rms_max(rms_handle_t *h)
{
    return h->max_err;
}

double rms_min(rms_handle_t *h)
{
    return h->min_err;
}

void rms_reset(rms_handle_t *h)
{
    rms_init(h);
}
