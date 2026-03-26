#ifndef RMS_H
#define RMS_H

typedef struct {
    double sum_sq;
    double max_err;
    double min_err;
    double sum_err;
    int count;
} rms_handle_t;

void rms_init(rms_handle_t *h);
void rms_process(rms_handle_t *h, double *x, double *y, int N);
double rms_get(rms_handle_t *h);
double rms_avg(rms_handle_t *h);
double rms_max(rms_handle_t *h);
double rms_min(rms_handle_t *h);
void rms_reset(rms_handle_t *h);

#endif
