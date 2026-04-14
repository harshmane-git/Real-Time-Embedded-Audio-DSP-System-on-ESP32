#ifndef GAIN_H
#define GAIN_H

void apply_gain(float *samples, int len, float gain);
void apply_gain_db(float *samples, int len, float gain_db);

#endif