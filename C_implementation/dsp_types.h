#ifndef DSP_TYPES_H
#define DSP_TYPES_H

#include <stdint.h>
#include <stdio.h>

// Q15 fixed-point format (1 sign bit + 15 fractional bits)
#define Q15_SHIFT 15
#define Q15_MAX   32767
#define Q15_MIN  -32768
#define Q15_ONE  (1 << Q15_SHIFT)  // represents 1.0 in Q15

// Safe saturated multiply: Q15 × Q15 → Q15
static inline int16_t q15_mult_sat(int16_t a, int16_t b) {
    int32_t prod = (int32_t)a * b;
    prod >>= Q15_SHIFT;
    if (prod > Q15_MAX) return Q15_MAX;
    if (prod < Q15_MIN) return Q15_MIN;
    return (int16_t)prod;
}

// Safe saturated add
static inline int16_t q15_add_sat(int16_t a, int16_t b) {
    int32_t sum = (int32_t)a + b;
    if (sum > Q15_MAX) return Q15_MAX;
    if (sum < Q15_MIN) return Q15_MIN;
    return (int16_t)sum;
}

#endif // DSP_TYPES_H
