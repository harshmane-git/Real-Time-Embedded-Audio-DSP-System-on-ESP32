#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <stdint.h>

/* Status return type used by all modules */
typedef enum {
    STATUS_OK = 0,      // Success
    STATUS_NOT_OK = 1   // Failure / Error
} STATUS;

#endif /* COMMON_TYPES_H */