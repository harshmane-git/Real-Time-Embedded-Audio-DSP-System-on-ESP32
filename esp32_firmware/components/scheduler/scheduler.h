#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "ring_buffer.h"

#define READ_DELAY_SLOTS  2
#define READ_TRIGGER      (READ_DELAY_SLOTS + 1)

void scheduler_task(void *arg);

#endif