#ifndef SWITCH_CONTROL_H
#define SWITCH_CONTROL_H

#include "pipeline.h"

void switch_control_init(pipeline_hdl_t hdl);
void switch_control_poll(void);

#endif // SWITCH_CONTROL_H