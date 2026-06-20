#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include "audio_config.h"

/* Load encrypted config.bin, decrypt it,
 * and populate audio_config_t structure.
 */
int config_load(audio_config_t *config);

#endif
