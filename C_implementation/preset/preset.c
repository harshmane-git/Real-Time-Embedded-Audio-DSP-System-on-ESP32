#include "preset.h"
#include <string.h>

const eq_preset_t EQ_FLAT = {0.0f, 0.0f, 0.0f};
const eq_preset_t EQ_BASS_BOOST = {4.0f, 0.0f, -2.0f};
const eq_preset_t EQ_TREBLE_BOOST = {-2.0f, 0.0f, 4.0f};
const eq_preset_t EQ_VOICE_BOOST = {-3.0f, 4.0f, 1.0f};
const eq_preset_t EQ_LOUDNESS = {3.0f, -1.0f, 3.0f};

const eq_preset_t* get_eq_preset(const char* name) {
    if (strcmp(name, "Bass Boost") == 0) return &EQ_BASS_BOOST;
    if (strcmp(name, "Treble Boost") == 0) return &EQ_TREBLE_BOOST;
    if (strcmp(name, "Voice Boost") == 0) return &EQ_VOICE_BOOST;
    if (strcmp(name, "Loudness") == 0) return &EQ_LOUDNESS;
    return &EQ_FLAT;  // default
}