#ifndef PRESET_H
#define PRESET_H


// EQ Presets
typedef struct {
    float low_db;
    float mid_db;
    float high_db;
} eq_preset_t;

extern const eq_preset_t EQ_FLAT;
extern const eq_preset_t EQ_BASS_BOOST;
extern const eq_preset_t EQ_TREBLE_BOOST;
extern const eq_preset_t EQ_VOICE_BOOST;
extern const eq_preset_t EQ_LOUDNESS;

// Get preset by name (simple string matching)
const eq_preset_t* get_eq_preset(const char* name);

#endif