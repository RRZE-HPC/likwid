

#define NUM_COUNTERS_POWER8 7

static RegisterMap power8_counter_map[NUM_COUNTERS_POWER8] = {
    {"PMC0", PMC0, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PURR", PMC6, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap power8_box_map[NUM_UNITS] = {
    [PMC] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
};

static char* power8_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/cpu",
};
