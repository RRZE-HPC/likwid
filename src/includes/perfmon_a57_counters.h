


#define NUM_COUNTERS_A57 6

static RegisterMap a57_counter_map[NUM_COUNTERS_A57] = {
    {"PMC0", PMC0, PMC, A57_PERFEVTSEL0, A57_PMC0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, A57_PERFEVTSEL1, A57_PMC1, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, A57_PERFEVTSEL2, A57_PMC2, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, A57_PERFEVTSEL3, A57_PMC3, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, A57_PERFEVTSEL4, A57_PMC4, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, A57_PERFEVTSEL5, A57_PMC5, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap a57_box_map[NUM_UNITS] = {
    [PMC] = {A57_PERF_CONTROL_CTRL, A57_OVERFLOW_STATUS, A57_OVERFLOW_FLAGS, 0, 0, 0, 32},
};

static char* a57_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/armv8_pmuv3",
};

static char* a53_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/cpu",
};
