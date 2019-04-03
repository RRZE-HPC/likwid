

#define NUM_COUNTERS_POWER9 30

static RegisterMap power9_counter_map[NUM_COUNTERS_POWER9] = {
    {"PMC0", PMC0, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX0C0", PMC6, MBOX0},
    {"MBOX0C1", PMC7, MBOX0},
    {"MBOX0C2", PMC8, MBOX0},
    {"MBOX1C0", PMC9, MBOX0},
    {"MBOX1C1", PMC10, MBOX0},
    {"MBOX1C2", PMC11, MBOX0},
    {"MBOX2C0", PMC12, MBOX0},
    {"MBOX2C1", PMC13, MBOX0},
    {"MBOX2C2", PMC14, MBOX0},
    {"MBOX3C0", PMC15, MBOX0},
    {"MBOX3C1", PMC16, MBOX0},
    {"MBOX3C2", PMC17, MBOX0},
    {"MBOX4C0", PMC18, MBOX0},
    {"MBOX4C1", PMC19, MBOX0},
    {"MBOX4C2", PMC20, MBOX0},
    {"MBOX5C0", PMC21, MBOX0},
    {"MBOX5C1", PMC22, MBOX0},
    {"MBOX5C2", PMC23, MBOX0},
    {"MBOX6C0", PMC24, MBOX0},
    {"MBOX6C1", PMC25, MBOX0},
    {"MBOX6C2", PMC26, MBOX0},
    {"MBOX7C0", PMC27, MBOX0},
    {"MBOX7C1", PMC28, MBOX0},
    {"MBOX7C2", PMC29, MBOX0},
};

static BoxMap power9_box_map[NUM_UNITS] = {
    [PMC] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX1] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX2] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX3] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX4] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX5] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX6] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX7] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
};

static char* power9_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [MBOX0] = "/sys/bus/event_source/devices/nest_mba0_imc",
    [MBOX1] = "/sys/bus/event_source/devices/nest_mba1_imc",
    [MBOX2] = "/sys/bus/event_source/devices/nest_mba2_imc",
    [MBOX3] = "/sys/bus/event_source/devices/nest_mba3_imc",
    [MBOX4] = "/sys/bus/event_source/devices/nest_mba4_imc",
    [MBOX5] = "/sys/bus/event_source/devices/nest_mba5_imc",
    [MBOX6] = "/sys/bus/event_source/devices/nest_mba6_imc",
    [MBOX7] = "/sys/bus/event_source/devices/nest_mba7_imc",
};
