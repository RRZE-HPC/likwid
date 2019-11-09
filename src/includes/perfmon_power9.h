#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>
#include <perfmon_power9_counters.h>
#include <perfmon_power9_events.h>

static int perfmon_numCountersPower9 = NUM_COUNTERS_POWER9;
static int perfmon_numCoreCountersPower9 = NUM_COUNTERS_POWER9;
static int perfmon_numArchEventsPower9 = NUM_ARCH_EVENTS_POWER9;

