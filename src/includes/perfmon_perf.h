#ifndef PERFMON_PERF_H
#define PERFMON_PERF_H

#include <perfmon_types.h>

#define MAX_SW_EVENTS 9


extern int init_perf_event(int cpu_id);

extern int setup_perf_event(int cpu_id, PerfmonEvent *event);

extern int read_perf_event(int cpu_id, uint64_t eventID, uint64_t *data);

extern int stop_perf_event(int cpu_id, uint64_t eventID);
extern int stop_all_perf_event(int cpu_id);

extern int clear_perf_event(int cpu_id, uint64_t eventID);
extern int clear_all_perf_event(int cpu_id);

extern int start_perf_event(int cpu_id, uint64_t eventID);
extern int start_all_perf_event(int cpu_id);

extern int close_perf_event(int cpu_id, uint64_t eventID);

extern int finalize_perf_event(int cpu_id);

#endif