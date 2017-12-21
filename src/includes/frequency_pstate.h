#ifndef LIKWID_FREQUENCY_PSTATE
#define LIKWID_FREQUENCY_PSTATE

uint64_t freq_pstate_getCpuClockMax(const int cpu_id );
uint64_t freq_pstate_getCpuClockMin(const int cpu_id );

int freq_pstate_getTurbo(const int cpu_id );

#endif
