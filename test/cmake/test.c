
#include <likwid-marker.h>
#include <likwid.h>

#if !defined(LIKWID_PERFMON)
#error "LIKWID_PERFMON is not defined"
#endif

#if !defined(LIKWID_NVMON)
#warning "LIKWID_NVMON is not defined"
#endif

int main()
{
  return 0;
}
