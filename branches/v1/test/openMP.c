#include <stdio.h>

#include <likwid.h>

int main()
{
  int ax[1024];
  int bx[1024];
  int cx[1024];
  int ix;

  int regionId;
  int coreId = 0;
  int threadId = 0;
  int numberOfThreads = 1;
  int numberOfRegions = 1;

#pragma omp parallel 
  {
#pragma omp master
    {
      numberOfThreads = omp_get_num_threads();
      printf(("Number of Threads requested = %d\n"),numberOfThreads);
      likwid_markerInit(numberOfThreads,numberOfRegions);
      regionId = likwid_markerRegisterRegion("Main");
    }
  }

    regionId = likwid_markerGetRegionId("Main");

#pragma omp parallel private(coreId, threadId)
    {
        threadId = omp_get_thread_num();
        coreId = likwid_threadGetProcessorId();
        printf("Marker start region for regionId= %d threadId = %d coreId = %d\n", regionId, threadId, coreId);
        likwid_markerStartRegion(threadId, coreId);

#pragma omp for
        for (ix = 0; ix < 1024; ++ix) {
            cx[ix] = ax[ix] + bx[ix];
        }

        printf("Marker stop region for regionId= %d threadId = %d coreId = %d\n", regionId, threadId, coreId);
        likwid_markerStopRegion(threadId, coreId, regionId);
    }

  printf("Closing marker\n");
  likwid_markerClose();
  printf("Closing marker done\n");

  return 0;
}
