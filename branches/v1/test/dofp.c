#include <stdio.h>
#ifdef PERFMON
#include <likwid.h>
#endif

#define SIZE 1000000
#define N    100

double sum = 0, a[SIZE], b[SIZE], c[SIZE];

main()
{
   int i, j;
   double alpha = 3.124;
   int RegionId, RegionId2;
   int numberOfThreads = 1;
   int numberOfRegions = 2;

#ifdef PERFMON
   int coreID = likwid_processGetProcessorId();
   printf("Using likwid\n");
   likwid_markerInit(numberOfThreads,numberOfRegions);
   RegionId = likwid_markerRegisterRegion("Main");
   RegionId2 = likwid_markerRegisterRegion("Accum");
#endif
   
   for (i = 0; i < SIZE; i++) {
      a[i] = 2.0;
      b[i] = 3.0;
      c[i] = 3.0;
   }
   
#ifdef PERFMON
   likwid_markerStartRegion(0, coreID);
#endif
   for (j = 0; j < N; j++)
   {
      for (i = 0; i < SIZE; i++) {
         a[i] = b[i] + alpha * c[i];
         sum += a[i];
      }
   }
#ifdef PERFMON
   likwid_markerStopRegion(0, coreID, RegionId);
#endif
 
   for (j = 0; j < N; j++)
   {
#ifdef PERFMON
	   likwid_markerStartRegion(0, coreID);
#endif
      for (i = 0; i < SIZE; i++) {
         a[i] = b[i] + alpha * c[i];
         sum += a[i];
      }
#ifdef PERFMON
	  likwid_markerStopRegion(0, coreID, RegionId2);
#endif
   }


   for (j = 0; j < N; j++)
      for (i = 0; i < SIZE; i++) {
         a[i] = b[i] + alpha * c[i];
         sum += a[i];
      }
#ifdef PERFMON
   likwid_markerClose();
#endif
   printf( "OK, dofp result = %e\n", sum);
}
