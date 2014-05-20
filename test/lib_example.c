#include <stdio.h>
#include <perfmon.h> /*<<<*/

#define SIZE 1000000
#define N    100

double sum = 0, a[SIZE], b[SIZE], c[SIZE];
/* Ich geb zu ist ein bisschen sperrig :-) */
/* Nehalem */
char* eventStr = "INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,FP_COMP_OPS_EXE_SSE_FP_PACKED:PMC0,FP_COMP_OPS_EXE_SSE_FP_SCALAR:PMC1,FP_COMP_OPS_EXE_SSE_SINGLE_PRECISION:PMC2,FP_COMP_OPS_EXE_SSE_DOUBLE_PRECISION:PMC3";
/* Core 2 */
//char* eventStr = "INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,SIMD_COMP_INST_RETIRED_PACKED_DOUBLE:PMC0,SIMD_COMP_INST_RETIRED_SCALAR_DOUBLE:PMC1";

main()
{
   int i, j;
   double alpha = 3.124;
   /* Ein thread ProcessorId 1, dass muss dann auch so sein ! (likwid-pin -c 1) */
   int threads[1] = {1};
   int threadId = 0;

   perfmon_init(1,threads);   /*<<<*/ 
   perfmon_setupEventSetC(eventStr);   /*<<<*/
   
   for (i = 0; i < SIZE; i++) {
      a[i] = 2.0;
      b[i] = 3.0;
      c[i] = 3.0;
   }
   
   perfmon_startCounters();   /*<<<*/
   for (j = 0; j < N; j++)
   {
      for (i = 0; i < SIZE; i++) 
      {
         a[i] = b[i] + alpha * c[i];
         sum += a[i];
      }
   }
   perfmon_stopCounters();     /*<<<*/ 

   printf("Cycles unhalted Core: %e \n", perfmon_getResult(threadId,"FIXC1"));   /*<<<*/
   printf("Instructions retired: %e \n", perfmon_getResult(threadId,"FIXC0"));   /*<<<*/
   printf("CPI: %e \n", perfmon_getResult(threadId,"FIXC1")/perfmon_getResult(threadId,"FIXC0"));  /*<<<*/
   printf("FP_COMP_OPS_EXE_SSE_FP_PACKED: %e \n", perfmon_getResult(threadId,"PMC0"));  /*<<<*/
   printf("FP_COMP_OPS_EXE_SSE_FP_SCALAR: %e \n", perfmon_getResult(threadId,"PMC1"));  /*<<<*/

   printf( "OK, dofp result = %e\n", sum);
}
