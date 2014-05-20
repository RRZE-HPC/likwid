/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.8 2007/02/19 23:57:39 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
#define _GNU_SOURCE
#include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <omp.h>
# include <limits.h>
# include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <pthread.h>

/* INSTRUCTIONS:
 *
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of 
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

# define N	60000000
# define NTIMES	10
# define OFFSET	0

/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *	4) Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */
#define gettid() syscall(SYS_gettid)
#include <likwid.h>

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

static double	a[N+OFFSET],
		b[N+OFFSET],
		c[N+OFFSET];

static double	avgtime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {
    2 * sizeof(double) * N,
    2 * sizeof(double) * N,
    3 * sizeof(double) * N,
    3 * sizeof(double) * N
    };

static int
getProcessorID(cpu_set_t* cpu_set)
{
    int processorId;

    for (processorId=0;processorId<128;processorId++)
    {
        if (CPU_ISSET(processorId,cpu_set))
        {
            break;
        }
    }
    return processorId;
}

int  threadGetProcessorId()
{
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);

    return getProcessorID(&cpu_set);
}

extern double mysecond();
extern void checkSTREAMresults();
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif
int
main()
    {
    int			quantum, checktick();
    int			BytesPerWord;
    register int	j, k;
    double		scalar, t, times[4][NTIMES];

    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    printf("STREAM version $Revision: 5.8 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(double);
    printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
	BytesPerWord);

    printf(HLINE);
    printf("Array size = %d, Offset = %d\n" , N, OFFSET);
    printf("Total memory required = %.1f MB.\n",
	(3.0 * BytesPerWord) * ( (double) N / 1048576.0));
    printf("Each test is run %d times, but only\n", NTIMES);
    printf("the *best* time for each is used.\n");

#ifdef LIKWID_PERFMON
    printf("Using likwid\n");
#endif

    LIKWID_MARKER_INIT;

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel
    {
	LIKWID_MARKER_THREADINIT;
#pragma omp master
	{
	    k = omp_get_num_threads();
	    printf ("Number of Threads requested = %i\n",k);
    }

    printf ("Thread %d running on processor %d ....\n",omp_get_thread_num(),threadGetProcessorId());
    }
#endif

    LIKWID_MARKER_START("init");
    /* Get initial value for system clock. */
//#pragma omp parallel for
    for (j=0; j<N; j++) {
	a[j] = 1.0;
	b[j] = 2.0;
	c[j] = 0.0;
	}
    LIKWID_MARKER_STOP("init");

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }

    t = mysecond();
#pragma omp parallel for
    for (j = 0; j < N; j++)
	a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);

    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
    {
        times[0][k] = mysecond();
#pragma omp parallel
	{
        LIKWID_MARKER_START("copy");
#pragma omp for
        for (j=0; j<N; j++)
            c[j] = a[j];
        LIKWID_MARKER_STOP("copy");
	}
        times[0][k] = mysecond() - times[0][k];

        times[1][k] = mysecond();
#pragma omp parallel
	{
        LIKWID_MARKER_START("scale");
#pragma omp for
        for (j=0; j<N; j++)
            b[j] = scalar*c[j];
        LIKWID_MARKER_STOP("scale");
	}
        times[1][k] = mysecond() - times[1][k];

        times[2][k] = mysecond();
#pragma omp parallel
	{
        LIKWID_MARKER_START("add");
#pragma omp for
        for (j=0; j<N; j++)
            c[j] = a[j]+b[j];
        LIKWID_MARKER_STOP("add");
	}
        times[2][k] = mysecond() - times[2][k];

        times[3][k] = mysecond();
#pragma omp parallel
	{
        LIKWID_MARKER_START("triad");
#pragma omp for
        for (j=0; j<N; j++)
            a[j] = b[j]+scalar*c[j];
        LIKWID_MARKER_STOP("triad");
	}
        times[3][k] = mysecond() - times[3][k];
    }

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}

    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
	avgtime[j] = avgtime[j]/(double)(NTIMES-1);

	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults();
    printf(HLINE);

    LIKWID_MARKER_CLOSE;
    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults ()
{
	double aj,bj,cj,scalar;
	double asum,bsum,csum;
	double epsilon;
	int	j,k;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }
	aj = aj * (double) (N);
	bj = bj * (double) (N);
	cj = cj * (double) (N);

	asum = 0.0;
	bsum = 0.0;
	csum = 0.0;
	for (j=0; j<N; j++) {
		asum += a[j];
		bsum += b[j];
		csum += c[j];
	}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
	epsilon = 1.e-8;

	if (abs(aj-asum)/asum > epsilon) {
		printf ("Failed Validation on array a[]\n");
		printf ("        Expected  : %f \n",aj);
		printf ("        Observed  : %f \n",asum);
	}
	else if (abs(bj-bsum)/bsum > epsilon) {
		printf ("Failed Validation on array b[]\n");
		printf ("        Expected  : %f \n",bj);
		printf ("        Observed  : %f \n",bsum);
	}
	else if (abs(cj-csum)/csum > epsilon) {
		printf ("Failed Validation on array c[]\n");
		printf ("        Expected  : %f \n",cj);
		printf ("        Observed  : %f \n",csum);
	}
	else {
		printf ("Solution Validates\n");
	}
}

