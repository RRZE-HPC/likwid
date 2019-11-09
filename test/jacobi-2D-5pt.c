#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
#include<sys/time.h>

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#endif

int main()
{
    //this is minimum iteration;
    //total iteration is a multiple of
    //this, depending on runtime
    int niter = 10;

    int startSize = 1000;
    int endSize = 20000;
    int incSize = 1000;

    printf("%5s\t%7s\t\t%10s\n","Thread", "Size", "MLUPS");

    int thread_num = 1;
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_INIT;
#endif

#pragma omp parallel
    {
#pragma omp single
        {
            thread_num = omp_get_num_threads();
        }
    }

    for(int Size= startSize; Size<endSize; Size+=incSize)
    {
        double *phi = (double*) malloc(Size*Size*sizeof(double));
        double *phi_new = (double*) malloc(Size*Size*sizeof(double));
        char regname[100];
        snprintf(regname, 99, "size-%d", Size);

#pragma omp parallel for schedule(runtime)
        for(int i=0;i<Size;++i)
        {
            for(int j=0;j<Size;++j)
            {
                phi[i*Size+j]=1;
                phi_new[i*Size+j]=1;
            }
        }

        int ctr = 0;

        struct timeval tym;
        gettimeofday(&tym,NULL);
        double wcs=tym.tv_sec+(tym.tv_usec*1e-6);
        double wce=wcs;

#ifdef LIKWID_PERFMON
#pragma omp parallel
{
        LIKWID_MARKER_START(regname);
}
#endif
        while((wce-wcs) < 0.1)
        {
            for(int iter=1;iter<=niter;++iter)
            {

#pragma omp parallel for schedule(runtime)
                for(int i=1;i<Size-1;++i)
                {
#pragma simd
                    for(int j=1;j<Size-1;++j)
                    {
                        phi_new[i*Size+j]=phi[i*Size+j] + phi[(i+1)*Size+j]+phi[(i-1)*Size+j]+phi[i*Size+j+1] +phi[i*Size+j-1];
                    }
                }
                //swap arrays
                double* temp = phi_new;
                phi_new = phi;
                phi = temp;
            }
            ++ctr;
            gettimeofday(&tym,NULL);
            wce=tym.tv_sec+(tym.tv_usec*1e-6);
        }
#ifdef LIKWID_PERFMON
#pragma omp parallel
{

        LIKWID_MARKER_STOP(regname);
}
#endif

        double size_d = Size;
        double mlups = (size_d*size_d*niter*ctr*1.0e-6)/(wce-wcs);

        char thread_num_str[5];
        sprintf(thread_num_str, "%d", thread_num);
        char Size_str[7];
        sprintf(Size_str, "%d", Size);
        char mlup_str[10];
        sprintf(mlup_str, "%6.4f", mlups);
        printf("%5s\t%7s\t\t%10s\n",thread_num_str, Size_str, mlup_str);

        free(phi);
        free(phi_new);

    }
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_CLOSE;
#endif
    return 0;
}
