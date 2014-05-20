#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#ifdef _OPENMP
extern int omp_get_num_threads();
extern int omp_get_thread_num();
#endif

#include <affinity.h>

#define MASTER(msg) \
    if (rank == 0)  printf(#msg "\n")

main(int argc, char **argv)
{
    int rank;
    char* host;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    host = (char*) malloc(HOST_NAME_MAX * sizeof(char));
    gethostname(host,HOST_NAME_MAX);

    MASTER(MPI started);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process with rank %d running on Node %s Core %d\n",rank ,host, likwid_getProcessorId());
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MASTER(Enter OpenMP parallel region);
    MPI_Barrier(MPI_COMM_WORLD);
#pragma omp parallel
    {
        int coreId = likwid_getProcessorId();
#pragma omp critical
        {
            printf ("Rank %d Thread %d running on core %d \n",rank,omp_get_thread_num(), coreId);
            fflush(stdout);
        }
    }

    sleep(2);

    MPI_Finalize();
}
