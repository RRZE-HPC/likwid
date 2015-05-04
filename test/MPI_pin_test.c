#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/types.h>

#ifdef _OPENMP
extern int omp_get_num_threads();
extern int omp_get_thread_num();
#endif

#include <sched.h>

#define HOST_NAME_MAX 1024
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
    printf("Process with rank %d running on Node %s Core %d\n",rank ,host, sched_getcpu());
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MASTER(Enter OpenMP parallel region);
    MPI_Barrier(MPI_COMM_WORLD);
#pragma omp parallel
    {
#pragma omp master
        {
            pid_t pid = getppid();
            char cmd[1024];
            sprintf(cmd, "pstree -p -H %d %d",pid, pid);
            system(cmd);
        }
#pragma omp critical
        {
            printf ("Rank %d Thread %d running on core %d \n",rank,omp_get_thread_num(), sched_getcpu());
            fflush(stdout);
        }

    }

    sleep(2);

    MPI_Finalize();
}
