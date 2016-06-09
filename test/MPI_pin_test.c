#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/types.h>
#include <string.h>
#include <sys/syscall.h>

#ifdef _OPENMP
extern int omp_get_num_threads();
extern int omp_get_thread_num();
#endif

#include <sched.h>

int get_cpu_id()
{
    int i;
    int cpu_id = 0;
    /* Get the the current process' stat file from the proc filesystem */
    FILE* procfile = fopen("/proc/self/stat", "r");
    long to_read = 8192;
    char* line;
    char buffer[to_read];
    int read = fread(buffer, sizeof(char), to_read, procfile);
    fclose(procfile);

    // Field with index 38 (zero-based counting) is the one we want
    line = strtok(buffer, " ");
    for (i = 1; i < 38; i++)
    {
        line = strtok(NULL, " ");
    }

    line = strtok(NULL, " ");
    cpu_id = atoi(line);
    return cpu_id;
}

#define HOST_NAME_MAX 1024
#define MASTER(msg) \
    if (rank == 0)  printf(#msg "\n")
#define gettid() (int)syscall(SYS_gettid)

main(int argc, char **argv)
{
    int rank;
    char* host;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    host = (char*) malloc(HOST_NAME_MAX * sizeof(char));
    gethostname(host, HOST_NAME_MAX);

    MASTER(MPI started);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process with rank %d running on Node %s Core %d/%d\n",rank ,host, sched_getcpu(),get_cpu_id());
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
#ifdef _OPENMP
#pragma omp critical
        {
            printf ("Rank %d Thread %d running on Node %s core %d/%d with pid %d and tid %d\n",rank,omp_get_thread_num(), host, sched_getcpu(),get_cpu_id(), getpid(),gettid());
        }
#endif

    }

    free(host);
    MPI_Finalize();
}
