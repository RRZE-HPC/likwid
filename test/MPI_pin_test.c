#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <errno.h>
#include <sys/types.h>
#include <string.h>
#include <sys/syscall.h>

#include <mpi.h>


#ifdef _OPENMP
extern int omp_get_num_threads();
extern int omp_get_thread_num();
#endif

#ifdef PTHREADS
#include <pthread.h>
#endif

#include <sched.h>

#define HOST_NAME_MAX 1024
#define MASTER(msg) \
    if (rank == 0)  printf(#msg "\n")
#define gettid() (int)syscall(SYS_gettid)

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


int get_sched()
{
    int i = 0;
    cpu_set_t my_set;
    int nproc = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO(&my_set);
    sched_getaffinity(gettid(), sizeof(cpu_set_t), &my_set);
    for (i = 0; i < nproc; i++)
    {
        if (CPU_ISSET(i, &my_set))
            return i;
    }
    return -1;
}

#ifdef PTHREADS
struct thread_info {
    int thread_id;
    int mpi_id;
    pid_t pid;
};

void *
thread_start(void *arg)
{
    
    int i = 0;
    struct thread_info *tinfo = arg;
    char host[HOST_NAME_MAX+1];
    if (host)
    {
        gethostname(host, HOST_NAME_MAX);
    }
    
    printf ("Rank %d Thread %d running on Node %s core %d/%d with pid %d and tid %d\n",tinfo->mpi_id, tinfo->thread_id, host, sched_getcpu(), get_sched(), getpid(),gettid());
    if (tinfo->thread_id == 0)
    {
        sleep(tinfo->mpi_id);
        char cmd[1024];
        pid_t pid = getppid();
        snprintf(cmd, 1023, "pstree -p -H %d %d",pid, pid);
/*        system(cmd);*/
    }
    
    pthread_exit(&i);
}
#endif


main(int argc, char **argv)
{
    int i = 0;
    int rank = 0, size = 1;
    char host[HOST_NAME_MAX];


    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(host, HOST_NAME_MAX);

    MASTER(MPI started);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process with rank %d running on Node %s Core %d/%d\n",rank ,host, sched_getcpu(),get_cpu_id());
    MPI_Barrier(MPI_COMM_WORLD);



#ifdef _OPENMP
    MASTER(Enter OpenMP parallel region);
    MPI_Barrier(MPI_COMM_WORLD);
#pragma omp parallel
    {
#pragma omp barrier

#pragma omp critical
        {
            printf ("Rank %d Thread %d running on Node %s core %d/%d with pid %d and tid %d\n",rank,omp_get_thread_num(), host, sched_getcpu(), get_sched(), getpid(),gettid());
        }
#pragma omp master
        {
            pid_t pid = getppid();
            char cmd[1024];
            sprintf(cmd, "pstree -p -H %d %d",pid, pid);
            system(cmd);
        }
    }
#endif


#ifdef PTHREADS
    int err = 0;
    struct thread_info tinfos[4];
    pthread_t threads[4] = {0};
    pthread_attr_t attrs[4];

    pid_t pid = getppid();
    for (i = 0; i < 4; i++)
    {
        tinfos[i].thread_id = i;
        tinfos[i].mpi_id = rank;
        tinfos[i].pid = pid;
    }

    for (i = 0; i < 4; i++)
    {
        err = pthread_create(&threads[i], &attrs[i], thread_start, (void*)&tinfos[i]);
        if (err != 0) printf("pthread_create %d error: %s\n", i, strerror(err));
    }

    for (i = 0; i < 4; i++)
    {
        pthread_join(threads[i], NULL);
    }
#endif

    MPI_Finalize();

    return 0;
}
