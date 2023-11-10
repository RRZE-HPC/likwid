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


static int print_cmd(char* cmd)
{
    int ret = 0;
    FILE* cmdfp = NULL;
    char *cmdout = NULL;
    size_t cmdout_size = 2000;
    cmdfp = popen(cmd, "r");
    if (cmdfp)
    {
        cmdout = malloc(cmdout_size * sizeof(char));
        if (!cmdout)
        {
            return -2;
        }
print_cmd_more:
        ret = fread(cmdout, sizeof(char), cmdout_size-1, cmdfp);
        if (ret > 0)
        {
            cmdout[ret] = '\0';
            printf("%s", cmdout);
            if (ret == cmdout_size-1)
            {
                goto print_cmd_more;
            }
            printf("\n");
        }
        free(cmdout);
        return pclose(cmdfp);
    }
    return -1;
}

#ifdef PTHREADS
struct thread_info {
    pthread_t pthread;
    pthread_attr_t pattr;
    int thread_id;
    int mpi_id;
    pid_t pid;
    pid_t ppid;
    pthread_barrier_t *barrier;
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
    pthread_barrier_wait(tinfo->barrier);
    printf ("Rank %d Thread %d running on Node %s core %d/%d with pid %d and tid %d\n",tinfo->mpi_id, tinfo->thread_id, host, sched_getcpu(), get_sched(), tinfo->pid ,gettid());
    pthread_barrier_wait(tinfo->barrier);
    if (tinfo->thread_id == 0 && (!system("command -v pstree > /dev/null 2>&1")))
    {
        sleep(tinfo->mpi_id+1);
        char cmd[1024];
        pid_t pid = getppid();
        snprintf(cmd, 1023, "pstree -p -H %d %d",pid, pid);
        print_cmd(cmd);
    }
    pthread_barrier_wait(tinfo->barrier);

    if (tinfo->thread_id != 0)
        pthread_exit(NULL);
}
#endif


main(int argc, char **argv)
{
    int i = 0;
    int rank = 0, size = 1;
    char host[HOST_NAME_MAX];
    pid_t master_pid = getpid();


    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(host, HOST_NAME_MAX);

    MASTER(MPI started);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process with rank %d running on Node %s core %d/%d with pid %d\n",rank ,host, sched_getcpu(),get_cpu_id(), master_pid);
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);



#ifdef _OPENMP
    MASTER(Enter OpenMP parallel region);
    MPI_Barrier(MPI_COMM_WORLD);
    MASTER(Start OpenMP threads);
#pragma omp parallel
    {
#pragma omp barrier

#pragma omp critical
        {
            printf ("Rank %d Thread %d running on Node %s core %d/%d with pid %d and tid %d\n",rank,omp_get_thread_num(), host, sched_getcpu(), get_sched(), master_pid ,gettid());
        }
#pragma omp barrier
#pragma omp master
        {
            if( !system("command -v pstree > /dev/null 2>&1") )
            {
                sleep(rank+1);
                pid_t pid = getppid();
                char cmd[1024];
                sprintf(cmd, "pstree -p -H %d %d",pid, pid);
                print_cmd(cmd);
            }
        }
    }
#endif


#ifdef PTHREADS
    int err = 0;
    int nthreads = 4;
    if (getenv("PTHREAD_THREADS") != NULL)
    {
        nthreads = atoi(getenv("PTHREAD_THREADS"));
    }
    if (nthreads <= 0)
    {
        MPI_Finalize();
        return -1;
    }
    struct thread_info* tinfos = NULL;
    pthread_barrier_t bar;
    pthread_barrierattr_t bar_attrs;


    tinfos = malloc(nthreads * sizeof(struct thread_info));
    if (!tinfos)
    {
        MPI_Finalize();
        return -1;
    }
    pthread_barrier_init(&bar, NULL, nthreads);


    pid_t pid = getppid();
    for (i = 0; i < nthreads; i++)
    {
        tinfos[i].pthread = 0;
        tinfos[i].pattr;
        tinfos[i].thread_id = i;
        tinfos[i].mpi_id = rank;
        tinfos[i].pid = master_pid;
        tinfos[i].ppid = pid;
        tinfos[i].barrier = &bar;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MASTER(Start Pthread threads);
    MPI_Barrier(MPI_COMM_WORLD);
    for (i = 1; i < nthreads; i++)
    {
        err = pthread_create(&tinfos[i].pthread, NULL, thread_start, (void*)&tinfos[i]);
        if (err != 0) printf("pthread_create %d error: %s\n", i, strerror(err));
    }
    thread_start((void*)&tinfos[0]);

    for (i = 1; i < nthreads; i++)
    {
        pthread_join(tinfos[i].pthread, NULL);
    }
    pthread_barrier_destroy(&bar);
#endif

    MPI_Finalize();

    return 0;
}
