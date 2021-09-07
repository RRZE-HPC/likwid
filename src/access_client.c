/*
 * =======================================================================================
 *
 *      Filename:  access_client.c
 *
 *      Description:  Interface to the access daemon for the access module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <pthread.h>
#include <sys/syscall.h>

#include <types.h>
#include <error.h>
#include <topology.h>
#include <access.h>
#include <access_client.h>
#include <configuration.h>
#include <affinity.h>

#if defined(__x86_64__) || defined(__i386__)
#include <access_x86_rdpmc.h>
#endif

#define gettid() syscall(SYS_gettid)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int globalSocket = -1;
static pid_t masterPid = 0;
static int cpuSockets_open = 0;
static int *cpuSockets = NULL;
static int nr_daemons = 0;
static pid_t *daemon_pids = NULL;
static int *daemon_pinned = NULL;
static pthread_mutex_t globalLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t *cpuLocks = NULL;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static char*
access_client_strerror(AccessErrorType det)
{
    switch (det)
    {
        case ERR_NOERROR:    return "No error";
        case ERR_UNKNOWN:    return "unknown command";
        case ERR_RESTREG:    return "access to this register is not allowed";
        case ERR_OPENFAIL:   return "failed to open device file";
        case ERR_RWFAIL:     return "failed to read/write register";
        case ERR_DAEMONBUSY: return "daemon already has a same/higher priority client";
        case ERR_NODEV:      return "no such pci device";
        case ERR_LOCKED:     return "access to registers is locked";
        default:             return "UNKNOWN errorcode";
    }
}

static int
access_client_errno(AccessErrorType det)
{
    switch (det)
    {
        case ERR_NOERROR:    return 0;
        case ERR_UNKNOWN:    return -EFAULT;
        case ERR_RESTREG:    return -EPERM;
        case ERR_OPENFAIL:   return -ENXIO;
        case ERR_RWFAIL:     return -EIO;
        case ERR_DAEMONBUSY: return -EBUSY;
        case ERR_NODEV:      return -ENODEV;
        default:             return -EFAULT;
    }
}

static int
access_client_startDaemon(int cpu_id)
{
    /* Check the function of the daemon here */
    int res = 0;
    char* filepath;
    char *newargv[] = { NULL };
    char *newenv[] = { NULL };
    char *safeexeprog = TOSTRING(ACCESSDAEMON);
    char exeprog[1024];
    struct sockaddr_un address;
    size_t address_length;
    int  ret;
    pid_t pid;
    int timeout = 1000;
    int socket_fd = -1;
    int print_once = 0;

    if (config.daemonPath != NULL)
    {
        strcpy(exeprog, config.daemonPath);
    }
    else
    {
        strcpy(exeprog, safeexeprog);
    }

    if (access(exeprog, X_OK))
    {
        ERROR_PRINT(Failed to find the daemon '%s'\n, exeprog);
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(DEBUGLEV_INFO, Starting daemon %s, exeprog);
    pid = fork();

    if (pid == 0)
    {
/*        Remove pinning here and delay it until first read or write call to check*/
/*        if we are running in a multi-threaded environment.*/
/*        if (cpu_id >= 0)*/
/*        {*/
/*            cpu_set_t cpuset;*/
/*            CPU_ZERO(&cpuset);*/
/*            CPU_SET(cpu_id, &cpuset);*/
/*            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);*/
/*        }*/
        ret = execve (exeprog, newargv, newenv);

        if (ret < 0)
        {
            //ERRNO_PRINT;
            ERROR_PRINT(Failed to execute the daemon '%s'\n, exeprog);
            exit(EXIT_FAILURE);
        }
    }
    else if (pid < 0)
    {
        ERROR_PRINT(Failed to fork access daemon for CPU %d, cpu_id);
        return pid;
    }

    EXIT_IF_ERROR(socket_fd = socket(AF_LOCAL, SOCK_STREAM, 0), socket() failed);

    address.sun_family = AF_LOCAL;
    address_length = sizeof(address);
    snprintf(address.sun_path, sizeof(address.sun_path), TOSTRING(LIKWIDSOCKETBASE) "-%d", pid);
    filepath = strdup(address.sun_path);

    res = connect(socket_fd, (struct sockaddr *) &address, address_length);
    while (res && timeout > 0)
    {
        usleep(2500);
        res = connect(socket_fd, (struct sockaddr *) &address, address_length);

        if (res == 0)
        {
            break;
        }

        timeout--;
        DEBUG_PRINT(DEBUGLEV_INFO, Still waiting for socket %s for CPU %d..., filepath, cpu_id);
    }

    if (timeout <= 0)
    {
        ERRNO_PRINT;  /* should hopefully still work, as we make no syscalls in between. */
        fprintf(stderr, "Exiting due to timeout: The socket file at '%s' could not be\n", filepath);
        fprintf(stderr, "opened within 10 seconds. Consult the error message above\n");
        fprintf(stderr, "this to find out why. If the error is 'no such file or directoy',\n");
        fprintf(stderr, "it usually means that likwid-accessD just failed to start.\n");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(DEBUGLEV_INFO, Successfully opened socket %s to daemon for CPU %d, filepath, cpu_id);
    free(filepath);
    daemon_pids[cpu_id] = pid;
    nr_daemons++;
    return socket_fd;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
access_client_init(int cpu_id)
{
    topology_init();
    numa_init();
    affinity_init();
    if (!cpuSockets)
    {
        cpuSockets = malloc(cpuid_topology.numHWThreads * sizeof(int));
        memset(cpuSockets, -1, cpuid_topology.numHWThreads * sizeof(int));
    }
    if (!daemon_pids)
    {
        daemon_pids = malloc(cpuid_topology.numHWThreads * sizeof(int));
        memset(daemon_pids, 0, cpuid_topology.numHWThreads * sizeof(int));
        daemon_pinned = malloc(cpuid_topology.numHWThreads * sizeof(int));
        memset(daemon_pinned, 0, cpuid_topology.numHWThreads * sizeof(int));
    }
    if (!cpuLocks)
    {
        cpuLocks = malloc(cpuid_topology.numHWThreads * sizeof(pthread_mutex_t));
        for (int i = 0; i < cpuid_topology.numHWThreads; i++)
        {
            pthread_mutex_init(&cpuLocks[i], NULL);
        }
    }
#if defined(__x86_64__) || defined(__i386__)
    access_x86_rdpmc_init(cpu_id);
#endif
    if (masterPid != 0 && gettid() == masterPid)
    {
        return 0;
    }
    if (cpuSockets[cpu_id] < 0)
    {
        pthread_mutex_lock(&cpuLocks[cpu_id]);
        cpuSockets[cpu_id] = access_client_startDaemon(cpu_id);
        if (cpuSockets[cpu_id] < 0)
        {
            //ERROR_PRINT(Start of access daemon failed for CPU %d, cpu_id);
            pthread_mutex_unlock(&cpuLocks[cpu_id]);
            return cpuSockets[cpu_id];
        }
        cpuSockets_open++;
        pthread_mutex_unlock(&cpuLocks[cpu_id]);
        if (globalSocket == -1)
        {
            pthread_mutex_lock(&globalLock);
            globalSocket = cpuSockets[cpu_id];
            masterPid = gettid();
            pthread_mutex_unlock(&globalLock);
        }
        return 0;
    }
    return -1;
}

int
access_client_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data)
{
    int ret;
    int socket = globalSocket;
    pthread_mutex_t* lockptr = &globalLock;
    AccessDataRecord record;
    memset(&record, 0, sizeof(AccessDataRecord));
    record.cpu = cpu_id;
    record.device = MSR_DEV;
    record.errorcode = ERR_OPENFAIL;

    if (cpuSockets_open == 0)
    {
        return -ENOENT;
    }

    if (cpuSockets[cpu_id] < 0 && gettid() != masterPid)
    {
        pthread_mutex_lock(&cpuLocks[cpu_id]);
        cpuSockets[cpu_id] = access_client_startDaemon(cpu_id);
        cpuSockets_open++;
        if (!daemon_pinned[cpu_id])
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_id, &cpuset);
            DEBUG_PRINT(DEBUGLEV_INFO, Pinning daemon %d to CPU %d, daemon_pids[cpu_id], cpu_id);
            sched_setaffinity(daemon_pids[cpu_id], sizeof(cpu_set_t), &cpuset);
            daemon_pinned[cpu_id] = 1;
        }
        pthread_mutex_unlock(&cpuLocks[cpu_id]);
    }
    else if (cpuSockets[cpu_id] > 0 && gettid() == masterPid &&
             cpuSockets_open > 1 && !daemon_pinned[cpu_id])
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        DEBUG_PRINT(DEBUGLEV_INFO, Pinning master daemon %d to CPU %d, daemon_pids[cpu_id], cpu_id);
        sched_setaffinity(daemon_pids[cpu_id], sizeof(cpu_set_t), &cpuset);
        daemon_pinned[cpu_id] = 1;
    }

    if ((cpuSockets[cpu_id] >= 0) && (cpuSockets[cpu_id] != globalSocket))
    {
        socket = cpuSockets[cpu_id];
        lockptr = &cpuLocks[cpu_id];
    }

    if (dev != MSR_DEV)
    {
        record.cpu = affinity_thread2socket_lookup[cpu_id];
        record.device = dev;
    }
    else
    {
#if defined(__x86_64__) || defined(__i386__)
        access_x86_rdpmc_read(cpu_id, reg, data);
#endif
    }
    if (socket != -1)
    {
        record.reg = reg;
        record.data = 0x00;
        record.type = DAEMON_READ;

        pthread_mutex_lock(lockptr);
        CHECK_ERROR(write(socket, &record, sizeof(AccessDataRecord)), socket write failed);
        CHECK_ERROR(read(socket, &record, sizeof(AccessDataRecord)), socket read failed);
        *data = record.data;
        pthread_mutex_unlock(lockptr);

        if (record.errorcode != ERR_NOERROR)
        {
            if (dev == MSR_DEV)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Got error '%s' from access daemon reading reg 0x%X at CPU %d,
                            access_client_strerror(record.errorcode), reg, cpu_id);
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Got error '%s' from access daemon reading reg 0x%X on socket %d,
                            access_client_strerror(record.errorcode), reg, cpu_id);
            }
            *data = 0;
            return access_client_errno(record.errorcode);
        }
    }
    else
    {
        *data = 0;
        return -EBADFD;
    }
    return 0;
}

int
access_client_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data)
{
    int socket = globalSocket;
    int ret;
    AccessDataRecord record;
    memset(&record, 0, sizeof(AccessDataRecord));
    record.cpu = cpu_id;
    record.device = MSR_DEV;
    pthread_mutex_t* lockptr = &globalLock;
    record.errorcode = ERR_OPENFAIL;

    if (cpuSockets_open == 0)
    {
        return -ENOENT;
    }

    if (cpuSockets[cpu_id] < 0 && gettid() != masterPid)
    {
        pthread_mutex_lock(&cpuLocks[cpu_id]);
        cpuSockets[cpu_id] = access_client_startDaemon(cpu_id);
        cpuSockets_open++;
        if (!daemon_pinned[cpu_id])
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_id, &cpuset);
            DEBUG_PRINT(DEBUGLEV_INFO, Pinning daemon %d to CPU %d, daemon_pids[cpu_id], cpu_id);
            sched_setaffinity(daemon_pids[cpu_id], sizeof(cpu_set_t), &cpuset);
            daemon_pinned[cpu_id] = 1;
        }
        pthread_mutex_unlock(&cpuLocks[cpu_id]);
    }
    else if (cpuSockets[cpu_id] > 0 && gettid() == masterPid &&
             cpuSockets_open > 1 && !daemon_pinned[cpu_id])
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        DEBUG_PRINT(DEBUGLEV_INFO, Pinning master daemon %d to CPU %d, daemon_pids[cpu_id], cpu_id);
        sched_setaffinity(daemon_pids[cpu_id], sizeof(cpu_set_t), &cpuset);
        daemon_pinned[cpu_id] = 1;
    }

    if ((cpuSockets[cpu_id] >= 0) && (cpuSockets[cpu_id] != socket))
    {
        socket = cpuSockets[cpu_id];
        lockptr = &cpuLocks[cpu_id];
    }

    if (dev != MSR_DEV)
    {
        record.cpu = affinity_thread2socket_lookup[cpu_id];
        record.device = dev;
    }
    if (socket != -1)
    {
        record.reg = reg;
        record.data = data;
        record.type = DAEMON_WRITE;

        pthread_mutex_lock(lockptr);
        CHECK_ERROR(write(socket, &record, sizeof(AccessDataRecord)), socket write failed);
        CHECK_ERROR(read(socket, &record, sizeof(AccessDataRecord)), socket read failed);
        pthread_mutex_unlock(lockptr);

        if (record.errorcode != ERR_NOERROR)
        {
            if (dev == MSR_DEV)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Got error '%s' from access daemon writing reg 0x%X at CPU %d,
                            access_client_strerror(record.errorcode), reg, cpu_id);
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Got error '%s' from access daemon writing reg 0x%X on socket %d,
                            access_client_strerror(record.errorcode), reg, cpu_id);
            }
            return access_client_errno(record.errorcode);
        }
    }
    else
    {
        return -EBADFD;
    }
    return 0;
}

void
access_client_finalize(int cpu_id)
{
    AccessDataRecord record;
    if (cpuSockets && cpuSockets[cpu_id] > 0)
    {
        memset(&record, 0, sizeof(AccessDataRecord));
        record.type = DAEMON_EXIT;
        record.cpu = cpu_id;
        CHECK_ERROR(write(cpuSockets[cpu_id], &record, sizeof(AccessDataRecord)),socket write failed);
        if (cpuSockets[cpu_id] == globalSocket)
        {
            globalSocket = -1;
        }
        CHECK_ERROR(close(cpuSockets[cpu_id]),socket close failed);
        cpuSockets[cpu_id] = -1;
        if (daemon_pids[cpu_id] != 0)
        {
            int status = 0;
            waitpid(daemon_pids[cpu_id], &status, 0);
            daemon_pids[cpu_id] = 0;
            nr_daemons--;
        }

        cpuSockets_open--;
    }
    if (cpuSockets_open == 0)
    {
        globalSocket = -1;
    }
    masterPid = 0;
#if defined(__x86_64__) || defined(__i386__)
    access_x86_rdpmc_finalize(cpu_id);
#endif
}

int
access_client_check(PciDeviceIndex dev, int cpu_id)
{
    int socket = globalSocket;
    pthread_mutex_t* lockptr = &globalLock;

    AccessDataRecord record;
    memset(&record, 0, sizeof(AccessDataRecord));
    record.cpu = cpu_id;
    record.device = dev;
    record.type = DAEMON_CHECK;
    record.errorcode = ERR_OPENFAIL;
    if (dev != MSR_DEV)
    {
        record.cpu = affinity_thread2socket_lookup[cpu_id];
    }
    if ((cpuSockets[cpu_id] > 0) && (cpuSockets[cpu_id] != globalSocket))
    {
        socket = cpuSockets[cpu_id];
        lockptr = &cpuLocks[cpu_id];
    }
    if ((cpuSockets[cpu_id] > 0) || ((cpuSockets_open == 1) && (globalSocket > 0)))
    {
        pthread_mutex_lock(lockptr);
        CHECK_ERROR(write(socket, &record, sizeof(AccessDataRecord)), socket write failed);
        CHECK_ERROR(read(socket, &record, sizeof(AccessDataRecord)), socket read failed);
        pthread_mutex_unlock(lockptr);
        if (record.errorcode == ERR_NOERROR )
        {
            return 1;
        }
        else
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Device check for dev %d on CPU %d with accessDaemon failed: %s\n, dev, cpu_id, access_client_strerror(record.errorcode));
        }
    }
    return 0;
}

void __attribute__((destructor (104))) close_access_client(void)
{
    if (cpuSockets)
    {
        for (int i = 0; i < cpuid_topology.numHWThreads; i++)
        {
            if (cpuSockets[i] > 0)
            {
                close(cpuSockets[i]);
                cpuSockets[i] = -1;
                cpuSockets_open--;
            }
        }
        free(cpuSockets);
        cpuSockets = NULL;
        cpuSockets_open = 0;
    }
    if (daemon_pids)
    {
        for (int i = 0; i < cpuid_topology.numHWThreads; i++)
        {
            if (daemon_pids[i] != 0)
            {
                int status = 0;
                waitpid(daemon_pids[i], &status, 0);
                daemon_pids[i] = 0;
                nr_daemons--;
            }
        }
        free(daemon_pids);
        daemon_pids = NULL;
        free(daemon_pinned);
        daemon_pinned = NULL;
        nr_daemons = 0;
    }
    if (cpuLocks)
    {
        for (int i = 0; i < cpuid_topology.numHWThreads; i++)
        {
            pthread_mutex_destroy(&cpuLocks[i]);
        }
        free(cpuLocks);
        cpuLocks = NULL;
    }
}
