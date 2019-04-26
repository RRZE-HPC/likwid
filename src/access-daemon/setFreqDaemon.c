/*
 * =======================================================================================
 *
 *      Filename:  accessDaemon.c
 *
 *      Description:  Implementation of access daemon.
 *
 *      Version:   4.3.1
 *      Released:  04.01.2018
 *
 *      Authors:  Michael Meier, michael.meier@rrze.fau.de
 *                Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <syslog.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/fsuid.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/mman.h>


#include <lock.h>
//#include <error.h>
#include <frequency_client.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define SA struct sockaddr
#define str(x) #x

#define CHECK_FILE_ERROR(func, msg)  \
    if ((func) == 0) { syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); }

#define LOG_AND_EXIT_IF_ERROR(func, msg)  \
    if ((func) < 0) {  \
        syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_ERROR(func, msg)  \
    if ((func) < 0) { \
        fprintf(stderr, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno));  \
    }

#define PCI_ROOT_PATH    "/proc/bus/pci/"

//#define MAX_NUM_NODES    4



/* Lock file controlled from outside which prevents likwid to start.
 * Can be used to synchronize access to the hardware counters
 * with an external monitoring system. */

/* #####   TYPE DEFINITIONS   ########### */


struct cpufreq_files {
    int  cur_freq;
    int  max_freq;
    int  min_freq;
    int  avail_freq;
    int  avail_govs;
    int  driver;
    int  set_freq;
    int  set_gov;
};

char* cpufreq_files[] ={
    "scaling_cur_freq",
    "scaling_max_freq",
    "scaling_min_freq",
    "scaling_available_frequencies",
    "scaling_available_governors",
    "scaling_driver",
    "scaling_setspeed",
    "scaling_governor",
    NULL,
};

char* pstate_files[] ={
    "scaling_cur_freq",
    "scaling_max_freq",
    "scaling_min_freq",
    "scaling_available_frequencies",
    "scaling_available_governors",
    "scaling_driver",
    "scaling_setspeed",
    "scaling_governor",
    NULL,
};

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int sockfd = -1;
static int connfd = -1; /* temporary in to make it compile */
static char* filepath;
static const char* ident = "setFreqD";
static int avail_cpus = 0;
static struct cpufreq_files* cpufiles = NULL;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int get_avail_cpus(void)
{
    FILE *fpipe = NULL;
    char *ptr = NULL;
    char cmd_cpu[] = "cat /proc/cpuinfo  | grep 'processor' | sort -u | wc -l";
    char buff[256];

    if ( !(fpipe = (FILE*)popen(cmd_cpu,"r")) )
    {  // If fpipe is NULL
        return -errno;
    }
    ptr = fgets(buff, 256, fpipe);
    if (pclose(fpipe))
        return -errno;
    return atoi(buff);
}


static void close_cpu(struct cpufreq_files* cpufiles)
{
    if (cpufiles)
    {
/*        if (cpufiles->cur_freq >= 0)*/
/*        {*/
/*            syslog(LOG_INFO, "Close cur_freq %d\n", cpufiles->cur_freq);*/
/*            close(cpufiles->cur_freq);*/
/*            cpufiles->cur_freq = -1;*/
/*        }*/
        if (cpufiles->max_freq >= 0)
        {
            syslog(LOG_INFO, "Close max_freq %d\n", cpufiles->cur_freq);
            close(cpufiles->max_freq);
            cpufiles->max_freq = -1;
        }
        if (cpufiles->min_freq >= 0)
        {
            syslog(LOG_INFO, "Close min_freq %d\n", cpufiles->cur_freq);
            close(cpufiles->min_freq);
            cpufiles->min_freq = -1;
        }
        if (cpufiles->set_freq >= 0)
        {
            syslog(LOG_INFO, "Close set_freq %d\n", cpufiles->cur_freq);
            close(cpufiles->set_freq);
            cpufiles->set_freq = -1;
        }
        if (cpufiles->set_gov >= 0)
        {
            syslog(LOG_INFO, "Close set_gov %d\n", cpufiles->cur_freq);
            close(cpufiles->set_gov);
            cpufiles->set_gov = -1;
        }
/*        if (cpufiles->avail_freq >= 0)*/
/*        {*/
/*            syslog(LOG_INFO, "Close avail_freq %d\n", cpufiles->cur_freq);*/
/*            close(cpufiles->avail_freq);*/
/*            cpufiles->avail_freq = -1;*/
/*        }*/
/*        if (cpufiles->avail_govs >= 0)*/
/*        {*/
/*            syslog(LOG_INFO, "Close avail_govs %d\n", cpufiles->cur_freq);*/
/*            close(cpufiles->avail_govs);*/
/*            cpufiles->avail_govs = -1;*/
/*        }*/
/*        if (cpufiles->driver >= 0)*/
/*        {*/
/*            syslog(LOG_INFO, "Close driver %d\n", cpufiles->cur_freq);*/
/*            close(cpufiles->driver);*/
/*            cpufiles->driver = -1;*/
/*        }*/
        free(cpufiles);
    }
}

static int open_cpu_file(char* filename, int* fd)
{
    int f = -1;
    int access_flag = R_OK|W_OK;
    int open_flag = O_RDWR;

    f = open(filename, open_flag);
    if (f < 0)
    {
        syslog(LOG_ERR, "Failed to open file %s \n", filename);
        *fd = -1;
        return 0;
    }
    *fd = f;
    syslog(LOG_INFO, "Opened %s %s = %d\n", filename, (open_flag == O_RDONLY ? "readable" : "writable"), *fd);
    return 0;
}

static int open_cpu(int cpu, struct cpufreq_files* files)
{
    char dname[1025];
    char fname[1025];
    //struct cpufreq_files* files;
    FILE* fp = NULL;
    if (cpu >= 0)
    {
        int ret = snprintf(dname, 1024, "/sys/devices/system/cpu/cpu%d/cpufreq", cpu);
        if (ret > 0)
        {
            dname[ret] = '\0';
        }
        else
        {
            return -1;
        }


/*        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_cur_freq");*/
/*        if (ret > 0)*/
/*        {*/
/*            fname[ret] = '\0';*/
/*            if (open_cpu_file(fname, &files->cur_freq) < 0)*/
/*            {*/
/*                goto cleanup;*/
/*            }*/
/*        }*/
        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_max_freq");
        if (ret > 0)
        {
            fname[ret] = '\0';
            if (open_cpu_file(fname, &files->max_freq) < 0)
            {
                goto cleanup;
            }
        }
        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_min_freq");
        if (ret > 0)
        {
            fname[ret] = '\0';
            if (open_cpu_file(fname, &files->min_freq) < 0)
            {
                goto cleanup;
            }
        }
        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_setspeed");
        if (ret > 0)
        {
            fname[ret] = '\0';
            if (open_cpu_file(fname, &files->set_freq) < 0)
            {
                goto cleanup;
            }
        }
        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_governor");
        if (ret > 0)
        {
            fname[ret] = '\0';
            if (open_cpu_file(fname, &files->set_gov) < 0)
            {
                goto cleanup;
            }
        }
/*        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_available_governors");*/
/*        if (ret > 0)*/
/*        {*/
/*            fname[ret] = '\0';*/
/*            if (open_cpu_file(fname, &files->avail_govs) < 0)*/
/*            {*/
/*                goto cleanup;*/
/*            }*/
/*        }*/
/*        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_available_frequencies");*/
/*        if (ret > 0)*/
/*        {*/
/*            fname[ret] = '\0';*/
/*            if (open_cpu_file(fname, &files->avail_freq) < 0)*/
/*            {*/
/*                goto cleanup;*/
/*            }*/
/*        }*/
/*        ret = snprintf(fname, 1024, "%s/%s", dname, "scaling_driver");*/
/*        if (ret > 0)*/
/*        {*/
/*            fname[ret] = '\0';*/
/*            if (open_cpu_file(fname, &files->driver) < 0)*/
/*            {*/
/*                goto cleanup;*/
/*            }*/
/*        }*/
        return 0;
    }
cleanup:
    syslog(LOG_ERR, "Cleanup\n");
    close_cpu(files);
    return -1;
}


static void
kill_client(void)
{
    if (connfd != -1)
    {
        CHECK_ERROR(close(connfd), socket close failed);
    }

    connfd = -1;
}

static void
stop_daemon(void)
{
    kill_client();

    if (sockfd != -1)
    {
        CHECK_ERROR(close(sockfd), socket close sockfd failed);
    }

    free(filepath);
    if (cpufiles)
    {
        for (int i=0;i<avail_cpus;i++)
        {
            close_cpu(&cpufiles[i]);
        }
        free(cpufiles);
        cpufiles = NULL;
    }
    closelog();
    exit(EXIT_SUCCESS);
}



static void
Signal_Handler(int sig)
{
    if (sig == SIGPIPE)
    {
        syslog(LOG_NOTICE, "SIGPIPE? client crashed?!");
        stop_daemon();
    }

    /* For SIGALRM we just return - we're just here to create a EINTR */
    if (sig == SIGTERM)
    {
        stop_daemon();
    }
}

static void
daemonize(int* parentPid)
{
    pid_t pid, sid;

    *parentPid = getpid();

    /* already a daemon */
    if ( getppid() == 1 ) return;

    /* Fork off the parent process */
    pid = fork();

    if (pid < 0)
    {
        syslog(LOG_ERR, "fork failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* If we got a good PID, then we can exit the parent process. */
    if (pid > 0)
    {
        exit(EXIT_SUCCESS);
    }

    /* At this point we are executing as the child process */

    /* Create a new SID for the child process */
    sid = setsid();

    if (sid < 0)
    {
        syslog(LOG_ERR, "setsid failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Change the current working directory.  This prevents the current
       directory from being locked; hence not being able to remove it. */
    if ((chdir("/")) < 0)
    {
        syslog(LOG_ERR, "chdir failed:  %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Redirect standard files to /dev/null */
    {
        CHECK_FILE_ERROR(freopen( "/dev/null", "r", stdin), freopen stdin failed);
        CHECK_FILE_ERROR(freopen( "/dev/null", "w", stdout), freopen stdout failed);
        CHECK_FILE_ERROR(freopen( "/dev/null", "w", stderr), freopen stderr failed);
    }
}

static int freq_read(FreqDataRecord *rec)
{
    int read_fd = -1;
    int cpu = rec->cpu;
    struct cpufreq_files* f = &cpufiles[cpu];
    switch(rec->loc)
    {
        case FREQ_LOC_CUR:
            read_fd = f->cur_freq;
            break;
        case FREQ_LOC_MIN:
            read_fd = f->min_freq;
            break;
        case FREQ_LOC_MAX:
            read_fd = f->max_freq;
            break;
        case FREQ_LOC_GOV:
            read_fd = f->set_gov;
            break;
    }
    if (read_fd < 0)
    {
        rec->errorcode = FREQ_ERR_NOFILE;
        return -1;
    }
    rec->data[0] = '\0';
    int ret = read(read_fd, rec->data, LIKWID_FREQUENCY_MAX_DATA_LENGTH);
    if (ret < 0)
    {
        rec->data[0] = '\0';
        rec->errorcode = FREQ_ERR_NOPERM;
    }
    else
    {
        rec->errorcode = FREQ_ERR_NONE;
        rec->data[ret] = '\0';
    }
    return 0;
}

static int freq_write(FreqDataRecord *rec)
{
    int write_fd = -1;
    int cpu = rec->cpu;
    struct cpufreq_files* f = &cpufiles[cpu];
    
    switch(rec->loc)
    {
        case FREQ_LOC_CUR:
            write_fd = f->cur_freq;
            syslog(LOG_INFO, "CMD WRITE CPU %d FREQ_LOC_CUR %d", cpu, write_fd);
            break;
        case FREQ_LOC_MIN:
            write_fd = f->min_freq;
            syslog(LOG_INFO, "CMD WRITE CPU %d FREQ_LOC_MIN %d", cpu, write_fd);
            break;
        case FREQ_LOC_MAX:
            write_fd = f->max_freq;
            syslog(LOG_INFO, "CMD WRITE CPU %d FREQ_LOC_MAX %d", cpu, write_fd);
            break;
        case FREQ_LOC_GOV:
            write_fd = f->set_gov;
            syslog(LOG_INFO, "CMD WRITE CPU %d FREQ_LOC_GOV %d", cpu, write_fd);
            break;
        default:
            syslog(LOG_ERR, "Invalid location specified in record\n");
            break;
    }
    if (write_fd < 0)
    {
        syslog(LOG_ERR,"No such file: %s\n", strerror(errno));
        rec->errorcode = FREQ_ERR_NOFILE;
        return -1;
    }
    syslog(LOG_INFO, "FD %d %.*s\n", write_fd, rec->datalen, rec->data);
    int ret = write(write_fd, rec->data, rec->datalen);
    if (ret < 0)
    {
        syslog(LOG_ERR,"No permission: %s\n", strerror(errno));
        rec->errorcode = FREQ_ERR_NOPERM;
        return -1;
    }
    syslog(LOG_ERR,"All good\n");
    rec->errorcode = FREQ_ERR_NONE;
    return 0;
}


/* #####  MAIN FUNCTION DEFINITION   ################## */

int main(void)
{
    int ret;
    pid_t pid;
    FreqDataRecord dRecord;
    struct sockaddr_un  addr1;
    socklen_t socklen;
    mode_t oldumask;
    uint32_t numHWThreads = sysconf(_SC_NPROCESSORS_CONF);
    

    avail_cpus = get_avail_cpus();
    if (avail_cpus < 0)
    {
        avail_cpus = numHWThreads;
    }

    openlog(ident, 0, LOG_USER);

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        stop_daemon();
    }

    daemonize(&pid);
    syslog(LOG_INFO, "FrequencyDaemon runs with UID %d, eUID %d\n", getuid(), geteuid());


    /* setup filename for socket */
    filepath = (char*) calloc(sizeof(addr1.sun_path), 1);
    snprintf(filepath, sizeof(addr1.sun_path), TOSTRING(LIKWIDSOCKETBASE) "-freq-%d", pid);
    syslog(LOG_INFO, "Filepath %s\n", filepath);
    /* get a socket */
    LOG_AND_EXIT_IF_ERROR(sockfd = socket(AF_LOCAL, SOCK_STREAM, 0), socket failed);

    /* initialize socket data structure */
    bzero(&addr1, sizeof(addr1));
    addr1.sun_family = AF_LOCAL;
    strncpy(addr1.sun_path, filepath, (sizeof(addr1.sun_path) - 1)); /* null terminated by the bzero() above! */

    /* Change the file mode mask so only the calling user has access
     * and switch the user/gid with which the following socket creation runs. */
    oldumask = umask(077);
    CHECK_ERROR(setfsuid(getuid()), setfsuid failed);

    /* bind and listen on socket */
    LOG_AND_EXIT_IF_ERROR(bind(sockfd, (SA*) &addr1, sizeof(addr1)), bind failed);
    LOG_AND_EXIT_IF_ERROR(listen(sockfd, 1), listen failed);
    LOG_AND_EXIT_IF_ERROR(chmod(filepath, S_IRUSR|S_IWUSR), chmod failed);

    socklen = sizeof(addr1);

    { /* Init signal handler */
        struct sigaction sia;
        sia.sa_handler = Signal_Handler;
        sigemptyset(&sia.sa_mask);
        sia.sa_flags = 0;
        sigaction(SIGALRM, &sia, NULL);
        sigaction(SIGPIPE, &sia, NULL);
        sigaction(SIGTERM, &sia, NULL);
    }
    
    /* setup an alarm to stop the daemon if there is no connect.*/
    alarm(15U);
    
    if ((connfd = accept(sockfd, (SA*) &addr1, &socklen)) < 0)
    {
        if (errno == EINTR)
        {
            syslog(LOG_ERR, "exiting due to timeout - no client connected after 15 seconds.");
        }
        else
        {
            syslog(LOG_ERR, "accept() failed:  %s", strerror(errno));
        }
        CHECK_ERROR(unlink(filepath), unlink of socket failed);
        exit(EXIT_FAILURE);
    }
    
    alarm(0);
    CHECK_ERROR(unlink(filepath), unlink of socket failed);

    /* Restore the old umask and fs ids. */
    (void) umask(oldumask);
    CHECK_ERROR(setfsuid(geteuid()), setfsuid failed);

    cpufiles = malloc(numHWThreads* sizeof(struct cpufreq_files));
    if (!cpufiles)
    {
        syslog(LOG_ERR,"Failed to allocate space\n");
        stop_daemon();
    }
    for (int i=0;i<avail_cpus;i++)
    {
        memset(&cpufiles[i], -1, sizeof(struct cpufreq_files));
        syslog(LOG_INFO,"Open files for CPU %d\n", i);
        ret = open_cpu(i, &cpufiles[i]);
        if (ret < 0)
        {
            syslog(LOG_ERR,"Failed to open files for CPU %d\n", i);
            stop_daemon();
        }
    }


LOOP:
    syslog(LOG_ERR, "Starting loop %d\n", avail_cpus);
    while (1)
    {
        ret = read(connfd, (void*) &dRecord, sizeof(FreqDataRecord));

        if (ret < 0)
        {
            syslog(LOG_ERR, "ERROR - Read returns %d", ret);
            stop_daemon();
        }
        else if ((ret == 0) && (dRecord.type != FREQ_EXIT))
        {
            syslog(LOG_ERR, "ERROR - [%s:%d] zero read, remote socket closed before reading", __FILE__, __LINE__);
            stop_daemon();
        }
        else if (ret != sizeof(FreqDataRecord))
        {
            syslog(LOG_ERR, "ERROR - [%s:%d] unaligned read", __FILE__, __LINE__);
            stop_daemon();
        }


        if (dRecord.type == FREQ_READ)
        {
            freq_read(&dRecord);
        }
        else if (dRecord.type == FREQ_WRITE)
        {
            freq_write(&dRecord);
        }
        else if (dRecord.type == FREQ_EXIT)
        {
            stop_daemon();
        }
        else
        {
            syslog(LOG_ERR, "unknown daemon command type %d", dRecord.type);
            dRecord.errorcode = FREQ_ERR_UNKNOWN;
        }

        LOG_AND_EXIT_IF_ERROR(write(connfd, (void*) &dRecord, sizeof(FreqDataRecord)), write failed);
    }

    /* never reached */
    return EXIT_SUCCESS;
}

