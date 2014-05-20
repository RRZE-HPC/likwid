/*
 * =======================================================================================
 *
 *      Filename:  accessDaemon.c
 *
 *      Description:  Implementation of access daemon.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Michael Meier, michael.meier@rrze.fau.de
 *                Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

#include <pci_types.h>
#include <accessClient_types.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define SA struct sockaddr
#define str(x) #x

#define CHECK_ERROR(func, msg)  \
    if ((func) < 0) { syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); }

#define CHECK_FILE_ERROR(func, msg)  \
    if ((func) == 0) { syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); }


#define EXIT_IF_ERROR(func, msg)  \
    if ((func) < 0) { syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); stop_daemon(); exit(EXIT_FAILURE); }


#define CPUID                    \
    __asm__ volatile ("cpuid"    \
            : "=a" (eax),            \
            "=b" (ebx)             \
            : "0" (eax))


#define  P6_FAMILY        0x6U
#define  K8_FAMILY        0xFU
#define  K10_FAMILY       0x10U
#define  K15_FAMILY       0x15U

#define SANDYBRIDGE          0x2AU
#define SANDYBRIDGE_EP       0x2DU

#define PCI_ROOT_PATH    "/proc/bus/pci/"
#define MAX_PATH_LENGTH   60
#define MAX_NUM_NODES    4

/* #####   TYPE DEFINITIONS   ########### */
typedef int (*FuncPrototype)(uint32_t);

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
static int sockfd = -1;
static int CONNFD[MAX_NUM_THREADS];
static int connfd = -1; /* temporary in to make it compile */
static pid_t lockID=0;
static char* filepath;
static const char* ident = "accessD";
static FuncPrototype allowed = NULL;
static int sysdaemonmode = 0;
static int FD_MSR[MAX_NUM_THREADS];
static int FD_PCI[MAX_NUM_NODES][MAX_NUM_DEVICES];
static int isSandyBridge = 0;

static char* pci_DevicePath[MAX_NUM_DEVICES] = {
    "13.5", "13.6", "13.1", "10.0", "10.1", "10.4",
    "10.5", "0e.1", "08.0", "09.0", "08.6", "09.6",
    "08.0", "09.0" };
static char pci_filepath[MAX_PATH_LENGTH];

/* Socket to bus mapping
 * Socket  Bus (2S)  Bus (4s)
 *   0        0xff      0x3f
 *   1        0x7f      0x7f
 *   2                  0xbf
 *   3                  0xff
 */

static char* socket_bus[] = { "7f/", "ff/",  "bf/",  "ff/" };

#ifndef ALLOWSYSDAEMONMODE
#define ALLOWSYSDAEMONMODE 1
#endif
#ifndef SYSDAEMONSOCKETPATH
#define SYSDAEMONSOCKETPATH "/var/run/likwid-msrd.sock"
#endif

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int allowed_intel(uint32_t reg)
{
    if ( ((reg & 0x0F8U) == 0x0C0U) ||
            ((reg & 0xFF0U) == 0x180U) ||
            ((reg & 0xF00U) == 0x300U) ||
            (reg == 0x1A0)  ||
            (reg == 0x0CE)  ||
            (reg == 0x1AD)  ||
            (reg == 0x1A6))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int allowed_sandybridge(uint32_t reg)
{
    if ( ((reg & 0x0F8U) == 0x0C0U) ||
            ((reg & 0xFF0U) == 0x180U) ||
            ((reg & 0xF00U) == 0x300U) ||
            ((reg & 0xF00U) == 0x600U) ||
            ((reg & 0xF00U) == 0xC00U) ||
            ((reg & 0xF00U) == 0xD00U) ||
            (reg == 0x1A0)  ||
            (reg == 0x0CE)  ||
            (reg == 0x1AD)  ||
            (reg == 0x1A6))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


static int allowed_amd(uint32_t reg)
{
    if ( (reg & 0xFFFFFFF0U) == 0xC0010000U)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int allowed_amd15(uint32_t reg)
{
    if ( ((reg & 0xFFFFFFF0U) == 0xC0010000U) ||
            ((reg & 0xFFFFFFF0U) == 0xC0010200U) ||
            ((reg & 0xFFFFFFF8U) == 0xC0010240U))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static void msr_read(AccessDataRecord * dRecord)
{
    uint64_t data;
    uint32_t cpu = dRecord->cpu;
    uint32_t reg = dRecord->reg;

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0;

    if (!allowed(reg))
    {
        syslog(LOG_ERR, "attempt to read from restricted register %x", reg);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (pread(FD_MSR[cpu], &data, sizeof(data), reg) != sizeof(data)) 
    {
        syslog(LOG_ERR, "Failed to read data from msr device file on core %u", cpu);
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }

    dRecord->data = data;
}

static void msr_write(AccessDataRecord * dRecord)
{
    uint32_t cpu = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint64_t data = dRecord->data;

    dRecord->errorcode = ERR_NOERROR;

    if (!allowed(reg))
    {
        syslog(LOG_ERR, "attempt to write to restricted register %x", reg);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (pwrite(FD_MSR[cpu], &data, sizeof data, reg) != sizeof(data)) 
    {
        syslog(LOG_ERR, "Failed to write data from msr device file on core %u", cpu);
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }
}

static void pci_read(AccessDataRecord* dRecord)
{
    uint32_t socketId = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint32_t device = dRecord->device;
    uint32_t data;

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0;

    if ( !FD_PCI[socketId][device] )
    {
        strncpy(pci_filepath, PCI_ROOT_PATH, 30);
        strncat(pci_filepath, socket_bus[socketId], 10);
        strncat(pci_filepath, pci_DevicePath[device], 20);

        FD_PCI[socketId][device] = open( pci_filepath, O_RDWR);

        if ( FD_PCI[socketId][device] < 0)
        {
            syslog(LOG_ERR, "Failed to open device file on socket %u", socketId);
            dRecord->errorcode = ERR_OPENFAIL;
            return;
        }
    }

    if ( pread(FD_PCI[socketId][device], &data, sizeof(data), reg) != sizeof(data)) 
    {
        syslog(LOG_ERR, "Failed to read data from pci device file on socket %u", socketId);
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }
    //    printf("READ Device %s cpu %d reg 0x%x data 0x%x \n",bdata(filepath), cpu, reg, data);

    dRecord->data = (uint64_t) data;
}



static void pci_write(AccessDataRecord* dRecord)
{
    uint32_t socketId = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint32_t device = dRecord->device;
    uint32_t data = (uint32_t) dRecord->data;

    dRecord->errorcode = ERR_NOERROR;

    if ( !FD_PCI[socketId][device] )
    {
        strncpy(pci_filepath, PCI_ROOT_PATH, 30);
        strncat(pci_filepath, socket_bus[socketId], 10);
        strncat(pci_filepath, pci_DevicePath[device], 20);

        FD_PCI[socketId][device] = open( pci_filepath, O_RDWR);

        if ( FD_PCI[socketId][device] < 0)
        {
            syslog(LOG_ERR, "Failed to open device file on socket %u", socketId);
            dRecord->errorcode = ERR_OPENFAIL;
            return;
        }
    }

    if (pwrite(FD_PCI[socketId][device], &data, sizeof data, reg) != sizeof data) 
    {
        syslog(LOG_ERR, "Failed to write data from pci device file on socket %u", socketId);
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }
}


static void kill_client(void)
{
    syslog(LOG_NOTICE, "daemon dropped client");

    if (connfd != -1) 
    {
        CHECK_ERROR(close(connfd), socket close failed);
    }

    connfd = -1;
}

static void stop_daemon(void)
{
    kill_client();
    syslog(LOG_NOTICE, "daemon exiting");

    if (sockfd != -1) 
    {
        CHECK_ERROR(close(sockfd), socket close sockfd failed);

        if (sysdaemonmode) 
        {
            CHECK_ERROR(unlink(filepath), unlink of socket failed);
        }
    }

    free(filepath);
    closelog();
    exit(EXIT_SUCCESS);
}

static void Signal_Handler(int sig)
{
    if (sig == SIGPIPE)
    {
        syslog(LOG_NOTICE, "SIGPIPE? client crashed?!");

        if (sysdaemonmode) 
        {
            kill_client();
        }
        else
        {
            stop_daemon();
        }
    }

    /* For SIGALRM we just return - we're just here to create a EINTR */
    if ((sig == SIGTERM) && (sysdaemonmode))
    {
        stop_daemon();
    }
}

static void daemonize(int* parentPid)
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

/* #####  MAIN FUNCTION DEFINITION   ################## */

int main(int argc, char ** argv) 
{
    int ret;
    pid_t pid;
    struct sockaddr_un  addr1;
    socklen_t socklen;
    AccessDataRecord dRecord;
    mode_t oldumask;
    static struct option long_options[] = {
        { "help",      0, 0, '?' },
        { "sysdaemon", 0, 0, 's' },
        { 0, 0, 0, 0 }
    };
    int c;
    uint32_t numHWThreads = sysconf(_SC_NPROCESSORS_CONF);

    while ((c = getopt_long(argc, argv, "h?", long_options, NULL)) > 0) {
        switch (c) {
            case '?':
            case 'h':
                printf("Syntax: %s --sysdaemon\n", argv[0]);
                printf("You should NOT manually call this program unless you");
                printf(" want to run it as a system daemon.\n");
                exit(EXIT_SUCCESS);
                break;
            case 's':
#if (ALLOWSYSDAEMONMODE == 1)
                if (getuid() != 0) {
                    printf("Sorry, only root is allowed to call sysdaemonmode.\n");
                    exit(EXIT_FAILURE);
                }
                sysdaemonmode = 1;
#else
                printf("sysdaemonmode has been disallowed at compiletime.\n");
                exit(EXIT_FAILURE);
#endif
                break;
            default:
                fprintf(stderr, "Invalid option. try -h for help.\n");
                exit(EXIT_FAILURE);
        };
    }

    openlog(ident, 0, LOG_USER);
    daemonize(&pid);

    {
        uint32_t  eax = 0x00;
        uint32_t  ebx = 0x00;
        int isIntel = 1;
        CPUID;
        if (ebx == 0x68747541U)
        {
            isIntel = 0;
        }

        eax = 0x01;
        CPUID;
        uint32_t family = ((eax >> 8) & 0xFU) + ((eax >> 20) & 0xFFU);
        uint32_t model  = (((eax >> 16) & 0xFU) << 4) + ((eax >> 4) & 0xFU);

        switch (family)
        {
            case P6_FAMILY:
                allowed = allowed_intel;

                if ((model == SANDYBRIDGE) || (model == SANDYBRIDGE_EP))
                {
                    allowed = allowed_sandybridge;
                    isSandyBridge = 1;
                }
                break;
            case K8_FAMILY:
                if (isIntel)
                {
                    syslog(LOG_ERR,
                            "ERROR - [%s:%d] - Netburst architecture is not supported! Exiting! \n",
                            __FILE__,__LINE__);
                    exit(EXIT_FAILURE);
                }
            case K10_FAMILY:
                allowed = allowed_amd;
                break;
            case K15_FAMILY:
                allowed = allowed_amd15;
                break;
            default:
                syslog(LOG_ERR, "ERROR - [%s:%d] - Unsupported processor. Exiting!  \n", __FILE__, __LINE__);
                exit(EXIT_FAILURE);
        }
    }

    /* setup filename for socket */
    filepath = (char*) calloc(sizeof(addr1.sun_path), 1);
    if (sysdaemonmode == 1) {
        snprintf(filepath, sizeof(addr1.sun_path), "%s", SYSDAEMONSOCKETPATH);
    } else {
        snprintf(filepath, sizeof(addr1.sun_path), "/tmp/likwid-%d", pid);
    }

    /* get a socket */
    EXIT_IF_ERROR(sockfd = socket(AF_LOCAL, SOCK_STREAM, 0), socket failed);

    /* initialize socket data structure */
    bzero(&addr1, sizeof(addr1));
    addr1.sun_family = AF_LOCAL;
    strncpy(addr1.sun_path, filepath, (sizeof(addr1.sun_path) - 1)); /* null terminated by the bzero() above! */

    /* Change the file mode mask so only the calling user has access
     * and switch the user/gid with which the following socket creation runs. */
    oldumask = umask(077);
    CHECK_ERROR(setfsuid(getuid()), setfsuid failed);

    /* bind and listen on socket */
    EXIT_IF_ERROR(bind(sockfd, (SA*) &addr1, sizeof(addr1)), bind failed);
    EXIT_IF_ERROR(listen(sockfd, 1), listen failed);
    EXIT_IF_ERROR(chmod(filepath, S_IRUSR|S_IWUSR), chmod failed);

    /* Restore the old umask and fs ids. */
    (void) umask(oldumask);
    CHECK_ERROR(setfsuid(geteuid()), setfsuid failed);

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

    syslog(LOG_NOTICE, "daemon started");

    if (sysdaemonmode == 0) 
    {
        /* The normal case: without sysdaemon mode. We only ever accept one client. */
        /* accept one connect from one client */

        /* setup an alarm to stop the daemon if there is no connect.
         * or to change to different metrics in sysdaemonmode. */
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
        syslog(LOG_NOTICE, "daemon accepted client");

        {
            char* msr_file_name = (char*) malloc(MAX_PATH_LENGTH * sizeof(char));

            /* Open MSR device files for less overhead.
             * NOTICE: This assumes consecutive processor Ids! */
            for ( uint32_t i=0; i < numHWThreads; i++ )
            {
                sprintf(msr_file_name,"/dev/cpu/%d/msr",i);
                FD_MSR[i] = open(msr_file_name, O_RDWR);

                if ( FD_MSR[i] < 0 )
                {
                    syslog(LOG_ERR, "Failed to open device files.");
                }
            }

            free(msr_file_name);

            if (isSandyBridge)
            {
                for (int j=0; j<MAX_NUM_NODES; j++)
                {
                    for (int i=0; i<MAX_NUM_DEVICES; i++)
                    {
                        FD_PCI[j][i] = 0;
                    }
                }

                /* TODO Exten to 4 socket systems */
#if 0
                if ( cpuid_topology.numSockets == 2 )
                {
                    /* Already correctly initialized */
                }
                else if ( cpuid_topology.numSockets == 4 )
                {
                    strcpy(socket_bus[1],"3f/");
                }
                else 
                {
                    /*TODO Check devices on single socket variants!! */
                    syslog(LOG_NOTICE, "Uncore currently not supported for single socket systems");
                }
#endif
            }
        }

        while (1)
        {
            ret = read(connfd, (void*) &dRecord, sizeof(AccessDataRecord));

            if (ret < 0)
            { 
                syslog(LOG_ERR, "ERROR - [%s:%d] read from client failed  - %s \n", __FILE__, __LINE__, strerror(errno));
                stop_daemon();
            }
            else if (ret == 0)
            {
                syslog(LOG_ERR, "ERROR - [%s:%d] zero read", __FILE__, __LINE__);
                stop_daemon();
            }
            else if (ret != sizeof(AccessDataRecord))
            {
                syslog(LOG_ERR, "ERROR - [%s:%d] unaligned read", __FILE__, __LINE__);
                stop_daemon();
            }


            if (dRecord.type == DAEMON_READ)
            {
                if (dRecord.device == DAEMON_AD_MSR)
                {
                    msr_read(&dRecord);
                }
                else
                {
                    pci_read(&dRecord);
                }
            }
            else if (dRecord.type == DAEMON_WRITE)
            {
                if (dRecord.device == DAEMON_AD_MSR)
                {
                    msr_write(&dRecord);
                    dRecord.data = 0x0ULL;
                }
                else
                {
                    pci_write(&dRecord);
                    dRecord.data = 0x0ULL;
                }
            }
            else if (dRecord.type == DAEMON_EXIT)
            {
                stop_daemon();
            }
            else
            {
                syslog(LOG_ERR, "unknown daemon access type  %d", dRecord.type);
                dRecord.errorcode = ERR_UNKNOWN;
            }

            //    syslog(LOG_NOTICE, "write: cpu %d reg 0x%x data 0x%x type %d device %d error %d \n",
            //	    dRecord.cpu, dRecord.reg, dRecord.data, dRecord.type, dRecord.device, dRecord.errorcode);

            EXIT_IF_ERROR(write(connfd, (void*) &dRecord, sizeof(AccessDataRecord)), write failed);
        }
    } else {  /* Sysdaemonmode. */
        /* Things are slightly more complicated in sysdaemonmode: We can accept
         * multiple clients, and clients can preemt lower priority clients.
         * While this does share quite a bit of copy&paste with above, it is
         * seperated intentionally, to keep the code for the setuid use case
         * above as simple as possible. */
        int haveclient = 0;
        int clientprio = 0;
        int numberOfConnects = 0;
        fd_set fds;

        while (1)
        {
            FD_ZERO(&fds);
            FD_SET(sockfd, &fds);

            if (haveclient) 
            {
                for (int i=0; i<(numberOfConnects); i++)
                {
                    if (CONNFD[i]) 
                    {
                        FD_SET(CONNFD[i], &fds);
                    }
                }
            }

            if (select(FD_SETSIZE, &fds, NULL, NULL, NULL) < 0) 
            {
                syslog(LOG_ERR, "exiting due to error on select().");
                exit(EXIT_FAILURE);
            }

            if (haveclient) 
            {
                if (FD_ISSET(connfd, &fds)) /* data (or an error) from our client */
                {
                    ret = read(connfd, (void*) &dRecord, sizeof(AccessDataRecord));

                    if (ret < 0)
                    {
                        syslog(LOG_ERR, "ERROR - [%s:%d] read from client failed  - %s \n", __FILE__, __LINE__, strerror(errno));
                        kill_client();
                        haveclient = 0;
                    }
                    else if (ret == 0)
                    {
                        syslog(LOG_ERR, "ERROR - [%s:%d] zero read", __FILE__, __LINE__);
                        kill_client();
                        haveclient = 0;
                    }
                    else if (ret != sizeof(AccessDataRecord))
                    {
                        syslog(LOG_ERR, "ERROR - [%s:%d] unaligned read", __FILE__, __LINE__);
                        kill_client();
                        haveclient = 0;
                    }

                    if (dRecord.type == DAEMON_READ)
                    {
                        msr_read(&dRecord);
                    }
                    else if (dRecord.type == DAEMON_WRITE)
                    {
                        msr_write(&dRecord);
                        dRecord.data = 0x0ULL;
                    }
                    else if (dRecord.type == DAEMON_MARK_CLIENT_LOWPRIO)
                    {
                        clientprio = 20;
                    }
                    else if (dRecord.type == DAEMON_EXIT)
                    {
                        kill_client();
                        haveclient = 0;
                        clientprio = 0;
                    }
                    else
                    {
                        syslog(LOG_ERR, "unknown msr access type  %d", dRecord.type);
                        dRecord.errorcode = ERR_UNKNOWN;
                    }
                    if (connfd > 0) /* Still >0? we might have lost the client above. */
                    { 
                        if (write(connfd, (void*) &dRecord, sizeof(AccessDataRecord)) < 0)
                        {
                            syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno));
                            kill_client();
                        }
                    }
                }
            }

            if (FD_ISSET(sockfd, &fds)) /* new connection */
            { 
                int newconnfd = accept(sockfd, (SA*) &addr1, &socklen);

                if (newconnfd < 0) 
                {
                    syslog(LOG_ERR, "error on accept() - ignoring");
                }
                else 
                {
                    if (haveclient) /* Check if the old client can be dropped */
                    { 
                        if (clientprio > 0)
                        {
                            kill_client();
                            /* FIXME reset MSR state? */
                            connfd = newconnfd;
                            clientprio = 0;
                        }
                        else
                        { /* it cannot, so drop the new client. */
                            memset(&dRecord, 0, sizeof(dRecord));
                            dRecord.errorcode = ERR_DAEMONBUSY;
                            EXIT_IF_ERROR(write(newconnfd, (void*) &dRecord, sizeof(AccessDataRecord)), write failed);
                            EXIT_IF_ERROR(close(newconnfd), close failed);
                        }
                    } 
                    else
                    {
                        connfd = newconnfd;
                        haveclient = 1;
                        clientprio = 0;
                    }
                }
            }
        }
    }

    /* never reached */
    return EXIT_SUCCESS;
}
