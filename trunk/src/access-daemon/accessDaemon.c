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
 *      Copyright (C) 2013 Jan Treibig 
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
#include <lock.h>
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
#define  K16_FAMILY       0x16U

#define SANDYBRIDGE          0x2AU
#define SANDYBRIDGE_EP       0x2DU
#define IVYBRIDGE            0x3AU
#define IVYBRIDGE_EP         0x3EU
#define HASWELL              0x3CU
#define ATOM_SILVERMONT      0x4DU

#define PCI_ROOT_PATH    "/proc/bus/pci/"
#define MAX_PATH_LENGTH   60
#define MAX_NUM_NODES    4

/* Lock file controlled from outside which prevents likwid to start.
 * Can be used to synchronize access to the hardware counters
 * with an external monitoring system. */

/* #####   TYPE DEFINITIONS   ########### */
typedef int (*FuncPrototype)(uint32_t);

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
static int sockfd = -1;
static int connfd = -1; /* temporary in to make it compile */
static char* filepath;
static const char* ident = "accessD";
static FuncPrototype allowed = NULL;
static int FD_MSR[MAX_NUM_THREADS];
static int FD_PCI[MAX_NUM_NODES][MAX_NUM_DEVICES];
static int isPCIUncore = 0;

static char* pci_DevicePath[MAX_NUM_DEVICES] = {
    "13.5", "13.6", "13.1", "10.0", "10.1", "10.4",
    "10.5", "0e.1", "08.0", "09.0", "08.6", "09.6",
    "08.0", "09.0" };
static char pci_filepath[MAX_PATH_LENGTH];

/* Socket to bus mapping -- will be determined at runtime;
 * typical mappings are:
 * Socket  Bus (2S)  Bus (4s)
 *   0        0xff      0x3f
 *   1        0x7f      0x7f
 *   2                  0xbf
 *   3                  0xff
 */
static char* socket_bus[MAX_NUM_NODES];


/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int allowed_intel(uint32_t reg)
{
    if ( ((reg & 0x0F8U) == 0x0C0U) ||
            ((reg & 0xFF0U) == 0x180U) ||
            ((reg & 0xF00U) == 0x300U) ||
            ((reg & 0xF00U) == 0xC00U) ||
            ((reg & 0xF00U) == 0xD00U) ||
            ((reg & 0xF00U) == 0xE00U) ||
            (reg == 0x1A0)  ||
            (reg == 0x0CE)  ||
            (reg == 0x19C)  ||
            (reg == 0x1A2)  ||
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
    if ((allowed_intel(reg)) ||
        (((reg & 0xF00U) == 0x600U) ||
        (reg == 0x1AD)))
    {
        return 1;
    }
    return 0;
}

static int allowed_silvermont(uint32_t reg)
{
    if (!allowed_sandybridge(reg))
    {
        return 0;
    }

    if ( ((reg & 0x0F8U) == 0x0C0U) ||
            ((reg & 0xFF0U) == 0x180U) ||
            ((reg & 0xF00U) == 0x300U) ||
            ((reg & 0xF00U) == 0x600U) ||
            ((reg & 0xF00U) == 0xC00U) ||
            ((reg & 0xF00U) == 0xD00U) ||
            (reg == 0x1A0)  ||
            (reg == 0x0CE)  ||
            (reg == 0x1AD)  ||
            (reg == 0x19C)  ||
            (reg == 0x1A2)  ||
            (reg == 0x1A6) ||
            (reg != 0x1A7))
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

static int allowed_amd16(uint32_t reg)
{
    if ( ((reg & 0xFFFFFFF0U) == 0xC0010000U) ||
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
        syslog(LOG_ERR, "Attempt to read to restricted register 0x%x on core %u", reg, cpu);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (pread(FD_MSR[cpu], &data, sizeof(data), reg) != sizeof(data))
    {
        syslog(LOG_ERR, "Failed to read data to register 0x%x on core %u", reg, cpu);
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
        syslog(LOG_ERR, "Attempt to write to restricted register 0x%x on core %u", reg, cpu);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (pwrite(FD_MSR[cpu], &data, sizeof(data), reg) != sizeof(data))
    {
        syslog(LOG_ERR, "Failed to write data to register 0x%x on core %u", reg, cpu);
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
            syslog(LOG_ERR, "Failed to open device file %s for device %u on socket %u", pci_filepath, device, socketId);
            dRecord->errorcode = ERR_OPENFAIL;
            return;
        }
    }

    if ( pread(FD_PCI[socketId][device], &data, sizeof(data), reg) != sizeof(data)) 
    {
        syslog(LOG_ERR, "Failed to read data from pci device file %s for device %u on socket %u",
                pci_filepath,device,socketId);
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
            syslog(LOG_ERR, "Failed to open device file %s for device %u on socket %u", pci_filepath, device, socketId);
            dRecord->errorcode = ERR_OPENFAIL;
            return;
        }
    }

    if (pwrite(FD_PCI[socketId][device], &data, sizeof data, reg) != sizeof data) 
    {
        syslog(LOG_ERR, "Failed to write data to pci device file %s for device %u on socket %u",pci_filepath, device, socketId);
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
        stop_daemon();
    }

    /* For SIGALRM we just return - we're just here to create a EINTR */
    if ((sig == SIGTERM))
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

int main(void)
{
    int ret;
    pid_t pid;
    struct sockaddr_un  addr1;
    socklen_t socklen;
    AccessDataRecord dRecord;
    mode_t oldumask;
    uint32_t numHWThreads = sysconf(_SC_NPROCESSORS_CONF);
    uint32_t model;

    openlog(ident, 0, LOG_USER);

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        stop_daemon();
    }

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
        model  = (((eax >> 16) & 0xFU) << 4) + ((eax >> 4) & 0xFU);

        switch (family)
        {
            case P6_FAMILY:
                allowed = allowed_intel;

                if ((model == SANDYBRIDGE)        ||
                        (model == SANDYBRIDGE_EP) ||
                        (model == IVYBRIDGE)      ||
                        (model == IVYBRIDGE_EP) )
                {
                    allowed = allowed_sandybridge;
                    isPCIUncore = 1;
                }
                else if (model == HASWELL)
                {
                    allowed = allowed_sandybridge;
                }
                else if (model == ATOM_SILVERMONT)
                {
                    allowed = allowed_silvermont;
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
            case K16_FAMILY:
                allowed = allowed_amd16;
	        break;
            default:
                syslog(LOG_ERR, "ERROR - [%s:%d] - Unsupported processor. Exiting!  \n",
                        __FILE__, __LINE__);
                exit(EXIT_FAILURE);
        }
    }

    /* setup filename for socket */
    filepath = (char*) calloc(sizeof(addr1.sun_path), 1);
    snprintf(filepath, sizeof(addr1.sun_path), "/tmp/likwid-%d", pid);

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

        if (isPCIUncore)
        {
            for (int j=0; j<MAX_NUM_NODES; j++)
            {
                for (int i=0; i<MAX_NUM_DEVICES; i++)
                {
                    FD_PCI[j][i] = 0;
                }
            }

            /* determine PCI-BUSID mapping ... */
            FILE *fptr;
            char buf[1024];
            uint32_t testDevice;
            uint32_t sbus, sdevfn, svend;
            int cntr = 0;

            for ( int i=0; i<MAX_NUM_NODES; i++ )
            {
                socket_bus[i] = "N-A";
            }

            if (model == SANDYBRIDGE_EP)
            {
                testDevice = 0x80863c44;
            }
            else if (model == IVYBRIDGE_EP)
            {
                testDevice = 0x80860e36;
            }
            else
            {
                testDevice = 0;
                syslog(LOG_NOTICE, "PCI Uncore not supported on this system");
            }

            if ( ((fptr = fopen("/proc/bus/pci/devices", "r")) == NULL) || !testDevice)
            {
                syslog(LOG_NOTICE, "Unable to open /proc/bus/pci/devices");
            }
            else
            {
                while( fgets(buf, sizeof(buf)-1, fptr) )
                {
                    if ( sscanf(buf, "%2x%2x %8x", &sbus, &sdevfn, &svend) == 3 &&
                            svend == testDevice )
                    {
                        socket_bus[cntr] = (char*)malloc(4);
                        sprintf(socket_bus[cntr++], "%02x/", sbus);
                    }
                }
                fclose(fptr);
            }

            if ( cntr == 0 )
            {
                syslog(LOG_NOTICE, "Uncore not supported on this system");
            }
        }
    }

    while (1)
    {
        ret = read(connfd, (void*) &dRecord, sizeof(AccessDataRecord));

        if (ret < 0)
        {
            syslog(LOG_ERR, "ERROR - [%s:%d] read from client failed  - %s \n",
                    __FILE__, __LINE__, strerror(errno));
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

        EXIT_IF_ERROR(write(connfd, (void*) &dRecord, sizeof(AccessDataRecord)), write failed);
    }

    /* never reached */
    return EXIT_SUCCESS;
}
