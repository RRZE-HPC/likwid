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
 *                Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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

#include <types.h>
#include <registers.h>
#include <topology.h>
#include <cpuid.h>
#include <lock.h>

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

#define PCI_ROOT_PATH    "/proc/bus/pci/"
#define MAX_PATH_LENGTH   80
#define MAX_POWER8_REGISTERS 14
#define POWER8_SYSFS_BASE "/sys/devices/system/cpu"
//#define MAX_NUM_NODES    4

/* Lock file controlled from outside which prevents likwid to start.
 * Can be used to synchronize access to the hardware counters
 * with an external monitoring system. */

/* #####   TYPE DEFINITIONS   ########### */

typedef int (*AllowedPrototype)(uint32_t);
typedef int (*AllowedPciPrototype)(PciDeviceType, uint32_t);

typedef struct {
    uint8_t width;
    char*  filename;
    int openflag;
} Power8RegisterInfo;

static Power8RegisterInfo Power8Registers[MAX_POWER8_REGISTERS] = {
    [IBM_MMCR0] = {64, "mmcr0", O_RDWR},
    [IBM_MMCR1] = {64, "mmcr1", O_RDWR},
    [IBM_MMCRA] = {64, "mmcra", O_RDWR},
    [IBM_MMCRC] = {64, "mmcrc", O_RDWR},
    [IBM_PMC0] = {32, "pmc1", O_RDWR},
    [IBM_PMC1] = {32, "pmc2", O_RDWR},
    [IBM_PMC2] = {32, "pmc3", O_RDWR},
    [IBM_PMC3] = {32, "pmc4", O_RDWR},
    [IBM_PMC4] = {32, "pmc5", O_RDWR},
    [IBM_PMC5] = {32, "pmc6", O_RDWR},
    [IBM_PIR] = {32, "pir", O_RDONLY},
    [IBM_PURR] = {64, "purr", O_RDWR},
    [IBM_SPURR] = {64, "spurr", O_RDONLY},
    [IBM_DSCR] = {25, "dscr", O_RDWR},
};


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int sockfd = -1;
static int connfd = -1; /* temporary in to make it compile */
static char* filepath;
static const char* ident = "accessD";
static AllowedPrototype allowed = NULL;
static AllowedPciPrototype allowedPci = NULL;
static int FD_MSR[MAX_NUM_THREADS][MAX_POWER8_REGISTERS];
static int FD_PCI[MAX_NUM_NODES][MAX_NUM_PCI_DEVICES];
static int isPCIUncore = 0;
static PciDevice* pci_devices_daemon = NULL;
static char pci_filepath[MAX_PATH_LENGTH];
static int num_pmc_counters = 0;

/* Socket to bus mapping -- will be determined at runtime;
 * typical mappings are:
 * Socket  Bus (2S)  Bus (4s)
 *   0        0xff      0x3f
 *   1        0x7f      0x7f
 *   2                  0xbf
 *   3                  0xff
 */
static char* socket_bus[MAX_NUM_NODES] = { [0 ... (MAX_NUM_NODES-1)] = NULL};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
allowed_power8(uint32_t reg)
{
    if (reg >= 0 && reg < MAX_POWER8_REGISTERS)
	return 1;
    return 0;
}

static void
msr_read(AccessDataRecord * dRecord)
{
    uint64_t data;
    uint32_t cpu = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    char tmp[100];

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0;

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (FD_MSR[cpu][reg] <= 0)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }

    if (!allowed(reg))
    {
        syslog(LOG_ERR, "Access to register 0x%X not allowed\n", reg);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (pread(FD_MSR[cpu][reg], tmp, 99, 0) == 0)
    {
        syslog(LOG_ERR, "Failed to read data to register 0x%x on core %u", reg, cpu);
        syslog(LOG_ERR, "%s", strerror(errno));
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }
    data = strtoull(tmp, NULL, 16);
    dRecord->data = data;
}

static void
msr_write(AccessDataRecord * dRecord)
{
    uint32_t cpu = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint64_t data = dRecord->data;
    int len = 0;
    char tmp[100];

    dRecord->errorcode = ERR_NOERROR;

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (FD_MSR[cpu][reg] <= 0)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (Power8Registers[reg].openflag == O_RDONLY)
    {
	dRecord->errorcode = ERR_NOERROR;
        return;
    }
    if (!allowed(reg))
    {
        syslog(LOG_ERR, "Attempt to write to restricted register 0x%x on core %u", reg, cpu);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    len = snprintf(tmp, 99, "%lx", data);
    tmp[len] = '\0';
    if (pwrite(FD_MSR[cpu][reg], tmp, 100, 0) == 0)
    {
        syslog(LOG_ERR, "Failed to write data to register 0x%x on core %u", reg, cpu);
        syslog(LOG_ERR, "%s", strerror(errno));
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }
}

static void
msr_check(AccessDataRecord * dRecord)
{
    uint32_t cpu = dRecord->cpu;
    dRecord->errorcode = ERR_NOERROR;

    if (FD_MSR[cpu][0] < 0)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    return;
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
    for (int i=0;i<MAX_NUM_NODES;i++)
    {
        if (socket_bus[i] != NULL)
        {
            free(socket_bus[i]);
        }
    }

    if (sockfd != -1)
    {
        CHECK_ERROR(close(sockfd), socket close sockfd failed);
    }

    free(filepath);
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
    uint32_t model = 0x0;
    uint32_t family = PPC_FAMILY;
    for (int i=0;i<MAX_NUM_THREADS;i++)
    {
	for (int d=0;d<MAX_POWER8_REGISTERS;d++)
            FD_MSR[i][d] = -1;
    }

    openlog(ident, 0, LOG_USER);

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        stop_daemon();
    }

    daemonize(&pid);

    FILE *fpipe = NULL;
    char buff[256];
    if ( !(fpipe = (FILE*)popen("grep POWER8 /proc/cpuinfo | wc -l","r")) )
    {  // If fpipe is NULL
        return 0;
    }
    ret = fread(buff, 1, 255, fpipe);
    buff[ret] = '\0';
    if (ret > 0)
    {
	model = POWER8;
    }
    if (pclose(fpipe))
        return 0;
    if (model == 0x0)
    {
	if ( !(fpipe = (FILE*)popen("grep POWER7 /proc/cpuinfo | wc -l","r")) )
	{  // If fpipe is NULL
	    return 0;
	}
	ret = fread(buff, 1, 255, fpipe);
	buff[ret] = '\0';
	if (ret > 0)
	{
	    model = POWER7;
	}
	if (pclose(fpipe))
	    return 0;
    }
    if (model == 0x0)
    {
	syslog(LOG_ERR, "ERROR - [%s:%d] - Unsupported processor. Exiting!  \n",
	                __FILE__, __LINE__);
	exit(EXIT_FAILURE);
    }
    allowed = allowed_power8;

    /* setup filename for socket */
    filepath = (char*) calloc(sizeof(addr1.sun_path), 1);
    snprintf(filepath, sizeof(addr1.sun_path), "/tmp/likwid-%d", pid);

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

    {
        char* msr_file_name = (char*) malloc(MAX_PATH_LENGTH * sizeof(char));

        /* Open MSR device files for less overhead.
         * NOTICE: This assumes consecutive processor Ids! */
        for ( uint32_t i=0; i < numHWThreads; i++ )
        {
	    for (uint32_t d=0; d < MAX_POWER8_REGISTERS; d++)
	    { 
		sprintf(msr_file_name,"%s/cpu%d/%s",POWER8_SYSFS_BASE, i, Power8Registers[d].filename);
		FD_MSR[i][d] = open(msr_file_name, Power8Registers[d].openflag);

		if ( FD_MSR[i][d] < 0 )
		{
		    syslog(LOG_ERR, "Failed to open device file %s: %s", msr_file_name, strerror(errno));
                }
            }
        }

        free(msr_file_name);
    }
    syslog(LOG_ERR, "Start loop");
LOOP:
    while (1)
    {
        ret = read(connfd, (void*) &dRecord, sizeof(AccessDataRecord));

        if (ret < 0)
        {
            stop_daemon();
        }
        else if ((ret == 0) && (dRecord.type != DAEMON_EXIT))
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
            msr_read(&dRecord);
        }
        else if (dRecord.type == DAEMON_WRITE)
        {
            msr_write(&dRecord);
            dRecord.data = 0x0ULL;
        }
        else if (dRecord.type == DAEMON_CHECK)
        {
            msr_check(&dRecord);
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

        LOG_AND_EXIT_IF_ERROR(write(connfd, (void*) &dRecord, sizeof(AccessDataRecord)), write failed);
    }

    /* never reached */
    return EXIT_SUCCESS;
}

