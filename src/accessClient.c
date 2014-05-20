/*
 * =======================================================================================
 *
 *      Filename:  accessClient.c
 *
 *      Description:  Implementation of client to the access daemon.
 *                   Provides API to read and write values to MSR or
 *                   PCI Cfg Adresses. This module is used by the 
 *                   msr and pci modules.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
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
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <types.h>
#include <error.h>
#include <cpuid.h>
#include <accessClient.h>

int accessClient_mode = ACCESSMODE;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
//static char* msr_strerror(AccessErrorType met);
static int accesslowprio = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
static char* accessClient_strerror(AccessErrorType det)
{
    switch (det)
    {
        case ERR_NOERROR:    return "No error";
        case ERR_UNKNOWN:    return "unknown command";
        case ERR_RESTREG:    return "access to this register is not allowed";
        case ERR_OPENFAIL:   return "failed to open device file";
        case ERR_RWFAIL:     return "failed to read/write register";
        case ERR_DAEMONBUSY: return "daemon already has a same/higher priority client";
        default:             return "UNKNOWN errorcode";
    }
}

static int startDaemon(void)
{
    /* Check the function of the daemon here */
    char* filepath;
    char *newargv[] = { NULL };
    char *newenv[] = { NULL };
    char *exeprog = TOSTRING(ACCESSDAEMON);
    struct sockaddr_un address;
    size_t address_length;
    int  ret;
    pid_t pid;
    int timeout = 10;
    int socket_fd = -1;

    if (accessClient_mode == DAEMON_AM_ACCESS_D)
    {
        pid = fork();

        if (pid == 0)
        {
            ret = execve (exeprog, newargv, newenv);

            if (ret < 0)
            {
                ERRNO_PRINT;
                fprintf(stderr, "Failed to execute the daemon '%s' (see error above)\n", exeprog);
                exit(EXIT_FAILURE);
            }
        }
        else if (pid < 0)
        {
            ERROR_PLAIN_PRINT(Failed to fork);
        }
    }

    EXIT_IF_ERROR(socket_fd = socket(AF_LOCAL, SOCK_STREAM, 0), socket() failed);

    address.sun_family = AF_LOCAL;
    address_length = sizeof(address);

    if (accessClient_mode == DAEMON_AM_SYSACCESS_D)
    {
        snprintf(address.sun_path, sizeof(address.sun_path), "%s", TOSTRING(SYSDAEMONSOCKETPATH));
    }
    else
    {
        snprintf(address.sun_path, sizeof(address.sun_path), "/tmp/likwid-%d", pid);
    }

    filepath = strdup(address.sun_path);
    DEBUG_PRINT(0, "%ssocket pathname is %s\n",
            ((accessClient_mode == DAEMON_AM_ACCESS_D) ? "Generated " : ""),
            filepath);

    while (timeout > 0)
    {
        int res;
        sleep(1);
        res = connect(socket_fd, (struct sockaddr *) &address, address_length);

        if (res == 0)
        {
            break;
        }

        timeout--;
        DEBUG_PRINT(1, "%s\n", "Still waiting for socket...");
    }

    if (timeout <= 0)
    {
        ERRNO_PRINT;  /* should hopefully still work, as we make no syscalls in between. */
        fprintf(stderr, "Exiting due to timeout: The socket file at '%s' could not be opened within 10 seconds.\n", filepath);
        fprintf(stderr, "Consult the error message above this to find out why.\n");
        fprintf(stderr, "If the error is 'no such file or directoy', it usually means that likwid-accessD just failed to start.\n");
        exit(EXIT_FAILURE);
    }

    DEBUG_PRINT(0, "%s\n", "Successfully opened socket to daemon.");
    free(filepath);

    return socket_fd;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
accessClient_setaccessmode(int mode)
{
    if ((accessClient_mode > DAEMON_AM_SYSACCESS_D) || (accessClient_mode < DAEMON_AM_DIRECT)) 
    {
        fprintf(stderr, "Invalid accessmode %d\n", accessClient_mode);
        exit(EXIT_FAILURE);
    }

    accessClient_mode = mode;
}


void
accessClient_setlowaccesspriority()
{
    accesslowprio = 1;
}

void
accessClient_initThread(int* socket_fd)
{
    if ((accessClient_mode == DAEMON_AM_ACCESS_D) ||
            (accessClient_mode == DAEMON_AM_SYSACCESS_D))
    {
        (*socket_fd) = startDaemon();
    }
}

void
accessClient_init(int* socket_fd)
{
    if ((accessClient_mode == DAEMON_AM_ACCESS_D))
    {
        (*socket_fd) = startDaemon();
    }

    if ((accessClient_mode == DAEMON_AM_SYSACCESS_D) && (accesslowprio))
    {
        AccessDataRecord data;

        DEBUG_PRINT(1, "%s\n", "Requesting low priority mode from daemon...");
        memset(&data, 0, sizeof(AccessDataRecord));
        data.type = DAEMON_MARK_CLIENT_LOWPRIO;
        CHECK_ERROR(write((*socket_fd), &data, sizeof(AccessDataRecord)), socket write failed);
        CHECK_ERROR(read((*socket_fd), &data, sizeof(AccessDataRecord)), socket read failed);

        if (data.errorcode != ERR_NOERROR)
        {
            fprintf(stderr, "Failed to request low prio mode from daemon: "
                    "daemon returned error %d '%s'\n",
                    data.errorcode, accessClient_strerror(data.errorcode));
            exit(EXIT_FAILURE);
        }
    }
}

void
accessClient_finalize(int socket_fd)
{
    if ( socket_fd != -1 )
    { /* Only if a socket is actually open */
        AccessDataRecord data;
        data.type = DAEMON_EXIT;
        CHECK_ERROR(write(socket_fd, &data, sizeof(AccessDataRecord)),socket write failed);
        CHECK_ERROR(close(socket_fd),socket close failed);
    }
}


uint64_t 
accessClient_read(int socket_fd, const int cpu, const int device, uint32_t reg)
{
    AccessDataRecord data;

    data.cpu = cpu;
    data.reg = reg;
    data.data = 0x00;
    data.type = DAEMON_READ;
    data.device = device;

    CHECK_ERROR(write(socket_fd, &data, sizeof(AccessDataRecord)), socket write failed);
    CHECK_ERROR(read(socket_fd, &data, sizeof(AccessDataRecord)), socket read failed);

    if (data.errorcode != ERR_NOERROR)
    {
        fprintf(stderr, "Failed to read data through daemon: "
                "daemon returned error %d '%s' for cpu %d reg %x\n",
                data.errorcode, accessClient_strerror(data.errorcode), cpu, reg);
        exit(EXIT_FAILURE);
    }

    return data.data;
}

void 
accessClient_write(int socket_fd, const int cpu, const int device, uint32_t reg, uint64_t sdata)
{
    AccessDataRecord data;

    data.cpu = cpu;
    data.reg = reg;
    data.data = sdata;
    data.type = DAEMON_WRITE;
    data.device = device;
    CHECK_ERROR(write(socket_fd, &data, sizeof(AccessDataRecord)), socket write failed);
    CHECK_ERROR(read(socket_fd, &data, sizeof(AccessDataRecord)), socket read failed);

    if (data.errorcode != ERR_NOERROR)
    {
        fprintf(stderr, "Failed to write data through daemon: "
                "daemon returned error %d '%s' for cpu %d reg 0x%x\n",
                data.errorcode, accessClient_strerror(data.errorcode), cpu, reg);
        exit(EXIT_FAILURE);
    }

    if (data.data != 0x00ULL)
    {
        ERROR_PLAIN_PRINT(daemon write failed);
    }
}


