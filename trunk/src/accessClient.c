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
#include <perfmon.h>
#include <configuration.h>

int accessClient_mode = ACCESSMODE;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
static char*
accessClient_strerror(AccessErrorType det)
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
        default:             return "UNKNOWN errorcode";
    }
}

static int
accessClient_errno(AccessErrorType det)
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
startDaemon(void)
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
    int timeout = 1000;
    int socket_fd = -1;

    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        return 0;
    }

    if (config.daemonPath != NULL)
    {
        strcpy(exeprog, config.daemonPath);
    }

    if (access(exeprog, X_OK))
    {
        ERROR_PRINT(Failed to find the daemon '%s'\n, exeprog);
        exit(EXIT_FAILURE);
    }

    if (accessClient_mode == ACCESSMODE_DAEMON)
    {
        pid = fork();

        if (pid == 0)
        {
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
            ERROR_PLAIN_PRINT(Failed to fork);
        }
    }

    EXIT_IF_ERROR(socket_fd = socket(AF_LOCAL, SOCK_STREAM, 0), socket() failed);

    address.sun_family = AF_LOCAL;
    address_length = sizeof(address);
    snprintf(address.sun_path, sizeof(address.sun_path), "/tmp/likwid-%d", pid);
    filepath = strdup(address.sun_path);
    if (accessClient_mode == ACCESSMODE_DAEMON)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Socket pathname is %s, filepath);
    }

    while (timeout > 0)
    {
        int res;
        usleep(1000);
        res = connect(socket_fd, (struct sockaddr *) &address, address_length);

        if (res == 0)
        {
            break;
        }

        timeout--;
        DEBUG_PRINT(DEBUGLEV_INFO, Still waiting for socket %s ..., filepath);
    }
    
    if (timeout <= 0)
    {
        ERRNO_PRINT;  /* should hopefully still work, as we make no syscalls in between. */
        fprintf(stderr, "Exiting due to timeout: The socket file at '%s' \
                could not be opened within 10 seconds.\n", filepath);
        fprintf(stderr, "Consult the error message above this to find out why.\n");
        fprintf(stderr, "If the error is 'no such file or directoy', \
                it usually means that likwid-accessD just failed to start.\n");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(DEBUGLEV_INFO, Successfully opened socket %s to daemon, filepath);
    free(filepath);

    return socket_fd;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void 
accessClient_setaccessmode(int mode)
{
    if ((accessClient_mode > ACCESSMODE_DAEMON) || (accessClient_mode < ACCESSMODE_DIRECT))
    {
        ERROR_PRINT(Invalid accessmode %d, accessClient_mode);
        exit(EXIT_FAILURE);
    }
    accessClient_mode = mode;
}

void 
accessClient_init(int* socket_fd)
{
    if (config.daemonMode != -1)
    {
        accessClient_mode = config.daemonMode;
    }
    if ((accessClient_mode == ACCESSMODE_DAEMON) && (*socket_fd == -1))
    {
        (*socket_fd) = startDaemon();
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


int
accessClient_read(
        int socket_fd,
        const int cpu,
        const int device,
        uint32_t reg,
        uint64_t *result)
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
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Got error '%s' from access daemon reading reg 0x%X at CPU %d, accessClient_strerror(data.errorcode), data.reg, data.cpu);
        *result = 0;
        return accessClient_errno(data.errorcode);
    }
    *result = data.data;
    return 0;
}

int 
accessClient_write(
        int socket_fd,
        const int cpu,
        const int device,
        uint32_t reg,
        uint64_t sdata)
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
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Got error '%s' from access daemon writing reg 0x%X at CPU %d, accessClient_strerror(data.errorcode), data.reg, data.cpu);
        return accessClient_errno(data.errorcode);
    }

    return 0;
}


