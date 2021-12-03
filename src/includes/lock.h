/*
 * =======================================================================================
 *
 *      Filename:  lock.h
 *
 *      Description:  Header File Locking primitive Module
 *
 *      Version:   5.2.1
 *      Released:  03.12.2021
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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
#ifndef LOCK_H
#define LOCK_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#define LOCK_INIT -1
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

static inline int
lock_acquire(int* var, int newval)
{
    int oldval = LOCK_INIT;
    return __sync_bool_compare_and_swap (var, oldval, newval);
}

static int
lock_check(void)
{
    struct stat buf;
    int lock_handle = -1;
    int result = 0;
    char* filepath = TOSTRING(LIKWIDLOCK);

    if ((lock_handle = open(filepath, O_RDONLY )) == -1 )
    {
        if (errno == ENOENT)
        {
            /* There is no lock file. Proceed. */
            result = 1;
        }
        else if (errno == EACCES)
        {
            /* There is a lock file. We cannot open it. */
            result = 0;
        }
        else
        {
            /* Another error occured. Proceed. */
            result = 1;
        }
    }
    else
    {
        /* There is a lock file and we can open it. Check if we own it. */
        stat(filepath, &buf);

        if ( buf.st_uid == getuid() )  /* Succeed, we own the lock */
        {
            result = 1;
        }
        else  /* we are not the owner */
        {

            result = 0;
        }
    }

    if (lock_handle > 0)
    {
        close(lock_handle);
    }

    return result;
}

#endif /*LOCK_H*/
