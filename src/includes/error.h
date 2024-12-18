/*
 * =======================================================================================
 *
 *      Filename:  error.h
 *
 *      Description:  Central error handling macros
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@gmail.com
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
#ifndef ERROR_H
#define ERROR_H

#include <likwid.h>
#include <stdio.h>
#include <stdlib.h>

#define ERRNO_PRINT \
    fprintf(stderr, "ERROR - [%s:%d] %s\n", __FILE__, __LINE__, strerror(errno))

#define ERROR \
    do { \
        ERRNO_PRINT; \
        exit(EXIT_FAILURE); \
    } while (0)

#define ERROR_PRINT(fmt, ...) \
   fprintf(stderr,  "ERROR - [%s:%s:%d] %s.\n" fmt "\n", __FILE__,  __func__,__LINE__, strerror(errno), ##__VA_ARGS__)

#define CHECK_ERROR(func, msg) \
    do { \
        if ((func) < 0) \
            ERROR_PRINT(msg); \
    } while (0)

#define CHECK_AND_RETURN_ERROR(func, msg)  \
    do { \
        if ((func) < 0) { \
            ERROR_PRINT(msg); \
            return errno; \
        } \
    } while (0)

#define EXIT_IF_ERROR(func, msg)  \
    do { \
        if ((func) < 0) {  \
            fprintf(stderr,"ERROR - [%s:%d] %s - %s \n", __FILE__, __LINE__, msg, strerror(errno)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define VERBOSEPRINTREG(cpuid, reg, flags, msg) \
    do { \
        if (perfmon_verbosity >= DEBUGLEV_DETAIL) \
        { \
            printf("DEBUG - [%s:%d] %s [%d] Register 0x%llX , Flags: 0x%llX \n", \
                    __func__, __LINE__, msg, (cpuid), LLU_CAST (reg), LLU_CAST (flags)); \
            fflush(stdout);  \
        } \
    } while (0)

#define VERBOSEPRINTPCIREG(cpuid, dev, reg, flags, msg) \
    do { \
        if (perfmon_verbosity >= DEBUGLEV_DETAIL) \
        { \
            printf("DEBUG - [%s:%d] %s [%d] Device %d Register 0x%llX , Flags: 0x%llX \n",  \
                    __func__, __LINE__, msg, (cpuid), dev, LLU_CAST (reg), LLU_CAST (flags)); \
            fflush(stdout);  \
        } \
    } while (0)

#define DEBUG_PRINT(lev, fmt, ...) \
    do { \
        if ((lev) >= 0 && (lev) <= perfmon_verbosity) { \
            fprintf(stdout, "DEBUG - [%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while (0)

#define GPUDEBUG_PRINT(lev, fmt, ...) \
    do { \
        if ((lev) >= 0 && (lev) <= likwid_nvmon_verbosity) { \
            fprintf(stdout, "DEBUG - [%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while (0)

#define ROCMON_DEBUG_PRINT(lev, fmt, ...) \
    do { \
        if ((lev) >= 0 && (lev) <= likwid_rocmon_verbosity) { \
            fprintf(stdout, "ROCMON DEBUG - [%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while (0)

#define INFO_PRINT(fmt, ...) \
    do { \
        if (perfmon_verbosity >= DEBUGLEV_INFO) \
            fprintf(stdout, "INFO - " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define GPUINFO_PRINT(fmt, ...) \
    do { \
        if (likwid_nvmon_verbosity >= DEBUGLEV_INFO) \
            fprintf(stdout, "INFO - " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define ROCMON_INFO_PRINT(fmt, ...) \
    do { \
        if (likwid_rocmon_verbosity >= DEBUGLEV_INFO) \
            fprintf(stdout, "ROCMON INFO - " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define TODO_PRINT(fmt, ...) \
    fprintf(stdout, "TODO - " fmt "\n", ##__VA_ARGS__)


#define CHECK_MSR_WRITE_ERROR(func) CHECK_AND_RETURN_ERROR(func, "MSR write operation failed")
#define CHECK_MSR_READ_ERROR(func) CHECK_AND_RETURN_ERROR(func, "MSR read operation failed")
#define CHECK_PCI_WRITE_ERROR(func) CHECK_AND_RETURN_ERROR(func, "PCI write operation failed")
#define CHECK_PCI_READ_ERROR(func) CHECK_AND_RETURN_ERROR(func, "PCI read operation failed")
#define CHECK_MMIO_WRITE_ERROR(func) CHECK_AND_RETURN_ERROR(func, "MMIO write operation failed")
#define CHECK_MMIO_READ_ERROR(func) CHECK_AND_RETURN_ERROR(func, "MMIO read operation failed")
#define CHECK_POWER_READ_ERROR(func) CHECK_AND_RETURN_ERROR(func, "Power register read operation failed")
#define CHECK_TEMP_READ_ERROR(func) CHECK_AND_RETURN_ERROR(func, "Temperature register read operation failed")

#endif /*ERROR_H*/
