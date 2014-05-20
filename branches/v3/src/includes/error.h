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

#ifndef ERROR_H
#define ERROR_H

#include <errno.h>
#include <string.h>
#include <msr.h>

#define str(x) #x

#define FINALIZE  msr_finalize()

#define ERRNO_PRINT fprintf(stderr, "ERROR - [%s:%d] %s\n", __FILE__, __LINE__, strerror(errno))

#define ERROR  \
    ERRNO_PRINT; \
    FINALIZE; \
    exit(EXIT_FAILURE)

#define ERROR_PLAIN_PRINT(msg) \
   fprintf(stderr,  "ERROR - [%s:%d] " str(msg) "\n", __FILE__, __LINE__);  \
   FINALIZE; \
   exit(EXIT_FAILURE)


#define ERROR_PRINT(fmt, ...) \
   fprintf(stderr,  "ERROR - [%s:%d] " str(fmt) "\n", __FILE__, __LINE__, __VA_ARGS__);  \
   FINALIZE; \
   exit(EXIT_FAILURE)

#define CHECK_ERROR(func, msg)  \
    if ((func) < 0) { \
        fprintf(stderr, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno));  \
    }

#define EXIT_IF_ERROR(func, msg)  \
    if ((func) < 0) {  \
        fprintf(stderr,"ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); \
        FINALIZE; \
        exit(EXIT_FAILURE); \
    }

#ifndef DEBUGLEV
#define DEBUGLEV 0
#endif

#define VERBOSEPRINTREG(cpuid,reg,flags,msg) \
    if (perfmon_verbose) {  \
        printf("DEBUG - [%s:%d] "  str(msg) " [%d] Register 0x%llX , Flags: 0x%llX \n",  \
                __FILE__, __LINE__,  (cpuid), LLU_CAST (reg), LLU_CAST (flags)); \
        fflush(stdout);  \
    } 


#define DEBUG_PRINT(lev, fmt, ...) \
    if (DEBUGLEV > lev) { \
        printf(fmt, __VA_ARGS__); \
        fflush(stdout); \
    }

#endif /*ERROR_H*/
