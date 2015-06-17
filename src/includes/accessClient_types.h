/*
 * =======================================================================================
 *
 *      Filename:  accessClient_types.h
 *
 *      Description:  Types file for accessClient module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#ifndef ACCESSCLIENT_TYPES_H
#define ACCESSCLIENT_TYPES_H

#include <stdint.h>

typedef enum {
    DAEMON_READ = 0,
    DAEMON_WRITE,
    DAEMON_EXIT
} AccessType;

typedef enum {
    DAEMON_AD_PCI_R3QPI_LINK_0 = 0,
    DAEMON_AD_PCI_R3QPI_LINK_1,
    DAEMON_AD_PCI_R2PCIE,
    DAEMON_AD_PCI_IMC_CH_0,
    DAEMON_AD_PCI_IMC_CH_1,
    DAEMON_AD_PCI_IMC_CH_2,
    DAEMON_AD_PCI_IMC_CH_3,
    DAEMON_AD_PCI_HA,
    DAEMON_AD_PCI_QPI_PORT_0,
    DAEMON_AD_PCI_QPI_PORT_1,
    DAEMON_AD_PCI_QPI_MASK_PORT_0,
    DAEMON_AD_PCI_QPI_MASK_PORT_1,
    DAEMON_AD_PCI_QPI_MISC_PORT_0,
    DAEMON_AD_PCI_QPI_MISC_PORT_1,
    DAEMON_AD_MSR
} AccessDevice;

typedef enum {
    ERR_NOERROR = 0,  /* no error */
    ERR_UNKNOWN,      /* unknown command */
    ERR_RESTREG,      /* attempt to access restricted MSR */
    ERR_OPENFAIL,     /* failure to open msr files */
    ERR_RWFAIL,       /* failure to read/write msr */
    ERR_DAEMONBUSY,   /* daemon already has another client */
    ERR_NODEV         /* No such device */
} AccessErrorType;

typedef struct {
    uint32_t cpu;
    uint32_t reg;
    uint64_t data;
    AccessDevice device;
    AccessType type;
    AccessErrorType errorcode; /* Only in replies - 0 if no error. */
} AccessDataRecord;

extern int accessClient_mode;

#endif /*ACCESSCLIENT_TYPES_H*/
