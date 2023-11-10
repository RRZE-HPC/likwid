/*
 * =======================================================================================
 *
 *      Filename:  frequency_client.h
 *
 *      Description:  Header File frequency module, the interface to the frequency daemon
 *
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
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

#ifndef LIKWID_SYSFS_CLIENT_H
#define LIKWID_SYSFS_CLIENT_H

#define LIKWID_SYSFS_MAX_DATA_LENGTH   1025

typedef enum {
    SYSFS_READ = 0,
    SYSFS_WRITE,
    SYSFS_EXIT
} SysfsDataRecordType;


typedef enum {
    SYSFS_ERR_NONE = 0,
    SYSFS_ERR_NOFILE,
    SYSFS_ERR_NOPERM,
    SYSFS_ERR_UNKNOWN
} SysfsDataRecordError;

typedef struct {
    SysfsDataRecordType type;
    SysfsDataRecordError errorcode;
    char filename[LIKWID_SYSFS_MAX_DATA_LENGTH];
    int datalen;
    char data[LIKWID_SYSFS_MAX_DATA_LENGTH];
} SysfsDataRecord;

#endif /* LIKWID_SYSFS_CLIENT_H */
