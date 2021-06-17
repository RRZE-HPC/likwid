/*
 * =======================================================================================
 *
 *      Filename:  frequency_client.h
 *
 *      Description:  Header File frequency module, the interface to the frequency daemon
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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

#ifndef LIKWID_FREQUENCY_CLIENT_H
#define LIKWID_FREQUENCY_CLIENT_H

#define LIKWID_FREQUENCY_MAX_DATA_LENGTH   200

typedef enum {
    FREQ_READ = 0,
    FREQ_WRITE,
    FREQ_EXIT
} FreqDataRecordType;


typedef enum {
    FREQ_LOC_MIN = 0,
    FREQ_LOC_MAX,
    FREQ_LOC_CUR,
    FREQ_LOC_GOV,
    FREQ_LOC_AVAIL_GOV,
    FREQ_LOC_AVAIL_FREQ,
    FREQ_LOC_CONF_MIN,
    FREQ_LOC_CONF_MAX,
    MAX_FREQ_LOCS
}FreqDataRecordLocation;

typedef enum {
    FREQ_ERR_NONE = 0,
    FREQ_ERR_NOFILE,
    FREQ_ERR_NOPERM,
    FREQ_ERR_UNKNOWN
} FreqDataRecordError;

typedef struct {
    uint32_t cpu;
    FreqDataRecordType type;
    FreqDataRecordLocation loc;
    FreqDataRecordError errorcode;
    int datalen;
    char data[LIKWID_FREQUENCY_MAX_DATA_LENGTH];
} FreqDataRecord;

#endif /* LIKWID_FREQUENCY_CLIENT_H */
