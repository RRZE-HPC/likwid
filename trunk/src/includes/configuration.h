/*
 * =======================================================================================
 *
 *      Filename:  configuration.h
 *
 *      Description:  Header File of Module configuration.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Thomas Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <types.h>
#include <error.h>

typedef struct {
    char* topologyCfgFileName;
    char* daemonPath;
    AccessMode daemonMode;
    int maxNumThreads;
    int maxNumNodes;
    int maxHashTableSize;
} Configuration;

typedef Configuration* Configuration_t;

extern Configuration config;
extern int init_config;

int init_configuration(void);
int destroy_configuration(void);
Configuration_t get_configuration(void);



#endif
