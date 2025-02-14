/*
 * =======================================================================================
 *
 *      Filename:  perfgroup.h
 *
 *      Description:  Header File of performance group and event set handler
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
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
#ifndef PERFGROUP_H
#define PERFGROUP_H

#include <bstrlib.h>
#include <bstrlib_helper.h>

#include <likwid.h>
#include "calculator_exptree.h"

typedef enum {
    GROUP_NONE = 0,
    GROUP_SHORT,
    GROUP_EVENTSET,
    GROUP_METRICS,
    GROUP_LONG,
    GROUP_LUA,
    MAX_GROUP_FILE_SECTIONS
} GroupFileSections;

static char* groupFileSectionNames[MAX_GROUP_FILE_SECTIONS] = {
    "NONE",
    "SHORT",
    "EVENTSET",
    "METRICS",
    "LONG",
    "LUA"
};

typedef struct CounterList {
    int counters; /*!< \brief Number of entries in the list */
    struct bstrList* cnames; /*!< \brief List of counter names */
    struct bstrList* cvalues; /*!< \brief List of counter values */
} CounterList;

//extern int get_groups(const char* grouppath, const char* architecture, char*** groupnames, char*** groupshort, char*** grouplong);
//extern void return_groups(int groups, char** groupnames, char** groupshort, char** grouplong);
//extern int read_group(const char* grouppath, const char* architecture, const char* groupname, GroupInfo* ginfo);
//extern int custom_group(const char* eventStr, GroupInfo* ginfo);
//extern char* get_eventStr(GroupInfo* ginfo);
//void put_eventStr(char* eventset);
//extern char* get_shortInfo(GroupInfo* ginfo);
//void put_shortInfo(char* sinfo);
//extern char* get_longInfo(GroupInfo* ginfo);
//void put_longInfo(char* linfo);
//extern void return_group(GroupInfo* ginfo);


extern void init_clist(CounterList* clist);
extern int add_to_clist(CounterList* clist, char* counter, double result);
extern int update_clist(CounterList* clist, char* counter, double result);
extern void destroy_clist(CounterList* clist);

extern int calc_metric(char* formula, CounterList* clist, double *result);
extern int calc_metric_new(const struct exptree_node* tree, const CounterList* clist, double *result);


#endif /* PERFGROUP_H */
