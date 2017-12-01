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
 *      Author:   Thomas Roehl (tr), thomas.roehl@gmail.com
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

 /*! \brief The groupInfo data structure describes a performance group

Groups can be either be read in from file or be a group with custom event set. For
performance groups commonly all values are set. For groups with custom event set,
the fields groupname and shortinfo are set to 'Custom', longinfo is NULL and in
general the nmetrics value is 0.
*/
typedef struct {
    char* groupname; /*!< \brief Name of the group: performance group name or 'Custom' */
    char* shortinfo; /*!< \brief Short info string for the group or 'Custom' */
    int nevents; /*!< \brief Number of event/counter combinations */
    char** events; /*!< \brief List of events */
    char** counters; /*!< \brief List of counter registers */
    int nmetrics; /*!< \brief Number of metrics */
    char** metricnames; /*!< \brief Metric names */
    char** metricformulas; /*!< \brief Metric formulas */
    char* longinfo; /*!< \brief Descriptive text about the group or empty */
    char* lua_funcs; /*!< \brief Custom Lua functions used in metric formulas */
} GroupInfo;

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

extern int get_groups(const char* grouppath, const char* architecture, char*** groupnames, char*** groupshort, char*** grouplong);
extern void return_groups(int groups, char** groupnames, char** groupshort, char** grouplong);
extern int read_group(const char* grouppath, const char* architecture, const char* groupname, GroupInfo* ginfo);
extern int custom_group(const char* eventStr, GroupInfo* ginfo);
extern char* get_eventStr(GroupInfo* ginfo);
void put_eventStr(char* eventset);
extern char* get_shortInfo(GroupInfo* ginfo);
void put_shortInfo(char* sinfo);
extern char* get_longInfo(GroupInfo* ginfo);
void put_longInfo(char* linfo);
extern void return_group(GroupInfo* ginfo);



extern int calc_add_str_def(char* name, char* value, int cpu);
extern int calc_add_int_def(char* name, int value, int cpu);
extern int calc_add_dbl_def(char* name, double value, int cpu);
extern int calc_add_str_var(char* name, char* value, bstring vars, bstring varlist);
extern int calc_add_dbl_var(char* name, double value, bstring vars, bstring varlist);
extern int calc_add_int_var(char* name, int value, bstring vars, bstring varlist);
extern int calc_set_user_funcs(char* s);
extern int calc_metric(int cpu, char* formula, bstring varstr, bstring varlist, double *result);
#endif /* PERFGROUP_H */
