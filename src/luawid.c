/*
 * =======================================================================================
 *
 *      Filename:  luawid.c
 *
 *      Description:  C part of the Likwid Lua interface
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <sched.h>
#include <sys/types.h>
#include <pwd.h>
#include <sys/syscall.h>

#include <lua.h>                               /* Always include this */
#include <lauxlib.h>                           /* Always include this */
#include <lualib.h>                            /* Always include this */

#include <likwid.h>
#include <tree.h>
#include <access.h>
#include <bstrlib.h>

#ifdef COLOR
#include <textcolor.h>
#endif

#define gettid() syscall(SYS_gettid)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int topology_isInitialized = 0;
static int numa_isInitialized = 0;
static int affinity_isInitialized = 0;
static int perfmon_isInitialized = 0;
static int timer_isInitialized = 0;
static int power_isInitialized = 0;
static int power_hasRAPL = 0;
static int config_isInitialized = 0;

/* #####   VARIABLES  -  EXPORTED VARIABLES   ############################# */

CpuInfo_t cpuinfo = NULL;
CpuTopology_t cputopo = NULL;
NumaTopology_t numainfo = NULL;
AffinityDomains_t affinity = NULL;
PowerInfo_t power;
Configuration_t configfile = NULL;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
lua_likwid_getConfiguration(lua_State* L)
{
    int ret = 0;
    if (config_isInitialized == 0)
    {
        ret = init_configuration();
        if (ret == 0)
        {
            config_isInitialized = 1;
            configfile = get_configuration();
        }
        else
        {
            lua_newtable(L);
            lua_pushstring(L, "configFile");
            lua_pushnil(L);
            lua_settable(L,-3);
            lua_pushstring(L, "topologyFile");
            lua_pushnil(L);
            lua_settable(L,-3);
            lua_pushstring(L, "daemonPath");
            lua_pushnil(L);
            lua_settable(L,-3);
            lua_pushstring(L, "groupPath");
            lua_pushnil(L);
            lua_settable(L,-3);
            lua_pushstring(L, "daemonMode");
            lua_pushinteger(L, -1);
            lua_settable(L,-3);
            lua_pushstring(L, "maxNumThreads");
            lua_pushinteger(L, 0);
            lua_settable(L,-3);
            lua_pushstring(L, "maxNumNodes");
            lua_pushinteger(L, 0);
            lua_settable(L,-3);
            return 1;
        }
    }
    if ((config_isInitialized) && (configfile == NULL))
    {
        configfile = get_configuration();
    }
    if (configfile)
    {
        lua_newtable(L);
        lua_pushstring(L, "configFile");
        if (configfile->configFileName != NULL)
            lua_pushstring(L, configfile->configFileName);
        else
            lua_pushnil(L);
        lua_settable(L,-3);
        lua_pushstring(L, "topologyFile");
        lua_pushstring(L, configfile->topologyCfgFileName);
        lua_settable(L,-3);
        lua_pushstring(L, "daemonPath");
        if (configfile->daemonPath != NULL)
            lua_pushstring(L, configfile->daemonPath);
        else
            lua_pushnil(L);
        lua_settable(L,-3);
        lua_pushstring(L, "groupPath");
        lua_pushstring(L, configfile->groupPath);
        lua_settable(L,-3);
        lua_pushstring(L, "daemonMode");
        lua_pushinteger(L, (int)configfile->daemonMode);
        lua_settable(L,-3);
        lua_pushstring(L, "maxNumThreads");
        lua_pushinteger(L, configfile->maxNumThreads);
        lua_settable(L,-3);
        lua_pushstring(L, "maxNumNodes");
        lua_pushinteger(L, configfile->maxNumNodes);
        lua_settable(L,-3);
        return 1;
    }
    return 0;
}

static int
lua_likwid_putConfiguration(lua_State* L)
{
    if (config_isInitialized == 1)
    {
        destroy_configuration();
        config_isInitialized = 0;
        configfile = NULL;
    }
    return 0;
}

static int
lua_likwid_setGroupPath(lua_State* L)
{
    int ret;
    const char* tmpString;
    if (config_isInitialized == 0)
    {
        ret = init_configuration();
        if (ret == 0)
        {
            config_isInitialized = 1;
        }
    }
    tmpString = luaL_checkstring(L, 1);
    ret = config_setGroupPath((char*)tmpString);
    if (ret < 0)
    {
        lua_pushstring(L,"Cannot set group path");
        lua_error(L);
    }
    return 0;
}

static int
lua_likwid_setAccessMode(lua_State* L)
{
    int flag;
    flag = luaL_checknumber(L,1);
    luaL_argcheck(L, flag >= 0 && flag <= 1, 1, "invalid access mode, only 0 (direct) and 1 (accessdaemon) allowed");
    HPMmode(flag);
    lua_pushinteger(L,0);
    return 1;
}

static int
lua_likwid_getAccessMode(lua_State* L)
{
#ifdef LIKWID_USE_PERFEVENT
    return ACCESSMODE_PERF;
#else
    init_configuration();
    Configuration_t config = get_configuration();
    return config->daemonMode;
#endif
}

static int
lua_likwid_init(lua_State* L)
{
    int ret;
    int nrThreads = luaL_checknumber(L,1);
    luaL_argcheck(L, nrThreads > 0, 1, "CPU count must be greater than 0");
    int cpus[nrThreads];
    if (!lua_istable(L, -1)) {
      lua_pushstring(L,"No table given as second argument");
      lua_error(L);
    }
    for (ret = 1; ret<=nrThreads; ret++)
    {
        lua_rawgeti(L,-1,ret);
#if LUA_VERSION_NUM == 501
        cpus[ret-1] = ((lua_Integer)lua_tointeger(L,-1));
#else
        cpus[ret-1] = ((lua_Unsigned)lua_tointegerx(L,-1, NULL));
#endif
        lua_pop(L,1);
    }
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cpuinfo = get_cpuInfo();
        cputopo = get_cpuTopology();
    }
    if ((topology_isInitialized) && (cpuinfo == NULL))
    {
        cpuinfo = get_cpuInfo();
    }
    if ((topology_isInitialized) && (cputopo == NULL))
    {
        cputopo = get_cpuTopology();
    }
    if (numa_isInitialized == 0)
    {
        numa_init();
        numa_isInitialized = 1;
        numainfo = get_numaTopology();
    }
    if ((numa_isInitialized) && (numainfo == NULL))
    {
        numainfo = get_numaTopology();
    }
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    if (perfmon_isInitialized == 0)
    {
        ret = perfmon_init(nrThreads, &(cpus[0]));
        if (ret != 0)
        {
            lua_pushstring(L,"Cannot initialize likwid perfmon");
            perfmon_finalize();
            lua_pushinteger(L,ret);
            return 1;
        }
        perfmon_isInitialized = 1;
        timer_isInitialized = 1;
        lua_pushinteger(L,ret);
    }
    return 1;
}

static int
lua_likwid_addEventSet(lua_State* L)
{
    int groupId, n;
    const char* tmpString;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    n = lua_gettop(L);
    tmpString = luaL_checkstring(L, n);
    luaL_argcheck(L, strlen(tmpString) > 0, n, "Event string must be larger than 0");

    groupId = perfmon_addEventSet((char*)tmpString);
    if (groupId >= 0)
        lua_pushinteger(L, groupId+1);
    else
        lua_pushinteger(L, groupId);
    return 1;
}

static int
lua_likwid_setupCounters(lua_State* L)
{
    int ret;
    int groupId = lua_tonumber(L,1);
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    ret = perfmon_setupCounters(groupId-1);
    lua_pushinteger(L,ret);
    return 1;
}

static int
lua_likwid_startCounters(lua_State* L)
{
    int ret;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    ret = perfmon_startCounters();
    lua_pushinteger(L,ret);
    return 1;
}

static int
lua_likwid_stopCounters(lua_State* L)
{
    int ret;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    ret = perfmon_stopCounters();
    lua_pushinteger(L,ret);
    return 1;
}

static int
lua_likwid_readCounters(lua_State* L)
{
    int ret;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    ret = perfmon_readCounters();
    lua_pushinteger(L,ret);
    return 1;
}

static int
lua_likwid_switchGroup(lua_State* L)
{
    int ret = -1;
    int newgroup = lua_tonumber(L,1)-1;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    if (newgroup >= perfmon_getNumberOfGroups())
    {
        newgroup = 0;
    }
    if (newgroup == perfmon_getIdOfActiveGroup())
    {
        lua_pushinteger(L, ret);
        return 1;
    }
    ret = perfmon_switchActiveGroup(newgroup);
    lua_pushinteger(L, ret);
    return 1;
}

static int
lua_likwid_finalize(lua_State* L)
{
    if (perfmon_isInitialized == 1)
    {
        perfmon_finalize();
        perfmon_isInitialized = 0;
    }
    if (affinity_isInitialized == 1)
    {
        affinity_finalize();
        affinity_isInitialized = 0;
        affinity = NULL;
    }
    if (numa_isInitialized == 1)
    {
        numa_finalize();
        numa_isInitialized = 0;
        numainfo = NULL;
    }
    if (topology_isInitialized == 1)
    {
        topology_finalize();
        topology_isInitialized = 0;
        cputopo = NULL;
        cpuinfo = NULL;
    }
    if (timer_isInitialized == 1)
    {
        timer_finalize();
        timer_isInitialized = 0;
    }
    if (config_isInitialized == 1)
    {
        destroy_configuration();
        config_isInitialized = 0;
        configfile = NULL;
    }
    return 0;
}

static int
lua_likwid_getResult(lua_State* L)
{
    int groupId, eventId, threadId;
    double result = 0;
    groupId = lua_tonumber(L,1);
    eventId = lua_tonumber(L,2);
    threadId = lua_tonumber(L,3);
    result = perfmon_getResult(groupId-1, eventId-1, threadId-1);
    lua_pushnumber(L,result);
    return 1;
}

static int
lua_likwid_getLastResult(lua_State* L)
{
    int groupId, eventId, threadId;
    double result = 0;
    groupId = lua_tonumber(L,1);
    eventId = lua_tonumber(L,2);
    threadId = lua_tonumber(L,3);
    result = perfmon_getLastResult(groupId-1, eventId-1, threadId-1);
    lua_pushnumber(L,result);
    return 1;
}

static int
lua_likwid_getMetric(lua_State* L)
{
    int groupId, metricId, threadId;
    double result = 0;
    groupId = lua_tonumber(L,1);
    metricId = lua_tonumber(L,2);
    threadId = lua_tonumber(L,3);
    result = perfmon_getMetric(groupId-1, metricId-1, threadId-1);
    lua_pushnumber(L,result);
    return 1;
}

static int
lua_likwid_getLastMetric(lua_State* L)
{
    int groupId, metricId, threadId;
    double result = 0;
    groupId = lua_tonumber(L,1);
    metricId = lua_tonumber(L,2);
    threadId = lua_tonumber(L,3);
    result = perfmon_getLastMetric(groupId-1, metricId-1, threadId-1);
    lua_pushnumber(L,result);
    return 1;
}

static int
lua_likwid_getNumberOfGroups(lua_State* L)
{
    int number;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    number = perfmon_getNumberOfGroups();
    lua_pushinteger(L,number);
    return 1;
}

static int
lua_likwid_getIdOfActiveGroup(lua_State* L)
{
    int number;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    number = perfmon_getIdOfActiveGroup();
    lua_pushinteger(L,number+1);
    return 1;
}

static int
lua_likwid_getRuntimeOfGroup(lua_State* L)
{
    double time;
    int groupId;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    time = perfmon_getTimeOfGroup(groupId-1);
    lua_pushnumber(L, time);
    return 1;
}

static int
lua_likwid_getNumberOfEvents(lua_State* L)
{
    int number, groupId;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    number = perfmon_getNumberOfEvents(groupId-1);
    lua_pushinteger(L,number);
    return 1;
}

static int
lua_likwid_getNumberOfThreads(lua_State* L)
{
    int number;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    number = perfmon_getNumberOfThreads();
    lua_pushinteger(L,number);
    return 1;
}

static int
lua_likwid_getNameOfEvent(lua_State* L)
{
    int eventId, groupId;
    char* tmp;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    eventId = lua_tonumber(L,2);
    tmp = perfmon_getEventName(groupId-1, eventId-1);
    lua_pushstring(L,tmp);
    return 1;
}

static int
lua_likwid_getNameOfCounter(lua_State* L)
{
    int eventId, groupId;
    char* tmp;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    eventId = lua_tonumber(L,2);
    tmp = perfmon_getCounterName(groupId-1, eventId-1);
    lua_pushstring(L,tmp);
    return 1;
}

static int
lua_likwid_getNumberOfMetrics(lua_State* L)
{
    int number, groupId;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    number = perfmon_getNumberOfMetrics(groupId-1);
    lua_pushinteger(L,number);
    return 1;
}

static int
lua_likwid_getNameOfMetric(lua_State* L)
{
    int metricId, groupId;
    char* tmp;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    metricId = lua_tonumber(L,2);
    tmp = perfmon_getMetricName(groupId-1, metricId-1);
    lua_pushstring(L,tmp);
    return 1;
}

static int
lua_likwid_getNameOfGroup(lua_State* L)
{
    int groupId;
    char* tmp;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    tmp = perfmon_getGroupName(groupId-1);
    lua_pushstring(L,tmp);
    return 1;
}

static int
lua_likwid_getShortInfoOfGroup(lua_State* L)
{
    int groupId;
    char* tmp;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    tmp = perfmon_getGroupInfoShort(groupId-1);
    lua_pushstring(L,tmp);
    return 1;
}

static int
lua_likwid_getLongInfoOfGroup(lua_State* L)
{
    int groupId;
    char* tmp;
    if (perfmon_isInitialized == 0)
    {
        return 0;
    }
    groupId = lua_tonumber(L,1);
    tmp = perfmon_getGroupInfoLong(groupId-1);
    lua_pushstring(L,tmp);
    return 1;
}

static int
lua_likwid_getGroups(lua_State* L)
{
    int i, ret;
    char** tmp, **infos, **longs;
    if (topology_isInitialized == 0)
    {
        topology_init();
    }
    ret = perfmon_getGroups(&tmp, &infos, &longs);
    if (ret > 0)
    {
        lua_newtable(L);
        for (i=0;i<ret;i++)
        {
            lua_pushinteger(L, (lua_Integer)( i+1));
            lua_newtable(L);
            lua_pushstring(L, "Name");
            lua_pushstring(L, tmp[i]);
            lua_settable(L,-3);
            lua_pushstring(L, "Info");
            lua_pushstring(L, infos[i]);
            lua_settable(L,-3);
            lua_pushstring(L, "Long");
            lua_pushstring(L, longs[i]);
            lua_settable(L,-3);
            lua_settable(L,-3);
        }
        perfmon_returnGroups(ret, tmp, infos, longs);
        return 1;
    }
    return 0;
}


static int
lua_likwid_printSupportedCPUs(lua_State* L)
{
    print_supportedCPUs();
    return 0;
}

static int
lua_likwid_getCpuInfo(lua_State* L)
{
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cpuinfo = get_cpuInfo();
    }
    if ((topology_isInitialized) && (cpuinfo == NULL))
    {
        cpuinfo = get_cpuInfo();
    }
    lua_newtable(L);
    lua_pushstring(L,"family");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->family));
    lua_settable(L,-3);
    lua_pushstring(L,"model");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->model));
    lua_settable(L,-3);
    lua_pushstring(L,"stepping");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->stepping));
    lua_settable(L,-3);
    lua_pushstring(L,"clock");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->clock));
    lua_settable(L,-3);
    lua_pushstring(L,"turbo");
    lua_pushinteger(L,cpuinfo->turbo);
    lua_settable(L,-3);
    lua_pushstring(L,"name");
    lua_pushstring(L,cpuinfo->name);
    lua_settable(L,-3);
    lua_pushstring(L,"osname");
    lua_pushstring(L,cpuinfo->osname);
    lua_settable(L,-3);
    lua_pushstring(L,"short_name");
    lua_pushstring(L,cpuinfo->short_name);
    lua_settable(L,-3);
    lua_pushstring(L,"features");
    lua_pushstring(L,cpuinfo->features);
    lua_settable(L,-3);
    lua_pushstring(L,"isIntel");
    lua_pushinteger(L,cpuinfo->isIntel);
    lua_settable(L,-3);
    lua_pushstring(L,"featureFlags");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->featureFlags));
    lua_settable(L,-3);
    lua_pushstring(L,"perf_version");
    lua_pushinteger(L, (lua_Integer)( cpuinfo->perf_version));
    lua_settable(L,-3);
    lua_pushstring(L,"perf_num_ctr");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->perf_num_ctr));
    lua_settable(L,-3);
    lua_pushstring(L,"perf_width_ctr");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->perf_width_ctr));
    lua_settable(L,-3);
    lua_pushstring(L,"perf_num_fixed_ctr");
    lua_pushinteger(L, (lua_Integer)(cpuinfo->perf_num_fixed_ctr));
    lua_settable(L,-3);
    return 1;
}

static int
lua_likwid_getCpuTopology(lua_State* L)
{
    int i;
    TreeNode* socketNode;
    int socketCount = 0;
    TreeNode* coreNode;
    int coreCount = 0;
    TreeNode* threadNode;
    int threadCount = 0;
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cputopo = get_cpuTopology();
    }
    if ((topology_isInitialized) && (cputopo == NULL))
    {
        cputopo = get_cpuTopology();
    }
    if (numa_isInitialized == 0)
    {
        if (numa_init() == 0)
        {
            numa_isInitialized = 1;
            numainfo = get_numaTopology();
        }
    }
    if ((numa_isInitialized) && (numainfo == NULL))
    {
        numainfo = get_numaTopology();
    }

    lua_newtable(L);

    lua_pushstring(L,"numHWThreads");
    lua_pushinteger(L, (lua_Integer)(cputopo->numHWThreads));
    lua_settable(L,-3);

    lua_pushstring(L,"activeHWThreads");
    lua_pushinteger(L, (lua_Integer)(cputopo->activeHWThreads));
    lua_settable(L,-3);

    lua_pushstring(L,"numSockets");
    lua_pushinteger(L, (lua_Integer)(cputopo->numSockets));
    lua_settable(L,-3);

    lua_pushstring(L,"numCoresPerSocket");
    lua_pushinteger(L, (lua_Integer)(cputopo->numCoresPerSocket));
    lua_settable(L,-3);

    lua_pushstring(L,"numThreadsPerCore");
    lua_pushinteger(L, (lua_Integer)(cputopo->numThreadsPerCore));
    lua_settable(L,-3);

    lua_pushstring(L,"numCacheLevels");
    lua_pushinteger(L,cputopo->numCacheLevels);
    lua_settable(L,-3);

    lua_pushstring(L,"threadPool");
    lua_newtable(L);
    for(i=0;i<cputopo->numHWThreads;i++)
    {
        lua_pushnumber(L,i);
        lua_newtable(L);
        lua_pushstring(L,"threadId");
        lua_pushinteger(L, (lua_Integer)(cputopo->threadPool[i].threadId));
        lua_settable(L,-3);
        lua_pushstring(L,"coreId");
        lua_pushinteger(L, (lua_Integer)(cputopo->threadPool[i].coreId));
        lua_settable(L,-3);
        lua_pushstring(L,"packageId");
        lua_pushinteger(L, (lua_Integer)(cputopo->threadPool[i].packageId));
        lua_settable(L,-3);
        lua_pushstring(L,"apicId");
        lua_pushinteger(L, (lua_Integer)(cputopo->threadPool[i].apicId));
        lua_settable(L,-3);
        lua_pushstring(L,"inCpuSet");
        lua_pushinteger(L, (lua_Integer)(cputopo->threadPool[i].inCpuSet));
        lua_settable(L,-3);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);

    lua_pushstring(L,"cacheLevels");
    lua_newtable(L);
    for(i=0;i<cputopo->numCacheLevels;i++)
    {
        lua_pushnumber(L,i+1);
        lua_newtable(L);

        lua_pushstring(L,"level");
        lua_pushinteger(L, (lua_Integer)(cputopo->cacheLevels[i].level));
        lua_settable(L,-3);

        lua_pushstring(L,"associativity");
        lua_pushinteger(L, (lua_Integer)(cputopo->cacheLevels[i].associativity));
        lua_settable(L,-3);

        lua_pushstring(L,"sets");
        lua_pushinteger(L, (lua_Integer)(cputopo->cacheLevels[i].sets));
        lua_settable(L,-3);

        lua_pushstring(L,"lineSize");
        lua_pushinteger(L, (lua_Integer)(cputopo->cacheLevels[i].lineSize));
        lua_settable(L,-3);

        lua_pushstring(L,"size");
        lua_pushinteger(L, (lua_Integer)(cputopo->cacheLevels[i].size));
        lua_settable(L,-3);

        lua_pushstring(L,"threads");
        lua_pushinteger(L, (lua_Integer)(cputopo->cacheLevels[i].threads));
        lua_settable(L,-3);

        lua_pushstring(L,"inclusive");
        lua_pushinteger(L, (lua_Integer)(cputopo->cacheLevels[i].inclusive));
        lua_settable(L,-3);

        lua_pushstring(L,"type");
        switch (cputopo->cacheLevels[i].type)
        {
            case DATACACHE:
                lua_pushstring(L,"DATACACHE");
                break;
            case INSTRUCTIONCACHE:
                lua_pushstring(L,"INSTRUCTIONCACHE");
                break;
            case UNIFIEDCACHE:
                lua_pushstring(L,"UNIFIEDCACHE");
                break;
            case ITLB:
                lua_pushstring(L,"ITLB");
                break;
            case DTLB:
                lua_pushstring(L,"DTLB");
                break;
            case NOCACHE:
            default:
                lua_pushstring(L,"NOCACHE");
                break;
        }
        lua_settable(L,-3);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);

    lua_pushstring(L,"topologyTree");
    lua_newtable(L);

    socketNode = tree_getChildNode(cputopo->topologyTree);
    while (socketNode != NULL)
    {
        lua_pushinteger(L, socketCount);
        lua_newtable(L);
        lua_pushstring(L, "ID");
        lua_pushinteger(L, (lua_Integer)(socketNode->id));
        lua_settable(L, -3);
        lua_pushstring(L, "Childs");
        lua_newtable(L);
        coreCount = 0;
        coreNode = tree_getChildNode(socketNode);
        while (coreNode != NULL)
        {
            lua_pushinteger(L, coreCount);
            lua_newtable(L);
            lua_pushstring(L, "ID");
            lua_pushinteger(L, (lua_Integer)(coreNode->id));
            lua_settable(L,-3);
            lua_pushstring(L, "Childs");
            lua_newtable(L);
            threadNode = tree_getChildNode(coreNode);
            threadCount = 0;
            while (threadNode != NULL)
            {
                lua_pushinteger(L, (lua_Integer)(threadCount));
                lua_pushinteger(L, (lua_Integer)(threadNode->id));
                lua_settable(L,-3);
                threadNode = tree_getNextNode(threadNode);
                threadCount++;
            }
            lua_settable(L,-3);
            coreNode = tree_getNextNode(coreNode);
            coreCount++;
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
        socketNode = tree_getNextNode(socketNode);
        socketCount++;
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    return 1;
}

static int
lua_likwid_putTopology(lua_State* L)
{
    if (topology_isInitialized == 1)
    {
        topology_finalize();
        topology_isInitialized = 0;
        cpuinfo = NULL;
        cputopo = NULL;
    }
    return 0;
}


static int
lua_likwid_getEventsAndCounters(lua_State* L)
{
    int i = 0, insert = 1;

    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cpuinfo = get_cpuInfo();
    }
    if ((topology_isInitialized) && (cpuinfo == NULL))
    {
        cpuinfo = get_cpuInfo();
    }
    perfmon_init_maps();
    perfmon_check_counter_map(0);
    lua_newtable(L);
    lua_pushstring(L,"Counters");
    lua_newtable(L);
    for(i=1;i<=perfmon_numCounters;i++)
    {
        if (counter_map[i-1].type == NOTYPE)
            continue;
        bstring optString = bfromcstr("");
        lua_pushinteger(L, (lua_Integer)(insert));
        lua_newtable(L);
        lua_pushstring(L,"Name");
        lua_pushstring(L,counter_map[i-1].key);
        lua_settable(L,-3);
        lua_pushstring(L,"Options");
        for(int j=1; j<NUM_EVENT_OPTIONS; j++)
        {
            if (counter_map[i-1].optionMask & REG_TYPE_MASK(j))
            {
                bstring tmp = bformat("%s|", eventOptionTypeName[j]);
                bconcat(optString, tmp);
                bdestroy(tmp);
            }
        }
        bdelete(optString, blength(optString)-1, 1);
        lua_pushstring(L,bdata(optString));
        lua_settable(L,-3);
        lua_pushstring(L,"Type");
        lua_pushinteger(L, (lua_Integer)( counter_map[i-1].type));
        lua_settable(L,-3);
        lua_pushstring(L,"TypeName");
        lua_pushstring(L, RegisterTypeNames[counter_map[i-1].type]);
        lua_settable(L,-3);
        lua_pushstring(L,"Index");
        lua_pushinteger(L, (lua_Integer)(counter_map[i-1].index));
        lua_settable(L,-3);
        lua_settable(L,-3);
        bdestroy(optString);
        insert++;
    }
    insert = 1;
    lua_settable(L,-3);
    lua_pushstring(L,"Events");
    lua_newtable(L);
    for(i=1;i<=perfmon_numArchEvents;i++)
    {
        if (strlen(eventHash[i-1].limit) == 0)
            continue;
        bstring optString = bfromcstr("");
        lua_pushinteger(L, (lua_Integer)(insert));
        lua_newtable(L);
        lua_pushstring(L,"Name");
        lua_pushstring(L,eventHash[i-1].name);
        lua_settable(L,-3);
        lua_pushstring(L,"ID");
        lua_pushinteger(L, (lua_Integer)(eventHash[i-1].eventId));
        lua_settable(L,-3);
        lua_pushstring(L,"UMask");
        lua_pushinteger(L, (lua_Integer)(eventHash[i-1].umask));
        lua_settable(L,-3);
        lua_pushstring(L,"Limit");
        lua_pushstring(L,eventHash[i-1].limit);
        lua_settable(L,-3);
        lua_pushstring(L,"Options");
        for(int j=1; j<NUM_EVENT_OPTIONS; j++)
        {
            if (eventHash[i-1].optionMask & REG_TYPE_MASK(j))
            {
                bstring tmp = bformat("%s|", eventOptionTypeName[j]);
                bconcat(optString, tmp);
                bdestroy(tmp);
            }
        }
        bdelete(optString, blength(optString)-1, 1);
        lua_pushstring(L,bdata(optString));
        lua_settable(L,-3);
        lua_settable(L,-3);
        bdestroy(optString);
        insert++;
    }
    lua_settable(L,-3);
    HPMfinalize();
    return 1;
}

static int
lua_likwid_getOnlineDevices(lua_State* L)
{
    int i;
    lua_newtable(L);
    for(i=0;i<=MAX_NUM_PCI_DEVICES;i++)
    {
        if (pci_devices[i].online)
        {
            lua_pushstring(L,pci_devices[i].likwid_name);
            lua_newtable(L);
            lua_pushstring(L, "Name");
            lua_pushstring(L,pci_devices[i].name);
            lua_settable(L,-3);
            lua_pushstring(L, "Path");
            lua_pushstring(L,pci_devices[i].path);
            lua_settable(L,-3);
            lua_pushstring(L, "Type");
            lua_pushstring(L,pci_types[pci_devices[i].type].name);
            lua_settable(L,-3);
            lua_pushstring(L, "TypeDescription");
            lua_pushstring(L,pci_types[pci_devices[i].type].desc);
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
    }
    return 1;
}

static int
lua_likwid_getNumaInfo(lua_State* L)
{
    uint32_t i,j;
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cpuinfo = get_cpuInfo();
        cputopo = get_cpuTopology();
    }
    if ((topology_isInitialized) && (cpuinfo == NULL))
    {
        cpuinfo = get_cpuInfo();
    }
    if ((topology_isInitialized) && (cputopo == NULL))
    {
        cputopo = get_cpuTopology();
    }
    if (numa_isInitialized == 0)
    {
        if (numa_init() == 0)
        {
            numa_isInitialized = 1;
            numainfo = get_numaTopology();
        }
        else
        {
            lua_newtable(L);
            lua_pushstring(L,"numberOfNodes");
            lua_pushinteger(L, (lua_Integer)(0));
            lua_settable(L,-3);
            lua_pushstring(L,"nodes");
            lua_newtable(L);
            lua_settable(L,-3);
            return 1;
        }
    }
    if ((numa_isInitialized) && (numainfo == NULL))
    {
        numainfo = get_numaTopology();
    }
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
        affinity = get_affinityDomains();
    }
    if ((affinity_isInitialized) && (affinity == NULL))
    {
        affinity = get_affinityDomains();
    }
    lua_newtable(L);
    lua_pushstring(L,"numberOfNodes");
    lua_pushinteger(L, (lua_Integer)(numainfo->numberOfNodes));
    lua_settable(L,-3);

    lua_pushstring(L,"nodes");
    lua_newtable(L);
    for(i=0;i<numainfo->numberOfNodes;i++)
    {
        lua_pushinteger(L, i+1);
        lua_newtable(L);
        lua_pushstring(L,"id");
        lua_pushinteger(L, (lua_Integer)(numainfo->nodes[i].id));
        lua_settable(L,-3);
        lua_pushstring(L,"totalMemory");
        lua_pushinteger(L, (lua_Integer)(numainfo->nodes[i].totalMemory));
        lua_settable(L,-3);
        lua_pushstring(L,"freeMemory");
        lua_pushinteger(L, (lua_Integer)(numainfo->nodes[i].freeMemory));
        lua_settable(L,-3);
        lua_pushstring(L,"numberOfProcessors");
        lua_pushinteger(L, (lua_Integer)(numainfo->nodes[i].numberOfProcessors));
        lua_settable(L,-3);
        lua_pushstring(L,"numberOfDistances");
        lua_pushinteger(L, (lua_Integer)(numainfo->nodes[i].numberOfDistances));
        lua_settable(L,-3);
        lua_pushstring(L,"processors");
        lua_newtable(L);
        for(j=0;j<numainfo->nodes[i].numberOfProcessors;j++)
        {
            lua_pushinteger(L, (lua_Integer)(j+1));
            lua_pushinteger(L, (lua_Integer)(numainfo->nodes[i].processors[j]));
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
        /*lua_pushstring(L,"processorsCompact");
        lua_newtable(L);
        for(j=0;j<numa->nodes[i].numberOfProcessors;j++)
        {
            lua_pushinteger(L, (lua_Integer)(j);
            lua_pushinteger(L, (lua_Integer)(numa->nodes[i].processorsCompact[j]);
            lua_settable(L,-3);
        }
        lua_settable(L,-3);*/
        lua_pushstring(L,"distances");
        lua_newtable(L);
        for(j=0;j<numainfo->nodes[i].numberOfDistances;j++)
        {
            lua_pushinteger(L,j+1);
            lua_newtable(L);
            lua_pushinteger(L,j);
            lua_pushinteger(L, (lua_Integer)(numainfo->nodes[i].distances[j]));
            lua_settable(L,-3);
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    return 1;
}

static int
lua_likwid_putNumaInfo(lua_State* L)
{
    if (numa_isInitialized)
    {
        numa_finalize();
        numa_isInitialized = 0;
        numainfo = NULL;
    }
    return 0;
}

static int
lua_likwid_setMemInterleaved(lua_State* L)
{
    int ret;
    int nrThreads = luaL_checknumber(L,1);
    luaL_argcheck(L, nrThreads > 0, 1, "Thread count must be greater than 0");
    int cpus[nrThreads];
    if (!lua_istable(L, -1)) {
      lua_pushstring(L,"No table given as second argument");
      lua_error(L);
    }
    for (ret = 1; ret<=nrThreads; ret++)
    {
        lua_rawgeti(L,-1,ret);
#if LUA_VERSION_NUM == 501
        cpus[ret-1] = ((lua_Integer)lua_tointeger(L,-1));
#else
        cpus[ret-1] = ((lua_Unsigned)lua_tointegerx(L,-1, NULL));
#endif
        lua_pop(L,1);
    }
    numa_setInterleaved(cpus, nrThreads);
    return 0;
}

static int
lua_likwid_getAffinityInfo(lua_State* L)
{
    int i,j;

    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cpuinfo = get_cpuInfo();
        cputopo = get_cpuTopology();
    }
    if ((topology_isInitialized) && (cpuinfo == NULL))
    {
        cpuinfo = get_cpuInfo();
    }
    if ((topology_isInitialized) && (cputopo == NULL))
    {
        cputopo = get_cpuTopology();
    }
    if (numa_isInitialized == 0)
    {
        if (numa_init() == 0)
        {
            numa_isInitialized = 1;
            numainfo = get_numaTopology();
        }
    }
    if ((numa_isInitialized) && (numainfo == NULL))
    {
        numainfo = get_numaTopology();
    }
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
        affinity = get_affinityDomains();
    }
    if ((affinity_isInitialized) && (affinity == NULL))
    {
        affinity = get_affinityDomains();
    }

    if (!affinity)
    {
        lua_pushstring(L,"Cannot initialize affinity groups");
        lua_error(L);
    }
    lua_newtable(L);
    lua_pushstring(L,"numberOfAffinityDomains");
    lua_pushinteger(L, (lua_Integer)(affinity->numberOfAffinityDomains));
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfSocketDomains");
    lua_pushinteger(L, (lua_Integer)(affinity->numberOfSocketDomains));
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfNumaDomains");
    lua_pushinteger(L, (lua_Integer)(affinity->numberOfNumaDomains));
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfProcessorsPerSocket");
    lua_pushinteger(L, (lua_Integer)(affinity->numberOfProcessorsPerSocket));
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfCacheDomains");
    lua_pushinteger(L, (lua_Integer)(affinity->numberOfCacheDomains));
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfCoresPerCache");
    lua_pushinteger(L, (lua_Integer)(affinity->numberOfCoresPerCache));
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfProcessorsPerCache");
    lua_pushinteger(L, (lua_Integer)(affinity->numberOfProcessorsPerCache));
    lua_settable(L,-3);
    lua_pushstring(L,"domains");
    lua_newtable(L);
    for(i=0;i<affinity->numberOfAffinityDomains;i++)
    {
        lua_pushinteger(L, (lua_Integer)( i+1));
        lua_newtable(L);
        lua_pushstring(L,"tag");
        lua_pushstring(L,bdata(affinity->domains[i].tag));
        lua_settable(L,-3);
        lua_pushstring(L,"numberOfProcessors");
        lua_pushinteger(L, (lua_Integer)(affinity->domains[i].numberOfProcessors));
        lua_settable(L,-3);
        lua_pushstring(L,"numberOfCores");
        lua_pushinteger(L, (lua_Integer)(affinity->domains[i].numberOfCores));
        lua_settable(L,-3);
        lua_pushstring(L,"processorList");
        lua_newtable(L);
        for(j=0;j<affinity->domains[i].numberOfProcessors;j++)
        {
            lua_pushinteger(L, (lua_Integer)(j+1));
            lua_pushinteger(L, (lua_Integer)(affinity->domains[i].processorList[j]));
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    return 1;
}

static int
lua_likwid_cpustr_to_cpulist(lua_State* L)
{
    int ret = 0;
    char* cpustr = (char *)luaL_checkstring(L, 1);
    if (!cputopo)
    {
        topology_init();
        cputopo = get_cpuTopology();
        topology_isInitialized = 1;
    }
    int* cpulist = (int*) malloc(cputopo->numHWThreads * sizeof(int));
    if (cpulist == NULL)
    {
        lua_pushstring(L,"Cannot allocate data for the CPU list");
        lua_error(L);
    }
    ret = cpustr_to_cpulist(cpustr, cpulist, cputopo->numHWThreads);
    if (ret <= 0)
    {
        lua_pushstring(L,"Cannot parse cpustring");
        lua_error(L);
    }
    lua_pushnumber(L, ret);
    lua_newtable(L);
    for (int i=0;i<ret;i++)
    {
        lua_pushinteger(L, (lua_Integer)( i+1));
        lua_pushinteger(L, (lua_Integer)( cpulist[i]));
        lua_settable(L,-3);
    }
    free(cpulist);
    return 2;
}

static int
lua_likwid_nodestr_to_nodelist(lua_State* L)
{
    int ret = 0;
    char* nodestr = (char *)luaL_checkstring(L, 1);
    if (!numainfo)
    {
        topology_init();
        numa_init();
        numainfo = get_numaTopology();
        topology_isInitialized = 1;
        numa_isInitialized = 1;
    }
    int* nodelist = (int*) malloc(numainfo->numberOfNodes * sizeof(int));
    if (nodelist == NULL)
    {
        lua_pushstring(L,"Cannot allocate data for the node list");
        lua_error(L);
    }
    ret = nodestr_to_nodelist(nodestr, nodelist, numainfo->numberOfNodes);
    if (ret <= 0)
    {
        lua_pushstring(L,"Cannot parse node string");
        lua_error(L);
    }
    lua_pushnumber(L, ret);
    lua_newtable(L);
    for (int i=0;i<ret;i++)
    {
        lua_pushinteger(L, (lua_Integer)( i+1));
        lua_pushinteger(L, (lua_Integer)( nodelist[i]));
        lua_settable(L,-3);
    }
    free(nodelist);
    return 2;
}

static int
lua_likwid_sockstr_to_socklist(lua_State* L)
{
    int ret = 0;
    char* sockstr = (char *)luaL_checkstring(L, 1);
    if (!cputopo)
    {
        topology_init();
        cputopo = get_cpuTopology();
        topology_isInitialized = 1;
    }
    int* socklist = (int*) malloc(cputopo->numSockets * sizeof(int));
    if (socklist == NULL)
    {
        lua_pushstring(L,"Cannot allocate data for the socket list");
        lua_error(L);
    }
    ret = nodestr_to_nodelist(sockstr, socklist, cputopo->numSockets);
    if (ret <= 0)
    {
        lua_pushstring(L,"Cannot parse socket string");
        lua_error(L);
    }
    lua_pushnumber(L, ret);
    lua_newtable(L);
    for (int i=0;i<ret;i++)
    {
        lua_pushinteger(L, (lua_Integer)( i+1));
        lua_pushinteger(L, (lua_Integer)( socklist[i]));
        lua_settable(L,-3);
    }
    free(socklist);
    return 2;
}

static int
lua_likwid_putAffinityInfo(lua_State* L)
{
    if (affinity_isInitialized)
    {
        affinity_finalize();
        affinity_isInitialized = 0;
        affinity = NULL;
    }
    return 0;
}

static int
lua_likwid_getPowerInfo(lua_State* L)
{
    int i;
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cpuinfo = get_cpuInfo();
        cputopo = get_cpuTopology();
    }
    if ((topology_isInitialized) && (cpuinfo == NULL))
    {
        cpuinfo = get_cpuInfo();
    }
    if ((topology_isInitialized) && (cputopo == NULL))
    {
        cputopo = get_cpuTopology();
    }
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
        affinity = get_affinityDomains();
    }
    if ((affinity_isInitialized) && (affinity == NULL))
    {
        affinity = get_affinityDomains();
    }
    if (power_isInitialized == 0)
    {
        power_hasRAPL = power_init(0);
        if (power_hasRAPL > 0)
        {
            for(i=0;i<affinity->numberOfAffinityDomains;i++)
            {
                if (bstrchrp(affinity->domains[i].tag, 'S', 0) != BSTR_ERR)
                {
                    HPMaddThread(affinity->domains[i].processorList[0]);
                }
            }
            power_isInitialized = 1;
            power = get_powerInfo();
        }
        else
        {
            return 0;
        }
    }
    lua_newtable(L);
    lua_pushstring(L,"hasRAPL");
    lua_pushboolean(L,power_hasRAPL);
    lua_settable(L,-3);
    lua_pushstring(L,"baseFrequency");
    lua_pushnumber(L,power->baseFrequency);
    lua_settable(L,-3);
    lua_pushstring(L,"minFrequency");
    lua_pushnumber(L,power->minFrequency);
    lua_settable(L,-3);
    lua_pushstring(L,"powerUnit");
    lua_pushnumber(L,power->powerUnit);
    lua_settable(L,-3);
    lua_pushstring(L,"timeUnit");
    lua_pushnumber(L,power->timeUnit);
    lua_settable(L,-3);
    lua_pushstring(L,"minUncoreFreq");
    lua_pushnumber(L,power->uncoreMinFreq);
    lua_settable(L,-3);
    lua_pushstring(L,"maxUncoreFreq");
    lua_pushnumber(L,power->uncoreMaxFreq);
    lua_settable(L,-3);
    lua_pushstring(L,"perfBias");
    lua_pushnumber(L,power->perfBias);
    lua_settable(L,-3);
    lua_pushstring(L,"turbo");
    lua_newtable(L);
    lua_pushstring(L,"numSteps");
    lua_pushinteger(L, (lua_Integer)(power->turbo.numSteps));
    lua_settable(L,-3);
    lua_pushstring(L,"steps");
    lua_newtable(L);
    for(i=0;i<power->turbo.numSteps;i++)
    {
        lua_pushinteger(L, (lua_Integer)(i+1));
        lua_pushnumber(L,power->turbo.steps[i]);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    lua_settable(L,-3);

    lua_pushstring(L,"domains");
    lua_newtable(L);
    for(i=0;i<NUM_POWER_DOMAINS;i++)
    {
        lua_pushstring(L,power_names[i]);
        lua_newtable(L);

        lua_pushstring(L, "ID");
        lua_pushnumber(L, power->domains[i].type);
        lua_settable(L,-3);
        lua_pushstring(L, "energyUnit");
        lua_pushnumber(L, power->domains[i].energyUnit);
        lua_settable(L,-3);
        lua_pushstring(L,"supportStatus");
        if (power->domains[i].supportFlags & POWER_DOMAIN_SUPPORT_STATUS)
        {
            lua_pushboolean(L, 1);
        }
        else
        {
            lua_pushboolean(L, 0);
        }
        lua_settable(L,-3);
        lua_pushstring(L,"supportPerf");
        if (power->domains[i].supportFlags & POWER_DOMAIN_SUPPORT_PERF)
        {
            lua_pushboolean(L, 1);
        }
        else
        {
            lua_pushboolean(L, 0);
        }
        lua_settable(L,-3);
        lua_pushstring(L,"supportPolicy");
        if (power->domains[i].supportFlags & POWER_DOMAIN_SUPPORT_POLICY)
        {
            lua_pushboolean(L, 1);
        }
        else
        {
            lua_pushboolean(L, 0);
        }
        lua_settable(L,-3);
        lua_pushstring(L,"supportLimit");
        if (power->domains[i].supportFlags & POWER_DOMAIN_SUPPORT_LIMIT)
        {
            lua_pushboolean(L, 1);
        }
        else
        {
            lua_pushboolean(L, 0);
        }
        lua_settable(L,-3);
        if (power->domains[i].supportFlags & POWER_DOMAIN_SUPPORT_INFO)
        {
            lua_pushstring(L,"supportInfo");
            lua_pushboolean(L, 1);
            lua_settable(L,-3);
            lua_pushstring(L,"tdp");
            lua_pushnumber(L, power->domains[i].tdp);
            lua_settable(L,-3);
            lua_pushstring(L,"minPower");
            lua_pushnumber(L, power->domains[i].minPower);
            lua_settable(L,-3);
            lua_pushstring(L,"maxPower");
            lua_pushnumber(L, power->domains[i].maxPower);
            lua_settable(L,-3);
            lua_pushstring(L,"maxTimeWindow");
            lua_pushnumber(L, power->domains[i].maxTimeWindow);
            lua_settable(L,-3);
        }
        else
        {
            lua_pushstring(L,"supportInfo");
            lua_pushboolean(L, 0);
            lua_settable(L,-3);
        }

        lua_settable(L,-3);
    }
    lua_settable(L,-3);

    return 1;
}

static int
lua_likwid_putPowerInfo(lua_State* L)
{
    if (power_isInitialized)
    {
        power_finalize();
        power_isInitialized = 0;
        power = NULL;
    }
    return 0;
}

static int
lua_likwid_startPower(lua_State* L)
{
    PowerData pwrdata;
    int cpuId = lua_tonumber(L,1);
    luaL_argcheck(L, cpuId >= 0, 1, "CPU ID must be greater than 0");
#if LUA_VERSION_NUM == 501
    PowerType type = (PowerType) ((lua_Integer)lua_tointeger(L,2));
#else
    PowerType type = (PowerType) ((lua_Unsigned)lua_tointegerx(L,2, NULL));
#endif
    luaL_argcheck(L, type >= PKG+1 && type <= NUM_POWER_DOMAINS, 2, "Type not valid");
    power_start(&pwrdata, cpuId, type-1);
    lua_pushnumber(L,pwrdata.before);
    return 1;
}

static int
lua_likwid_stopPower(lua_State* L)
{
    PowerData pwrdata;
    int cpuId = lua_tonumber(L,1);
    luaL_argcheck(L, cpuId >= 0, 1, "CPU ID must be greater than 0");
#if LUA_VERSION_NUM == 501
    PowerType type = (PowerType) ((lua_Integer)lua_tointeger(L,2));
#else
    PowerType type = (PowerType) ((lua_Unsigned)lua_tointegerx(L,2, NULL));
#endif
    luaL_argcheck(L, type >= PKG+1 && type <= NUM_POWER_DOMAINS, 2, "Type not valid");
    power_stop(&pwrdata, cpuId, type-1);
    lua_pushnumber(L,pwrdata.after);
    return 1;
}

static int
lua_likwid_printEnergy(lua_State* L)
{
    PowerData pwrdata;
    pwrdata.before = lua_tonumber(L,1);
    pwrdata.after = lua_tonumber(L,2);
    pwrdata.domain = lua_tonumber(L,3);
    lua_pushnumber(L,power_printEnergy(&pwrdata));
    return 1;
}

static int
lua_likwid_power_limitGet(lua_State* L)
{
    int err;
    int cpuId = lua_tonumber(L,1);
    int domain = lua_tonumber(L,2);
    double power = 0.0;
    double time = 0.0;
    err = power_limitGet(cpuId, domain, &power, &time);
    if (err < 0)
    {
        lua_pushnumber(L,err);
        return 1;
    }
    lua_pushnumber(L,power);
    lua_pushnumber(L,time);
    return 2;
}

static int
lua_likwid_power_limitSet(lua_State* L)
{
    int cpuId = lua_tonumber(L,1);
    int domain = lua_tonumber(L,2);
    double power = lua_tonumber(L,3);
    double time = lua_tonumber(L,4);
    int clamp  = lua_tonumber(L,5);
    lua_pushinteger(L, power_limitSet(cpuId, domain, power, time, clamp));
    return 1;
}

static int
lua_likwid_power_limitState(lua_State* L)
{
    int cpuId = lua_tonumber(L,1);
    int domain = lua_tonumber(L,2);
    lua_pushnumber(L,power_limitState(cpuId, domain));
    return 1;
}

static int
lua_likwid_getCpuClock(lua_State* L)
{
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    lua_pushnumber(L,timer_getCpuClock());
    return 1;
}

static int
lua_likwid_getCycleClock(lua_State* L)
{
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    lua_pushnumber(L,timer_getCycleClock());
    return 1;
}

static int
lua_sleep(lua_State* L)
{
#if LUA_VERSION_NUM == 501
    lua_pushnumber(L, timer_sleep(((lua_Integer)lua_tointeger(L,-1))));
#else
    lua_pushnumber(L, timer_sleep(((lua_Unsigned)lua_tointegerx(L,-1, NULL))));
#endif
    return 1;
}

static int
lua_likwid_startClock(lua_State* L)
{
    TimerData timer;
    double value;
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    timer_start(&timer);
    value = (double)timer.start.int64;
    lua_pushnumber(L, value);
    return 1;
}

static int
lua_likwid_stopClock(lua_State* L)
{
    TimerData timer;
    double value;
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    timer_stop(&timer);
    value = (double)timer.stop.int64;
    lua_pushnumber(L, value);
    return 1;
}

static int
lua_likwid_getClockCycles(lua_State* L)
{
    TimerData timer;
    double start, stop;
    start = lua_tonumber(L,1);
    stop = lua_tonumber(L,2);
    timer.start.int64 = (uint64_t)start;
    timer.stop.int64 = (uint64_t)stop;
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    lua_pushnumber(L, (double)timer_printCycles(&timer));
    return 1;
}

static int
lua_likwid_getClock(lua_State* L)
{
    TimerData timer;
    double runtime, start, stop;
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    start = lua_tonumber(L,1);
    stop = lua_tonumber(L,2);
    timer.start.int64 = (uint64_t)start;
    timer.stop.int64 = (uint64_t)stop;
    runtime = timer_print(&timer);
    lua_pushnumber(L, runtime);
    return 1;
}

static int
lua_likwid_initTemp(lua_State* L)
{
#if LUA_VERSION_NUM == 501
    int cpuid = ((lua_Integer)lua_tointeger(L,-1));
#else
    int cpuid = ((lua_Unsigned)lua_tointegerx(L,-1, NULL));
#endif
    thermal_init(cpuid);
    return 0;
}

static int
lua_likwid_readTemp(lua_State* L)
{
#if LUA_VERSION_NUM == 501
    int cpuid = ((lua_Integer)lua_tointeger(L,-1));
#else
    int cpuid = ((lua_Unsigned)lua_tointegerx(L,-1, NULL));
#endif
    uint32_t data;
    if (thermal_read(cpuid, &data)) {
        lua_pushstring(L,"Cannot read thermal data");
        lua_error(L);
    }
    lua_pushnumber(L, data);
    return 1;
}

static volatile int recv_sigint = 0;

static void signal_catcher(int signo)
{
    if (signo == SIGINT)
    {
        recv_sigint++;
    }
    return;
}

static int
lua_likwid_catch_signal(lua_State* L)
{
    signal(SIGINT,signal_catcher);
    return 0;
}

static int
lua_likwid_return_signal_state(lua_State* L)
{
    lua_pushnumber(L, recv_sigint);
    return 1;
}

static int
lua_likwid_send_signal(lua_State* L)
{
    int err = 0;
#if LUA_VERSION_NUM == 501
    pid_t pid = ((lua_Integer)lua_tointeger(L,1));
    int signal = ((lua_Integer)lua_tointeger(L,2));
#else
    pid_t pid = ((lua_Unsigned)lua_tointegerx(L,1, NULL));
    int signal = ((lua_Unsigned)lua_tointegerx(L,2, NULL));
#endif
    err = kill(pid, signal);
    lua_pushnumber(L, err);
    return 1;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
parse(char *line, char **argv)
{
     while (*line != '\0') {       /* if not the end of line ....... */
          while (*line == ' ' || *line == '\t' || *line == '\n')
               *line++ = '\0';     /* replace white spaces with 0    */
          *argv++ = line;          /* save the argument position     */
          while (*line != '\0' && *line != ' ' &&
                 *line != '\t' && *line != '\n')
               line++;             /* skip the argument until ...    */
     }
     *argv = '\0';                 /* mark the end of argument list  */
}

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static void
catch_sigchild(int signo)
{
    ;;
}

static int
lua_likwid_startProgram(lua_State* L)
{
    pid_t pid, ppid;
    int status;
    char *exec;
    char *argv[4096];
    exec = (char *)luaL_checkstring(L, 1);
    int nrThreads = luaL_checknumber(L,2);
    CpuTopology_t cputopo = get_cpuTopology();
    if (nrThreads > cputopo->numHWThreads)
    {
        lua_pushstring(L,"Number of threads greater than available HW threads");
        lua_error(L);
        return 0;
    }
    int *cpus = malloc(cputopo->numHWThreads * sizeof(int));
    if (!cpus)
        return 0;
    cpu_set_t cpuset;
    if (nrThreads > 0)
    {
        if (!lua_istable(L, -1)) {
          lua_pushstring(L,"No table given as second argument");
          lua_error(L);
          free(cpus);
        }
        for (status = 1; status<=nrThreads; status++)
        {
            lua_rawgeti(L,-1,status);
#if LUA_VERSION_NUM == 501
            cpus[status-1] = ((lua_Integer)lua_tointeger(L,-1));
#else
            cpus[status-1] = ((lua_Unsigned)lua_tointegerx(L,-1, NULL));
#endif
            lua_pop(L,1);
        }
    }
    else
    {
        for (nrThreads = 0; nrThreads < cpuid_topology.numHWThreads; nrThreads++)
            cpus[nrThreads] = cpuid_topology.threadPool[nrThreads].apicId;
        nrThreads = cpuid_topology.numHWThreads;
    }
    parse(exec, argv);
    ppid = getpid();
    pid = fork();
    if (pid < 0)
    {
        free(cpus);
        return 0;
    }
    else if ( pid == 0)
    {
        if (nrThreads > 0)
        {
            affinity_pinProcesses(nrThreads, cpus);
        }
        timer_sleep(10);
        status = execvp(*argv, argv);
        if (status < 0)
        {
            kill(ppid, SIGCHLD);
        }
        return 0;
    }
    else
    {
        signal(SIGCHLD, catch_sigchild);
        free(cpus);
        lua_pushnumber(L, pid);
    }
    return 1;
}

static int
lua_likwid_checkProgram(lua_State* L)
{
    int ret = -1;
    if (lua_gettop(L) == 1)
    {
        int status = 0;
        pid_t retpid = 0;
        pid_t pid = lua_tonumber(L, 1);
        retpid = waitpid(pid, &status, WNOHANG);
        if (retpid == pid)
        {
            if (WIFEXITED(status))
            {
                ret = WEXITSTATUS(status);
            }
            else if (WIFSIGNALED(status))
            {
                ret = 128 + WTERMSIG(status);
            }
        }
    }
    lua_pushinteger(L, (lua_Integer)ret);
    return 1;
}

static int
lua_likwid_killProgram(lua_State* L)
{
    pid_t pid = lua_tonumber(L, 1);
    kill(pid, SIGTERM);
    return 0;
}

static int
lua_likwid_waitpid(lua_State* L)
{
    int status = 0;
    int ret = -1;
    pid_t pid = lua_tonumber(L, 1);
    pid_t retpid = waitpid(pid, &status, 0);
    if (pid == retpid)
    {
        if (WIFEXITED(status))
        {
            ret = WEXITSTATUS(status);
        }
        else if (WIFSIGNALED(status))
        {
            ret = 128 + WTERMSIG(status);
        }
    }
    lua_pushinteger(L, (lua_Integer)ret);
    return 1;
}

static int
lua_likwid_memSweep(lua_State* L)
{
    int i;
    int nrThreads = luaL_checknumber(L,1);
    luaL_argcheck(L, nrThreads > 0, 1, "Thread count must be greater than 0");
    int cpus[nrThreads];
    if (!lua_istable(L, -1)) {
      lua_pushstring(L,"No table given as second argument");
      lua_error(L);
    }
    for (i = 1; i <= nrThreads; i++)
    {
        lua_rawgeti(L,-1,i);
#if LUA_VERSION_NUM == 501
        cpus[i-1] = ((lua_Integer)lua_tointeger(L,-1));
#else
        cpus[i-1] = ((lua_Unsigned)lua_tointegerx(L,-1, NULL));
#endif
        lua_pop(L,1);
    }
    memsweep_threadGroup(cpus, nrThreads);
    return 0;
}

static int
lua_likwid_memSweepDomain(lua_State* L)
{
    int domain = luaL_checknumber(L,1);
    luaL_argcheck(L, domain >= 0, 1, "Domain ID must be greater or equal 0");
    memsweep_domain(domain);
    return 0;
}

static int
lua_likwid_pinProcess(lua_State* L)
{
    int cpuID = luaL_checknumber(L,-2);
    int silent = luaL_checknumber(L,-1);
    luaL_argcheck(L, cpuID >= 0, 1, "CPU ID must be greater or equal 0");
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
        affinity = get_affinityDomains();
    }
    affinity_pinProcess(cpuID);
    if (!silent)
    {
#ifdef COLOR
            color_on(BRIGHT, COLOR);
#endif
            printf("[likwid-pin] Main PID -> core %d - OK",  cpuID);
#ifdef COLOR
            color_reset();
#endif
            printf("\n");
    }
    return 0;
}

static int
lua_likwid_pinThread(lua_State* L)
{
    int cpuID = luaL_checknumber(L,-2);
    int silent = luaL_checknumber(L,-1);
#ifdef HAS_SCHEDAFFINITY
    luaL_argcheck(L, cpuID >= 0, 1, "CPU ID must be greater or equal 0");
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
        affinity = get_affinityDomains();
    }
    affinity_pinThread(cpuID);
    if (!silent)
    {
#ifdef COLOR
            color_on(BRIGHT, COLOR);
#endif
            printf("[likwid-pin] PID %lu -> core %d - OK", gettid(), cpuID);
#ifdef COLOR
            color_reset();
#endif
            printf("\n");
    }
#else
    printf("Pinning of threads is not supported by your system\n")
#endif
    return 0;
}

static int
lua_likwid_setenv(lua_State* L)
{
    const char* element = (const char*)luaL_checkstring(L, -2);
    const char* value = (const char*)luaL_checkstring(L, -1);
    setenv(element, value, 1);
    return 0;
}

static int
lua_likwid_unsetenv(lua_State* L)
{
    const char* element = (const char*)luaL_checkstring(L, -1);
    unsetenv(element);
    return 0;
}

static int
lua_likwid_getpid(lua_State* L)
{
    lua_pushinteger(L, (lua_Integer)(getpid()));
    return 1;
}

static int
lua_likwid_setVerbosity(lua_State* L)
{
    int verbosity = lua_tointeger(L,-1);
    luaL_argcheck(L, (verbosity >= 0 && verbosity <= DEBUGLEV_DEVELOP), -1,
                "Verbosity must be between 0 (only errors) and 3 (developer)");
    perfmon_verbosity = verbosity;
    return 0;
}

static int
lua_likwid_getVerbosity(lua_State* L)
{
    lua_pushinteger(L, (lua_Integer)(perfmon_verbosity));
    return 1;
}

static int
lua_likwid_access(lua_State* L)
{
    int flags = 0;
    const char* file = (const char*)luaL_checkstring(L, 1);
    const char* perm = (const char*)luaL_checkstring(L, 2);
    if (!perm)
    {
        flags = F_OK;
    }
    else
    {
        for (int i=0;i<strlen(perm);i++)
        {
            if (perm[i] == 'r') {
                flags |= R_OK;
            } else if (perm[i] == 'w') {
                flags |= W_OK;
            } else if (perm[i] == 'x') {
                flags |= X_OK;
            } else if (perm[i] == 'e') {
                flags |= F_OK;
            }
        }
    }
    if (file)
    {
        lua_pushinteger(L, access(file, flags));
        return 1;
    }
    lua_pushinteger(L, -1);
    return 1;
}

static int
lua_likwid_markerInit(lua_State* L)
{
    likwid_markerInit();
    return 0;
}

static int
lua_likwid_markerThreadInit(lua_State* L)
{
    likwid_markerThreadInit();
    return 0;
}

static int
lua_likwid_markerClose(lua_State* L)
{
    likwid_markerClose();
    return 0;
}

static int
lua_likwid_markerNext(lua_State* L)
{
    likwid_markerNextGroup();
    return 0;
}

static int
lua_likwid_registerRegion(lua_State* L)
{
    const char* tag = (const char*)luaL_checkstring(L, -1);
    lua_pushinteger(L, likwid_markerRegisterRegion(tag));
    return 1;
}

static int
lua_likwid_startRegion(lua_State* L)
{
    const char* tag = (const char*)luaL_checkstring(L, -1);
    lua_pushinteger(L, likwid_markerStartRegion(tag));
    return 1;
}

static int
lua_likwid_stopRegion(lua_State* L)
{
    const char* tag = (const char*)luaL_checkstring(L, -1);
    lua_pushinteger(L, likwid_markerStopRegion(tag));
    return 1;
}

static int
lua_likwid_getRegion(lua_State* L)
{
    int i = 0;
    const char* tag = (const char*)luaL_checkstring(L, -2);
    int nr_events = perfmon_getNumberOfEvents(perfmon_getIdOfActiveGroup());
    double* events = NULL;
    double time = 0.0;
    int count = 0;
    events = (double*) malloc(nr_events * sizeof(double));
    if (events == NULL)
    {
        lua_pushstring(L,"Cannot allocate memory for event data\n");
        lua_error(L);
    }
    for (i = 0; i < nr_events; i++)
    {
        events[i] = 0.0;
    }
    likwid_markerGetRegion(tag, &nr_events, events, &time, &count);
    lua_pushinteger(L, nr_events);
    lua_newtable(L);
    for (i=0;i<nr_events;i++)
    {
        lua_pushinteger(L, i+1);
        lua_pushnumber(L, events[i]);
        lua_settable(L, -3);
    }
    lua_pushnumber(L, time);
    lua_pushinteger(L, count);
    free(events);
    return 4;
}

static int
lua_likwid_cpuFeatures_init(lua_State* L)
{
    cpuFeatures_init();
    return 0;
}

static int
lua_likwid_cpuFeatures_print(lua_State* L)
{
    int cpu = lua_tointeger(L,-1);
    cpuFeatures_print(cpu);
    return 0;
}

static int
lua_likwid_cpuFeatures_get(lua_State* L)
{
    int cpu = lua_tointeger(L,-2);
    CpuFeature feature = lua_tointeger(L,-1);
    lua_pushinteger(L, cpuFeatures_get(cpu, feature));
    return 1;
}

static int
lua_likwid_cpuFeatures_name(lua_State* L)
{
    char* name = NULL;
#if LUA_VERSION_NUM == 501
    CpuFeature feature = ((lua_Integer)lua_tointeger(L,-1));
#else
    CpuFeature feature = ((lua_Unsigned)lua_tointegerx(L,-1, NULL));
#endif
    name = cpuFeatures_name(feature);
    if (name != NULL)
    {
        lua_pushstring(L, name);
        return 1;
    }
    return 0;
}

static int
lua_likwid_cpuFeatures_enable(lua_State* L)
{
    int cpu = lua_tointeger(L,-3);
    CpuFeature feature = lua_tointeger(L,-2);
    int verbose = lua_tointeger(L,-1);
    lua_pushinteger(L, cpuFeatures_enable(cpu, feature, verbose));
    return 1;
}

static int
lua_likwid_cpuFeatures_disable(lua_State* L)
{
    int cpu = lua_tointeger(L,-3);
    CpuFeature feature = lua_tointeger(L,-2);
    int verbose = lua_tointeger(L,-1);
    lua_pushinteger(L, cpuFeatures_disable(cpu, feature, verbose));
    return 1;
}

static int
lua_likwid_markerFile_read(lua_State* L)
{
    const char* filename = (const char*)luaL_checkstring(L, -1);
    int ret = perfmon_readMarkerFile(filename);
    lua_pushinteger(L, ret);
    return 1;
}

static int
lua_likwid_markerFile_destroy(lua_State* L)
{
    perfmon_destroyMarkerResults();
    return 0;
}

static int
lua_likwid_markerNumRegions(lua_State* L)
{
    lua_pushinteger(L, perfmon_getNumberOfRegions());
    return 1;
}

static int
lua_likwid_markerRegionGroup(lua_State* L)
{
    int region = lua_tointeger(L,-1);
    lua_pushinteger(L, perfmon_getGroupOfRegion(region-1)+1);
    return 1;
}

static int
lua_likwid_markerRegionTag(lua_State* L)
{
    int region = lua_tointeger(L,-1);
    lua_pushstring(L, perfmon_getTagOfRegion(region-1));
    return 1;
}

static int
lua_likwid_markerRegionEvents(lua_State* L)
{
    int region = lua_tointeger(L,-1);
    lua_pushinteger(L, perfmon_getEventsOfRegion(region-1));
    return 1;
}

static int
lua_likwid_markerRegionThreads(lua_State* L)
{
    int region = lua_tointeger(L,-1);
    lua_pushinteger(L, perfmon_getThreadsOfRegion(region-1));
    return 1;
}

static int
lua_likwid_markerRegionCpulist(lua_State* L)
{
    int i = 0;
    int region = lua_tointeger(L,-1);
    int* cpulist;
    int regionCPUs = 0;
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
        cpuinfo = get_cpuInfo();
        cputopo = get_cpuTopology();
    }
    if ((topology_isInitialized) && (cpuinfo == NULL))
    {
        cpuinfo = get_cpuInfo();
    }
    if ((topology_isInitialized) && (cputopo == NULL))
    {
        cputopo = get_cpuTopology();
    }
    cpulist = (int*)malloc(cputopo->numHWThreads * sizeof(int));
    if (cpulist == NULL)
    {
        return 0;
    }
    regionCPUs = perfmon_getCpulistOfRegion(region-1, cputopo->numHWThreads, cpulist);
    if (regionCPUs > 0)
    {
        lua_newtable(L);
        for (i=0; i < regionCPUs; i++)
        {
            lua_pushinteger(L, i+1);
            lua_pushinteger(L, cpulist[i]);
            lua_settable(L, -3);
        }
        return 1;
    }
    return 0;
}

static int
lua_likwid_markerRegionTime(lua_State* L)
{
    int region = lua_tointeger(L,-2);
    int thread = lua_tointeger(L,-1);
    lua_pushnumber(L, perfmon_getTimeOfRegion(region-1, thread-1));
    return 1;
}

static int
lua_likwid_markerRegionCount(lua_State* L)
{
    int region = lua_tointeger(L,-2);
    int thread = lua_tointeger(L,-1);
    lua_pushinteger(L, perfmon_getCountOfRegion(region-1, thread-1));
    return 1;
}

static int
lua_likwid_markerRegionResult(lua_State* L)
{
    int region = lua_tointeger(L,-3);
    int event = lua_tointeger(L,-2);
    int thread = lua_tointeger(L,-1);
    lua_pushnumber(L, perfmon_getResultOfRegionThread(region-1, event-1, thread-1));
    return 1;
}

static int
lua_likwid_markerRegionMetric(lua_State* L)
{
    int region = lua_tointeger(L,-3);
    int metric = lua_tointeger(L,-2);
    int thread = lua_tointeger(L,-1);
    lua_pushnumber(L, perfmon_getMetricOfRegionThread(region-1, metric-1, thread-1));
    return 1;
}

static int
lua_likwid_getCpuClockCurrent(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-1);
    lua_pushnumber(L, freq_getCpuClockCurrent(cpu_id));
    return 1;
}

static int
lua_likwid_getCpuClockMin(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-1);
    lua_pushnumber(L, freq_getCpuClockMin(cpu_id));
    return 1;
}

static int
lua_likwid_setCpuClockMin(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-2);
    const unsigned long freq = lua_tointeger(L,-1);
    lua_pushnumber(L, freq_setCpuClockMin(cpu_id, freq));
    return 1;
}

static int
lua_likwid_getCpuClockMax(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-1);
    lua_pushnumber(L, freq_getCpuClockMax(cpu_id));
    return 1;
}

static int
lua_likwid_setCpuClockMax(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-2);
    const unsigned long freq = lua_tointeger(L,-1);
    lua_pushnumber(L, freq_setCpuClockMax(cpu_id, freq));
    return 1;
}

static int
lua_likwid_setTurbo(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-2);
    const int turbo = lua_tointeger(L,-1);
    lua_pushnumber(L, freq_setTurbo(cpu_id, turbo));
    return 1;
}

static int
lua_likwid_getTurbo(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-1);
    lua_pushnumber(L, freq_getTurbo(cpu_id));
    return 1;
}

static int
lua_likwid_getGovernor(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-1);
    char *gov = freq_getGovernor(cpu_id);
    if (gov)
    {
        lua_pushstring(L, gov);
        free(gov);
    }
    else
        lua_pushnil(L);
    return 1;
}

static int
lua_likwid_setGovernor(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-2);
    const char* gov = (const char*)luaL_checkstring(L, -1);
    lua_pushnumber(L, freq_setGovernor(cpu_id, gov));
    return 1;
}

static int
lua_likwid_getAvailFreq(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-1);
    char* avail = freq_getAvailFreq(cpu_id);
    if (avail)
    {
        lua_pushstring(L, avail);
        free(avail);
    }
    else
        lua_pushnil(L);
    return 1;
}

static int
lua_likwid_getAvailGovs(lua_State* L)
{
    const int cpu_id = lua_tointeger(L,-1);
    char* avail = freq_getAvailGovs(cpu_id);
    if (avail)
        lua_pushstring(L, avail);
    else
        lua_pushnil(L);
    return 1;
}


static int
lua_likwid_setUncoreFreqMin(lua_State* L)
{
    const int socket_id = lua_tointeger(L,-2);
    const uint64_t freq = lua_tointeger(L,-1);
    int err = freq_setUncoreFreqMin(socket_id, freq);
    lua_pushinteger(L, err);
    return 1;
}

static int
lua_likwid_getUncoreFreqMin(lua_State* L)
{
    const int socket_id = lua_tointeger(L,-1);
    uint64_t freq = freq_getUncoreFreqMin(socket_id);
    lua_pushinteger(L, freq);
    return 1;
}

static int
lua_likwid_setUncoreFreqMax(lua_State* L)
{
    const int socket_id = lua_tointeger(L,-2);
    const uint64_t freq = lua_tointeger(L,-1);
    int err = freq_setUncoreFreqMax(socket_id, freq);
    lua_pushinteger(L, err);
    return 1;
}

static int
lua_likwid_getUncoreFreqMax(lua_State* L)
{
    const int socket_id = lua_tointeger(L,-1);
    uint64_t freq = freq_getUncoreFreqMax(socket_id);
    lua_pushinteger(L, freq);
    return 1;
}

static int
lua_likwid_getUncoreFreqCur(lua_State* L)
{
    const int socket_id = lua_tointeger(L,-1);
    uint64_t freq = freq_getUncoreFreqCur(socket_id);
    lua_pushinteger(L, freq);
    return 1;
}

static int
lua_likwid_getuid(lua_State* L)
{
    int r = geteuid();
    lua_pushnumber(L, r);
    return 1;
}

static int
lua_likwid_geteuid(lua_State* L)
{
    int r = geteuid();
    lua_pushnumber(L, r);
    return 1;
}

static int
lua_likwid_setuid(lua_State* L)
{
    int id = (int) lua_tonumber(L, 1);
    int r = setuid((uid_t) id);
    if (r == 0)
    {
        lua_pushboolean(L, 1);
    }
    else
    {
        lua_pushboolean(L, 0);
    }
    return 1;
}

static int
lua_likwid_seteuid(lua_State* L)
{
    int id = (int) lua_tonumber(L, 1);
    int r = seteuid((uid_t) id);
    if (r == 0)
    {
        lua_pushboolean(L, 1);
    }
    else
    {
        lua_pushboolean(L, 0);
    }
    return 1;
}

static int
lua_likwid_setresuid(lua_State* L)
{
    int ruid = (int) lua_tonumber(L, 1);
    int euid = (int) lua_tonumber(L, 2);
    int suid = (int) lua_tonumber(L, 3);
    int r = setresuid((uid_t)ruid, (uid_t)euid, (uid_t)suid);
    if (r == 0)
    {
        lua_pushboolean(L, 1);
    }
    else
    {
        lua_pushboolean(L, 0);
    }
    return 1;
}

static int
lua_likwid_setresuser(lua_State* L)
{
    const char* ruser = (const char*) luaL_checkstring(L, 1);
    const char* euser = (const char*) luaL_checkstring(L, 2);
    const char* suser = (const char*) luaL_checkstring(L, 3);
    struct passwd *p;
    p = getpwnam(ruser);
    if ( p == NULL )
    {
        lua_pushboolean(L, 0);
        return 1;
    }
    uid_t ruid = p->pw_uid;
    p = getpwnam(euser);
    if ( p == NULL )
    {
        lua_pushboolean(L, 0);
        return 1;
    }
    uid_t euid = p->pw_uid;
    p = getpwnam(suser);
    if ( p == NULL )
    {
        lua_pushboolean(L, 0);
        return 1;
    }
    uid_t suid = p->pw_uid;

    int r = setresuid(ruid, euid, suid);
    if (r == 0)
    {
        lua_pushboolean(L, 1);
    }
    else
    {
        lua_pushboolean(L, 0);
    }
    return 1;
}



/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int __attribute__ ((visibility ("default") ))
luaopen_liblikwid(lua_State* L){
    // Configuration functions
    lua_register(L, "likwid_getConfiguration", lua_likwid_getConfiguration);
    lua_register(L, "likwid_setGroupPath", lua_likwid_setGroupPath);
    lua_register(L, "likwid_putConfiguration", lua_likwid_putConfiguration);
    // Perfmon functions
    //lua_register(L, "accessClient_setaccessmode",lua_accessClient_setaccessmode);
    lua_register(L, "likwid_setAccessClientMode",lua_likwid_setAccessMode);
    lua_register(L, "likwid_getAccessClientMode",lua_likwid_getAccessMode);
    lua_register(L, "likwid_init",lua_likwid_init);
    lua_register(L, "likwid_addEventSet", lua_likwid_addEventSet);
    lua_register(L, "likwid_setupCounters",lua_likwid_setupCounters);
    lua_register(L, "likwid_startCounters",lua_likwid_startCounters);
    lua_register(L, "likwid_stopCounters",lua_likwid_stopCounters);
    lua_register(L, "likwid_readCounters",lua_likwid_readCounters);
    lua_register(L, "likwid_switchGroup",lua_likwid_switchGroup);
    lua_register(L, "likwid_finalize",lua_likwid_finalize);
    lua_register(L, "likwid_getEventsAndCounters", lua_likwid_getEventsAndCounters);
    // Perfmon results functions
    lua_register(L, "likwid_getResult",lua_likwid_getResult);
    lua_register(L, "likwid_getLastResult",lua_likwid_getLastResult);
    lua_register(L, "likwid_getMetric",lua_likwid_getMetric);
    lua_register(L, "likwid_getLastMetric",lua_likwid_getLastMetric);
    lua_register(L, "likwid_getNumberOfGroups",lua_likwid_getNumberOfGroups);
    lua_register(L, "likwid_getRuntimeOfGroup", lua_likwid_getRuntimeOfGroup);
    lua_register(L, "likwid_getIdOfActiveGroup",lua_likwid_getIdOfActiveGroup);
    lua_register(L, "likwid_getNumberOfEvents",lua_likwid_getNumberOfEvents);
    lua_register(L, "likwid_getNumberOfMetrics",lua_likwid_getNumberOfMetrics);
    lua_register(L, "likwid_getNumberOfThreads",lua_likwid_getNumberOfThreads);
    lua_register(L, "likwid_getNameOfEvent",lua_likwid_getNameOfEvent);
    lua_register(L, "likwid_getNameOfCounter",lua_likwid_getNameOfCounter);
    lua_register(L, "likwid_getNameOfMetric",lua_likwid_getNameOfMetric);
    lua_register(L, "likwid_getNameOfGroup",lua_likwid_getNameOfGroup);
    lua_register(L, "likwid_getGroups",lua_likwid_getGroups);
    lua_register(L, "likwid_getShortInfoOfGroup",lua_likwid_getShortInfoOfGroup);
    lua_register(L, "likwid_getLongInfoOfGroup",lua_likwid_getLongInfoOfGroup);
    // Topology functions
    lua_register(L, "likwid_getCpuInfo",lua_likwid_getCpuInfo);
    lua_register(L, "likwid_getCpuTopology",lua_likwid_getCpuTopology);
    lua_register(L, "likwid_putTopology",lua_likwid_putTopology);
    lua_register(L, "likwid_getNumaInfo",lua_likwid_getNumaInfo);
    lua_register(L, "likwid_putNumaInfo",lua_likwid_putNumaInfo);
    lua_register(L, "likwid_setMemInterleaved", lua_likwid_setMemInterleaved);
    lua_register(L, "likwid_getAffinityInfo",lua_likwid_getAffinityInfo);
    lua_register(L, "likwid_putAffinityInfo",lua_likwid_putAffinityInfo);
    lua_register(L, "likwid_getPowerInfo",lua_likwid_getPowerInfo);
    lua_register(L, "likwid_putPowerInfo",lua_likwid_putPowerInfo);
    lua_register(L, "likwid_getOnlineDevices", lua_likwid_getOnlineDevices);
    lua_register(L, "likwid_printSupportedCPUs", lua_likwid_printSupportedCPUs);
    // CPU string parse functions
    lua_register(L, "likwid_cpustr_to_cpulist",lua_likwid_cpustr_to_cpulist);
    lua_register(L, "likwid_nodestr_to_nodelist",lua_likwid_nodestr_to_nodelist);
    lua_register(L, "likwid_sockstr_to_socklist",lua_likwid_sockstr_to_socklist);
    // Timer functions
    lua_register(L, "likwid_getCpuClock",lua_likwid_getCpuClock);
    lua_register(L, "likwid_getCycleClock",lua_likwid_getCycleClock);
    lua_register(L, "likwid_startClock",lua_likwid_startClock);
    lua_register(L, "likwid_stopClock",lua_likwid_stopClock);
    lua_register(L, "likwid_getClockCycles",lua_likwid_getClockCycles);
    lua_register(L, "likwid_getClock",lua_likwid_getClock);
    lua_register(L, "sleep",lua_sleep);
    // Power functions
    lua_register(L, "likwid_startPower",lua_likwid_startPower);
    lua_register(L, "likwid_stopPower",lua_likwid_stopPower);
    lua_register(L, "likwid_printEnergy",lua_likwid_printEnergy);
    lua_register(L, "likwid_powerLimitGet",lua_likwid_power_limitGet);
    lua_register(L, "likwid_powerLimitSet",lua_likwid_power_limitSet);
    lua_register(L, "likwid_powerLimitState",lua_likwid_power_limitState);
    // Temperature functions
    lua_register(L, "likwid_initTemp",lua_likwid_initTemp);
    lua_register(L, "likwid_readTemp",lua_likwid_readTemp);
    // MemSweep functions
    lua_register(L, "likwid_memSweep", lua_likwid_memSweep);
    lua_register(L, "likwid_memSweepDomain", lua_likwid_memSweepDomain);
    // Pinning functions
    lua_register(L, "likwid_pinProcess", lua_likwid_pinProcess);
    lua_register(L, "likwid_pinThread", lua_likwid_pinThread);
    // Helper functions
    lua_register(L, "likwid_setenv", lua_likwid_setenv);
    lua_register(L, "likwid_unsetenv", lua_likwid_unsetenv);
    lua_register(L, "likwid_getpid", lua_likwid_getpid);
    lua_register(L, "likwid_access", lua_likwid_access);
    lua_register(L, "likwid_startProgram", lua_likwid_startProgram);
    lua_register(L, "likwid_checkProgram", lua_likwid_checkProgram);
    lua_register(L, "likwid_killProgram", lua_likwid_killProgram);
    lua_register(L, "likwid_catchSignal", lua_likwid_catch_signal);
    lua_register(L, "likwid_getSignalState", lua_likwid_return_signal_state);
    lua_register(L, "likwid_waitpid", lua_likwid_waitpid);
    lua_register(L, "likwid_sendSignal", lua_likwid_send_signal);
    // Verbosity functions
    lua_register(L, "likwid_setVerbosity", lua_likwid_setVerbosity);
    lua_register(L, "likwid_getVerbosity", lua_likwid_getVerbosity);
    // Marker API functions
    lua_register(L, "likwid_markerInit", lua_likwid_markerInit);
    lua_register(L, "likwid_markerThreadInit", lua_likwid_markerThreadInit);
    lua_register(L, "likwid_markerNextGroup", lua_likwid_markerNext);
    lua_register(L, "likwid_markerClose", lua_likwid_markerClose);
    lua_register(L, "likwid_registerRegion", lua_likwid_registerRegion);
    lua_register(L, "likwid_startRegion", lua_likwid_startRegion);
    lua_register(L, "likwid_stopRegion", lua_likwid_stopRegion);
    lua_register(L, "likwid_getRegion", lua_likwid_getRegion);
    // CPU feature manipulation functions
    lua_register(L, "likwid_cpuFeaturesInit", lua_likwid_cpuFeatures_init);
    lua_register(L, "likwid_cpuFeaturesGet", lua_likwid_cpuFeatures_get);
    lua_register(L, "likwid_cpuFeaturesEnable", lua_likwid_cpuFeatures_enable);
    lua_register(L, "likwid_cpuFeaturesDisable", lua_likwid_cpuFeatures_disable);
    // Marker API related functions
    lua_register(L, "likwid_readMarkerFile", lua_likwid_markerFile_read);
    lua_register(L, "likwid_destroyMarkerFile", lua_likwid_markerFile_destroy);
    lua_register(L, "likwid_markerNumRegions", lua_likwid_markerNumRegions);
    lua_register(L, "likwid_markerRegionGroup", lua_likwid_markerRegionGroup);
    lua_register(L, "likwid_markerRegionTag", lua_likwid_markerRegionTag);
    lua_register(L, "likwid_markerRegionEvents", lua_likwid_markerRegionEvents);
    lua_register(L, "likwid_markerRegionThreads", lua_likwid_markerRegionThreads);
    lua_register(L, "likwid_markerRegionCpulist", lua_likwid_markerRegionCpulist);
    lua_register(L, "likwid_markerRegionTime", lua_likwid_markerRegionTime);
    lua_register(L, "likwid_markerRegionCount", lua_likwid_markerRegionCount);
    lua_register(L, "likwid_markerRegionResult", lua_likwid_markerRegionResult);
    lua_register(L, "likwid_markerRegionMetric", lua_likwid_markerRegionMetric);
    // CPU frequency functions
    lua_register(L, "likwid_getCpuClockCurrent", lua_likwid_getCpuClockCurrent);
    lua_register(L, "likwid_getCpuClockMin", lua_likwid_getCpuClockMin);
    lua_register(L, "likwid_setCpuClockMin", lua_likwid_setCpuClockMin);
    lua_register(L, "likwid_getCpuClockMax", lua_likwid_getCpuClockMax);
    lua_register(L, "likwid_setCpuClockMax", lua_likwid_setCpuClockMax);
    lua_register(L, "likwid_getGovernor", lua_likwid_getGovernor);
    lua_register(L, "likwid_setGovernor", lua_likwid_setGovernor);
    lua_register(L, "likwid_getAvailFreq", lua_likwid_getAvailFreq);
    lua_register(L, "likwid_getAvailGovs", lua_likwid_getAvailGovs);
    lua_register(L, "likwid_setTurbo", lua_likwid_setTurbo);
    lua_register(L, "likwid_getTurbo", lua_likwid_getTurbo);
    lua_register(L, "likwid_setUncoreFreqMin", lua_likwid_setUncoreFreqMin);
    lua_register(L, "likwid_getUncoreFreqMin", lua_likwid_getUncoreFreqMin);
    lua_register(L, "likwid_setUncoreFreqMax", lua_likwid_setUncoreFreqMax);
    lua_register(L, "likwid_getUncoreFreqMax", lua_likwid_getUncoreFreqMax);
    lua_register(L, "likwid_getUncoreFreqCur", lua_likwid_getUncoreFreqCur);
    // setuid&friends
    lua_register(L, "likwid_getuid", lua_likwid_getuid);
    lua_register(L, "likwid_geteuid", lua_likwid_geteuid);
    lua_register(L, "likwid_setuid", lua_likwid_setuid);
    lua_register(L, "likwid_seteuid", lua_likwid_seteuid);
    lua_register(L, "likwid_setresuid", lua_likwid_setresuid);
    lua_register(L, "likwid_setresuser", lua_likwid_setresuser);
#ifdef __MIC__
    setuid(0);
    seteuid(0);
#endif
    return 0;
}

