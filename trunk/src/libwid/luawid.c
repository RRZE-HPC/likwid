#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lua.h>                               /* Always include this */
#include <lauxlib.h>                           /* Always include this */
#include <lualib.h>                            /* Always include this */

#include <perfmon.h>
#include <topology.h>
#include <accessClient.h>

static int topology_isInitialized = 0;
static int numa_isInitialized = 0;
static int affinity_isInitialized = 0;
static int perfmon_isInitialized = 0;

static int iaccessClient_setaccessmode(lua_State* L)
{
    int flag;
    flag = luaL_checknumber(L,1);
    luaL_argcheck(L, flag >= 0 && flag <= 1, 1, "invalid access mode, only 0 (direct) and 1 (accessdaemon) allowed");
    accessClient_setaccessmode(flag);
    return 0;
}

static int ilikwid_init(lua_State* L)
{
    int ret;
    int nrThreads = luaL_checknumber(L,1);
    int* cpulist = (int*)lua_topointer(L,2);
    luaL_argcheck(L, nrThreads > 0, 1, "Thread count must be greater than 0");
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
    }
    if (numa_isInitialized == 0)
    {
        numa_init();
        numa_isInitialized = 1;
    }
    if (perfmon_isInitialized == 0)
    {
        ret = perfmon_init(nrThreads, &(cpulist[1]));
        perfmon_isInitialized = 1;
    } else {
        ret = -1;
    }
    
    lua_pushnumber(L,ret);
    return 1;    
}


static int ilikwid_addEventSet(lua_State* L)
{
    int groupId, n;
    char eventString[1000];
    const char* tmpString;
    n = lua_gettop(L);
    tmpString = lua_tostring(L,n);
    strcpy(eventString, tmpString);
    groupId = perfmon_addEventSet(eventString);
    lua_pushnumber(L, groupId);
    return 1;
}

static int ilikwid_setupCounters(lua_State* L)
{
    int ret;
    int groupId = lua_tonumber(L,1);
    ret = perfmon_setupCounters(groupId);
    lua_pushnumber(L,ret);
    return 1;
}


static int ilikwid_startCounters(lua_State* L)
{
    int ret;
    ret = perfmon_startCounters();
    lua_pushnumber(L,ret);
    return 1;
}

static int ilikwid_stopCounters(lua_State* L)
{
    int ret;
    ret = perfmon_stopCounters();
    lua_pushnumber(L,ret);
    return 1;
}

static int ilikwid_readCounters(lua_State* L)
{
    int ret;
    ret = perfmon_readCounters();
    lua_pushnumber(L,ret);
    return 1;
}

static int ilikwid_finalize(lua_State* L)
{
    perfmon_finalize();
    return 0;
}

static int ilikwid_getResult(lua_State* L)
{
    int groupId, eventId, threadId;
    uint64_t result = 0;
    groupId = lua_tonumber(L,1);
    eventId = lua_tonumber(L,2);
    threadId = lua_tonumber(L,3);
    result = perfmon_getResult(groupId, eventId, threadId);
    lua_pushnumber(L,result);
    return 1;
}

static int ilikwid_getNumberOfGroups(lua_State* L)
{
    int number;
    number = perfmon_getNumberOfGroups();
    lua_pushnumber(L,number);
    return 1;
}

static int ilikwid_getNumberOfActiveGroup(lua_State* L)
{
    int number;
    number = perfmon_getNumberOfActiveGroup();
    lua_pushnumber(L,number);
    return 1;
}

static int ilikwid_getNumberOfEvents(lua_State* L)
{
    int number, groupId;
    groupId = lua_tonumber(L,1);
    number = perfmon_getNumberOfEvents(groupId);
    lua_pushnumber(L,number);
    return 1;
}

static int ilikwid_getNumberOfThreads(lua_State* L)
{
    int number;
    number = perfmon_getNumberOfThreads();
    lua_pushnumber(L,number);
    return 1;
}

/*
    uint32_t family;
    uint32_t model;
    uint32_t stepping;
    uint64_t clock;
    int      turbo;
    char*  name;
    char*  features;
    int         isIntel;
    uint32_t featureFlags;
    uint32_t perf_version;
    uint32_t perf_num_ctr;
    uint32_t perf_width_ctr;
    uint32_t perf_num_fixed_ctr;
*/

static int ilikwid_getCpuInfo(lua_State* L)
{
    CpuInfo_t cpuinfo = get_cpuInfo();
    lua_newtable(L);
    lua_pushstring(L,"family");
    lua_pushunsigned(L,cpuinfo->family);
    lua_settable(L,-3);
    lua_pushstring(L,"model");
    lua_pushunsigned(L,cpuinfo->model);
    lua_settable(L,-3);
    lua_pushstring(L,"stepping");
    lua_pushunsigned(L,cpuinfo->stepping);
    lua_settable(L,-3);
    lua_pushstring(L,"clock");
    lua_pushunsigned(L,cpuinfo->clock);
    lua_settable(L,-3);
    lua_pushstring(L,"turbo");
    lua_pushinteger(L,cpuinfo->turbo);
    lua_settable(L,-3);
    lua_pushstring(L,"name");
    lua_pushstring(L,cpuinfo->name);
    lua_settable(L,-3);
    lua_pushstring(L,"features");
    lua_pushstring(L,cpuinfo->features);
    lua_settable(L,-3);
    lua_pushstring(L,"isIntel");
    lua_pushinteger(L,cpuinfo->isIntel);
    lua_settable(L,-3);
    lua_pushstring(L,"featureFlags");
    lua_pushunsigned(L,cpuinfo->featureFlags);
    lua_settable(L,-3);
    lua_pushstring(L,"perf_version");
    lua_pushunsigned(L,cpuinfo->perf_version);
    lua_settable(L,-3);
    lua_pushstring(L,"perf_num_ctr");
    lua_pushunsigned(L,cpuinfo->perf_num_ctr);
    lua_settable(L,-3);
    lua_pushstring(L,"perf_width_ctr");
    lua_pushunsigned(L,cpuinfo->perf_width_ctr);
    lua_settable(L,-3);
    lua_pushstring(L,"perf_num_fixed_ctr");
    lua_pushunsigned(L,cpuinfo->perf_num_fixed_ctr);
    lua_settable(L,-3);
    return 1;
}

int luaopen_luawid(lua_State* L){
    // Perfmon functions
    lua_register(L, "accessClient_setaccessmode",iaccessClient_setaccessmode);
    lua_register(L, "likwid_init",ilikwid_init);
    lua_register(L, "likwid_addEventSet", ilikwid_addEventSet);
    lua_register(L, "likwid_setupCounters",ilikwid_setupCounters);
    lua_register(L, "likwid_startCounters",ilikwid_startCounters);
    lua_register(L, "likwid_stopCounters",ilikwid_stopCounters);
    lua_register(L, "likwid_readCounters",ilikwid_readCounters);
    lua_register(L, "likwid_finalize",ilikwid_finalize);
    // Perfmon results functions 
    lua_register(L, "likwid_getResult",ilikwid_getResult);
    lua_register(L, "likwid_getNumberOfGroups",ilikwid_getNumberOfGroups);
    lua_register(L, "likwid_getNumberOfActiveGroup",ilikwid_getNumberOfActiveGroup);
    lua_register(L, "likwid_getNumberOfEvents",ilikwid_getNumberOfEvents);
    lua_register(L, "likwid_getNumberOfThreads",ilikwid_getNumberOfThreads);
    // Topology functions
    lua_register(L, "likwid_getCpuInfo",ilikwid_getCpuInfo);
    return 0;
}
