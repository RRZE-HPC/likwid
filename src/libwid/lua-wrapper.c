#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lua.h>                               /* Always include this */
#include <lauxlib.h>                           /* Always include this */
#include <lualib.h>                            /* Always include this */

#include <perfmon.h>
#include <topology.h>
#include <accessClient.h>



static int iaccessClient_setaccessmode(lua_State *L)
{
	int flag = lua_tonumber(L,1);
	if ((flag != 0) && (flag != 1))
	{
		lua_pushnumber(L,1);
		return 1;
	}
	accessClient_setaccessmode(flag);
	return 0;
}

static int ilikwid_init(lua_State *L)
{
	int ret;
	int nrThreads = lua_tonumber(L,1);
	int reformat_cpulist[nrThreads];
	int* cpulist = (int*)lua_topointer(L,2);
	topology_init();
	numa_init();
	for(ret=1;ret<=nrThreads;ret++)
	{
		reformat_cpulist[ret-1] = cpulist[ret];
	}
	ret = perfmon_init(nrThreads, reformat_cpulist);
	lua_pushnumber(L,ret);
	return 1;	
}


static int ilikwid_addEventSet(lua_State *L)
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

static int ilikwid_setupCounters(lua_State *L)
{
	int ret;
	int groupId = lua_tonumber(L,1);
	ret = perfmon_setupCounters(groupId);
	lua_pushnumber(L,ret);
	return 1;
}


static int ilikwid_startCounters(lua_State *L)
{
	int ret;
	ret = perfmon_startCounters();
	lua_pushnumber(L,ret);
	return 1;
}

static int ilikwid_stopCounters(lua_State *L)
{
	int ret;
	ret = perfmon_stopCounters();
	lua_pushnumber(L,ret);
	return 1;
}

static int ilikwid_readCounters(lua_State *L)
{
	int ret;
	ret = perfmon_readCounters();
	lua_pushnumber(L,ret);
	return 1;
}

static int ilikwid_finalize(lua_State *L)
{
	perfmon_finalize();
	return 0;
}

static int ilikwid_getResult(lua_State *L)
{
	int groupId, eventId, threadId;
	uint64_t result = 0;
	groupId = lua_tonumber(L,1);
	eventId = lua_tonumber(L,2);
	threadId = lua_tonumber(L,3);
	result = perfmon_getResults(groupId, eventId, threadId);
	lua_pushnumber(L,result);
	return 1;
}

int luaopen_luawid(lua_State *L){
	lua_register(L, "accessClient_setaccessmode",iaccessClient_setaccessmode);
	lua_register(L, "likwid_init",ilikwid_init);
	lua_register(L, "likwid_addEventSet", ilikwid_addEventSet);
	lua_register(L, "likwid_setupCounters",ilikwid_setupCounters);
	lua_register(L, "likwid_startCounters",ilikwid_startCounters);
	lua_register(L, "likwid_stopCounters",ilikwid_stopCounters);
	lua_register(L, "likwid_readCounters",ilikwid_readCounters);
	lua_register(L, "likwid_finalize",ilikwid_finalize);
	lua_register(L, "likwid_getResult",ilikwid_getResult);
	return 0;
}
