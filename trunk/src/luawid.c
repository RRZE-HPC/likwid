#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include <lua.h>                               /* Always include this */
#include <lauxlib.h>                           /* Always include this */
#include <lualib.h>                            /* Always include this */


#include <perfmon.h>
#include <topology.h>
#include <numa.h>
#include <affinity.h>
#include <power.h>
#include <bstrlib.h>
#include <accessClient.h>
#include <thermal.h>
#include <memsweep.h>
#include <daemon.h>
#include <configuration.h>
#include <msr.h>

#ifdef COLOR
#include <textcolor.h>
#endif

static int topology_isInitialized = 0;
static int numa_isInitialized = 0;
static int affinity_isInitialized = 0;
static int perfmon_isInitialized = 0;
static int timer_isInitialized = 0;
static int power_isInitialized = 0;
static int config_isInitialized = 0;


static int lua_likwid_getConfiguration(lua_State* L)
{
    Configuration_t config;
    if (config_isInitialized == 0)
    {
        init_configuration();
        config_isInitialized = 1;
    }
    config = get_configuration();
    lua_newtable(L);
    lua_pushstring(L, "topologyFile");
    lua_pushstring(L, config->topologyCfgFileName);
    lua_settable(L,-3);
    lua_pushstring(L, "daemonPath");
    lua_pushstring(L, config->daemonPath);
    lua_settable(L,-3);
    lua_pushstring(L, "daemonMode");
    lua_pushinteger(L, (int)config->daemonMode);
    lua_settable(L,-3);
    lua_pushstring(L, "maxNumThreads");
    lua_pushinteger(L, config->maxNumThreads);
    lua_settable(L,-3);
    lua_pushstring(L, "maxNumNodes");
    lua_pushinteger(L, config->maxNumNodes);
    lua_settable(L,-3);
    lua_pushstring(L, "maxHashTabelSize");
    lua_pushinteger(L, config->maxHashTableSize);
    lua_settable(L,-3);
    return 1;
}

static int lua_likwid_putConfiguration(lua_State* L)
{
    if (config_isInitialized == 1)
    {
        destroy_configuration();
        config_isInitialized = 0;
    }
    return 0;
}

static int lua_accessClient_setaccessmode(lua_State* L)
{
    int flag;
    flag = luaL_checknumber(L,1);
    luaL_argcheck(L, flag >= 0 && flag <= 1, 1, "invalid access mode, only 0 (direct) and 1 (accessdaemon) allowed");
    accessClient_setaccessmode(flag);
    if (flag == 0 && msr_init(0) != 0)
    {
        lua_pushnumber(L,1);
        return 1;
    }
    else if (flag == 1)
    {
        perfmon_accessClientInit();
    }
    lua_pushnumber(L,0);
    return 1;
}

static int lua_likwid_init(lua_State* L)
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
        cpus[ret-1] = lua_tounsigned(L,-1);
        lua_pop(L,1);
    }
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
        ret = perfmon_init(nrThreads, &(cpus[0]));
        if (ret != 0)
        {
            lua_pushstring(L,"Cannot initialize likwid perfmon");
            lua_error(L);
            return 0;
        }
        perfmon_isInitialized = 1;
        timer_isInitialized = 1;
        lua_pushinteger(L,ret);
    }
    return 1;
}


static int lua_likwid_addEventSet(lua_State* L)
{
    int groupId, n;
    char eventString[1000];
    const char* tmpString;
    n = lua_gettop(L);
    tmpString = luaL_checkstring(L, n);
    luaL_argcheck(L, strlen(tmpString) > 0, n, "Event string must be larger than 0");
    
    groupId = perfmon_addEventSet((char*)tmpString);
    lua_pushnumber(L, groupId+1);
    return 1;
}

static int lua_likwid_setupCounters(lua_State* L)
{
    int ret;
    int groupId = lua_tonumber(L,1);
    ret = perfmon_setupCounters(groupId-1);
    lua_pushnumber(L,ret);
    return 1;
}


static int lua_likwid_startCounters(lua_State* L)
{
    int ret;
    ret = perfmon_startCounters();
    lua_pushnumber(L,ret);
    return 1;
}

static int lua_likwid_stopCounters(lua_State* L)
{
    int ret;
    ret = perfmon_stopCounters();
    lua_pushnumber(L,ret);
    return 1;
}

static int lua_likwid_readCounters(lua_State* L)
{
    int ret;
    ret = perfmon_readCounters();
    lua_pushnumber(L,ret);
    return 1;
}

static int lua_likwid_finalize(lua_State* L)
{
    perfmon_finalize();
    return 0;
}

static int lua_likwid_getResult(lua_State* L)
{
    int groupId, eventId, threadId;
    uint64_t result = 0;
    groupId = lua_tonumber(L,1);
    eventId = lua_tonumber(L,2);
    threadId = lua_tonumber(L,3);
    result = perfmon_getResult(groupId-1, eventId, threadId);
    lua_pushnumber(L,result);
    return 1;
}

static int lua_likwid_getNumberOfGroups(lua_State* L)
{
    int number;
    number = perfmon_getNumberOfGroups();
    lua_pushnumber(L,number);
    return 1;
}

static int lua_likwid_getIdOfActiveGroup(lua_State* L)
{
    int number;
    number = perfmon_getIdOfActiveGroup();
    lua_pushnumber(L,number);
    return 1;
}

static int lua_likwid_getRuntimeOfGroup(lua_State* L)
{
    int number, groupId;
    groupId = lua_tonumber(L,1);
    number = perfmon_getTimeOfGroup(groupId-1);
    lua_pushnumber(L, number);
    return 1;
}

static int lua_likwid_getNumberOfEvents(lua_State* L)
{
    int number, groupId;
    groupId = lua_tonumber(L,1);
    number = perfmon_getNumberOfEvents(groupId);
    lua_pushnumber(L,number);
    return 1;
}

static int lua_likwid_getNumberOfThreads(lua_State* L)
{
    int number;
    number = perfmon_getNumberOfThreads();
    lua_pushnumber(L,number);
    return 1;
}


static int lua_likwid_getCpuInfo(lua_State* L)
{
    CpuInfo_t cpuinfo;
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
    }
    cpuinfo = get_cpuInfo();
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
    lua_pushunsigned(L,cpuinfo->featureFlags);
    lua_settable(L,-3);
    lua_pushstring(L,"perf_version");
    lua_pushunsigned(L, cpuinfo->perf_version);
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

static int lua_likwid_getCpuTopology(lua_State* L)
{
    int i;
    CpuTopology_t cputopo;
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
    }
    if (numa_isInitialized == 0)
    {
        numa_init();
        numa_isInitialized = 1;
    }
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
    }
    cputopo = get_cpuTopology();

    lua_newtable(L);
    
    lua_pushstring(L,"numHWThreads");
    lua_pushunsigned(L,cputopo->numHWThreads);
    lua_settable(L,-3);
    
    lua_pushstring(L,"numSockets");
    lua_pushunsigned(L,cputopo->numSockets);
    lua_settable(L,-3);
    
    lua_pushstring(L,"numCoresPerSocket");
    lua_pushunsigned(L,cputopo->numCoresPerSocket);
    lua_settable(L,-3);
    
    lua_pushstring(L,"numThreadsPerCore");
    lua_pushunsigned(L,cputopo->numThreadsPerCore);
    lua_settable(L,-3);
    
    lua_pushstring(L,"numCacheLevels");
    lua_pushinteger(L,cputopo->numCacheLevels);
    lua_settable(L,-3);
    
    lua_pushstring(L,"threadPool");
    lua_newtable(L);
    for(i=0;i<cputopo->numHWThreads;i++)
    {
        lua_pushnumber(L,i+1);
        lua_newtable(L);
        lua_pushstring(L,"threadId");
        lua_pushunsigned(L,cputopo->threadPool[i].threadId);
        lua_settable(L,-3);
        lua_pushstring(L,"coreId");
        lua_pushunsigned(L,cputopo->threadPool[i].coreId);
        lua_settable(L,-3);
        lua_pushstring(L,"packageId");
        lua_pushunsigned(L,cputopo->threadPool[i].packageId);
        lua_settable(L,-3);
        lua_pushstring(L,"apicId");
        lua_pushunsigned(L,cputopo->threadPool[i].apicId);
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
        lua_pushunsigned(L,cputopo->cacheLevels[i].level);
        lua_settable(L,-3);
        
        lua_pushstring(L,"associativity");
        lua_pushunsigned(L,cputopo->cacheLevels[i].associativity);
        lua_settable(L,-3);
        
        lua_pushstring(L,"sets");
        lua_pushunsigned(L,cputopo->cacheLevels[i].sets);
        lua_settable(L,-3);
        
        lua_pushstring(L,"lineSize");
        lua_pushunsigned(L,cputopo->cacheLevels[i].lineSize);
        lua_settable(L,-3);
        
        lua_pushstring(L,"size");
        lua_pushunsigned(L,cputopo->cacheLevels[i].size);
        lua_settable(L,-3);
        
        lua_pushstring(L,"threads");
        lua_pushunsigned(L,cputopo->cacheLevels[i].threads);
        lua_settable(L,-3);
        
        lua_pushstring(L,"inclusive");
        lua_pushunsigned(L,cputopo->cacheLevels[i].inclusive);
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
        lua_pushunsigned(L,socketNode->id);
        lua_newtable(L);
        coreCount = 0;
        coreNode = tree_getChildNode(socketNode);

        while (coreNode != NULL)
        {
            lua_pushunsigned(L,coreNode->id);
            lua_newtable(L);
            threadNode = tree_getChildNode(coreNode);
            threadCount = 0;
            while (threadNode != NULL)
            {
                lua_pushunsigned(L,threadCount);
                lua_pushunsigned(L,threadNode->id);
                lua_settable(L,-3);
                threadNode = tree_getNextNode(threadNode);
                threadCount++;
            }
            coreNode = tree_getNextNode(coreNode);
            coreCount++;
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
        socketNode = tree_getNextNode(socketNode);
        socketCount++;
    }
    lua_settable(L,-3);
    

    return 1;
}

static int lua_likwid_putTopology(lua_State* L)
{
    if (topology_isInitialized == 0)
    {
        return 0;
    }
    topology_finalize();
    topology_isInitialized = 0;
    return 0;
}


static int lua_likwid_getEventsAndCounters(lua_State* L)
{
    int i;
    perfmon_init_maps();
    lua_newtable(L);
    lua_pushstring(L,"Counters");
    lua_newtable(L);
    for(i=1;i<=perfmon_numCounters;i++)
    {
        lua_pushunsigned(L,i);
        lua_pushstring(L,counter_map[i-1].key);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    lua_pushstring(L,"Events");
    lua_newtable(L);
    for(i=1;i<=perfmon_numArchEvents;i++)
    {
        lua_pushunsigned(L,i);
        lua_newtable(L);
        lua_pushstring(L,"Name");
        lua_pushstring(L,eventHash[i-1].name);
        lua_settable(L,-3);
        lua_pushstring(L,"ID");
        lua_pushunsigned(L,eventHash[i-1].eventId);
        lua_settable(L,-3);
        lua_pushstring(L,"UMask");
        lua_pushunsigned(L,eventHash[i-1].umask);
        lua_settable(L,-3);
        lua_pushstring(L,"Limit");
        lua_pushstring(L,eventHash[i-1].limit);
        lua_settable(L,-3);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    return 1;
}

static int lua_likwid_getNumaInfo(lua_State* L)
{
    uint32_t i,j;
    NumaTopology_t numa;
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
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
    }
    numa = get_numaTopology();
    lua_newtable(L);
    lua_pushstring(L,"numberOfNodes");
    lua_pushunsigned(L,numa->numberOfNodes);
    lua_settable(L,-3);
    
    lua_pushstring(L,"nodes");
    lua_newtable(L);
    
    for(i=0;i<numa->numberOfNodes;i++)
    {
        lua_pushinteger(L, i+1);
        lua_newtable(L);
        
        lua_pushstring(L,"id");
        lua_pushunsigned(L,numa->nodes[i].id);
        lua_settable(L,-3);
        lua_pushstring(L,"totalMemory");
        lua_pushunsigned(L,numa->nodes[i].totalMemory);
        lua_settable(L,-3);
        lua_pushstring(L,"freeMemory");
        lua_pushunsigned(L,numa->nodes[i].freeMemory);
        lua_settable(L,-3);
        lua_pushstring(L,"numberOfProcessors");
        lua_pushunsigned(L,numa->nodes[i].numberOfProcessors);
        lua_settable(L,-3);
        lua_pushstring(L,"numberOfDistances");
        lua_pushunsigned(L,numa->nodes[i].numberOfDistances);
        lua_settable(L,-3);
        
        lua_pushstring(L,"processors");
        lua_newtable(L);
        for(j=0;j<numa->nodes[i].numberOfProcessors;j++)
        {
            lua_pushunsigned(L,j+1);
            lua_pushunsigned(L,numa->nodes[i].processors[j]);
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
        
        /*lua_pushstring(L,"processorsCompact");
        lua_newtable(L);
        for(j=0;j<numa->nodes[i].numberOfProcessors;j++)
        {
            lua_pushunsigned(L,j);
            lua_pushunsigned(L,numa->nodes[i].processorsCompact[j]);
            lua_settable(L,-3);
        }
        lua_settable(L,-3);*/
        
        lua_pushstring(L,"distances");
        lua_newtable(L);
        for(j=0;j<numa->nodes[i].numberOfDistances;j++)
        {
            lua_pushinteger(L,j+1);
            lua_newtable(L);
            lua_pushinteger(L,j);
            lua_pushunsigned(L,numa->nodes[i].distances[j]);
            lua_settable(L,-3);
            lua_settable(L,-3);
            /*lua_pushstring(L,"src");
            lua_pushunsigned(L,i);
            lua_settable(L,-3);
            lua_pushstring(L,"dest");
            lua_pushunsigned(L,j);
            lua_settable(L,-3);
            lua_pushstring(L,"distance");
            lua_pushunsigned(L,numa->nodes[i].distances[j]);
            lua_settable(L,-3);*/
        }
        lua_settable(L,-3);
        
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    return 1;
}

static int lua_likwid_putNumaInfo(lua_State* L)
{
    numa_finalize();
    numa_isInitialized = 0;
    return 0;
}

static int lua_likwid_setMemInterleaved(lua_State* L)
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
        cpus[ret-1] = lua_tounsigned(L,-1);
        lua_pop(L,1);
    }
    numa_setInterleaved(cpus, nrThreads);
    return 0;
}

static int lua_likwid_getAffinityInfo(lua_State* L)
{
    int i,j;
    AffinityDomains_t affinity;
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
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
    }

    affinity = get_affinityDomains();
    if (!affinity)
    {
        return 0;
    }
    lua_newtable(L);
    lua_pushstring(L,"numberOfAffinityDomains");
    lua_pushunsigned(L,affinity->numberOfAffinityDomains);
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfSocketDomains");
    lua_pushunsigned(L,affinity->numberOfSocketDomains);
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfNumaDomains");
    lua_pushunsigned(L,affinity->numberOfNumaDomains);
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfProcessorsPerSocket");
    lua_pushunsigned(L,affinity->numberOfProcessorsPerSocket);
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfCacheDomains");
    lua_pushunsigned(L,affinity->numberOfCacheDomains);
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfCoresPerCache");
    lua_pushunsigned(L,affinity->numberOfCoresPerCache);
    lua_settable(L,-3);
    lua_pushstring(L,"numberOfProcessorsPerCache");
    lua_pushunsigned(L,affinity->numberOfProcessorsPerCache);
    lua_settable(L,-3);
    lua_pushstring(L,"domains");
    lua_newtable(L);
    for(i=0;i<affinity->numberOfAffinityDomains;i++)
    {
        lua_pushstring(L,bdata(affinity->domains[i].tag));
        lua_newtable(L);
        
        lua_pushstring(L,"numberOfProcessors");
        lua_pushunsigned(L,affinity->domains[i].numberOfProcessors);
        lua_settable(L,-3);
        lua_pushstring(L,"numberOfCores");
        lua_pushunsigned(L,affinity->domains[i].numberOfCores);
        lua_settable(L,-3);
        lua_pushstring(L,"processorList");
        lua_newtable(L);
        for(j=0;j<affinity->domains[i].numberOfProcessors;j++)
        {
            lua_pushunsigned(L,j+1);
            lua_pushunsigned(L,affinity->domains[i].processorList[j]);
            lua_settable(L,-3);
        }
        lua_settable(L,-3);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    return 1;
}

static int lua_likwid_putAffinityInfo(lua_State* L)
{
    affinity_finalize();
    affinity_isInitialized = 0;
    return 0;
}

static int lua_likwid_printAffinityDomains(lua_State* L)
{
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
    }
    affinity_printDomains();
    return 0;
}


static int lua_likwid_getPowerInfo(lua_State* L)
{
    PowerInfo_t power;
    int i;
    if (topology_isInitialized == 0)
    {
        topology_init();
        topology_isInitialized = 1;
    }
    if (power_isInitialized == 0)
    {
        power_init(0);
        power_isInitialized = 1;
    }
    power = get_powerInfo();
    

    lua_newtable(L);
    lua_pushstring(L,"baseFrequency");
    lua_pushnumber(L,power->baseFrequency);
    lua_settable(L,-3);
    lua_pushstring(L,"minFrequency");
    lua_pushnumber(L,power->minFrequency);
    lua_settable(L,-3);
    lua_pushstring(L,"powerUnit");
    lua_pushnumber(L,power->powerUnit);
    lua_settable(L,-3);
    lua_pushstring(L,"energyUnit");
    lua_pushnumber(L,power->energyUnit);
    lua_settable(L,-3);
    lua_pushstring(L,"timeUnit");
    lua_pushnumber(L,power->timeUnit);
    lua_settable(L,-3);
    lua_pushstring(L,"tdp");
    lua_pushnumber(L,power->tdp);
    lua_settable(L,-3);
    lua_pushstring(L,"minPower");
    lua_pushnumber(L,power->minPower);
    lua_settable(L,-3);
    lua_pushstring(L,"maxPower");
    lua_pushnumber(L,power->maxPower);
    lua_settable(L,-3);
    lua_pushstring(L,"maxTimeWindow");
    lua_pushnumber(L,power->maxTimeWindow);
    lua_settable(L,-3);
    
    lua_pushstring(L,"turbo");
    lua_newtable(L);
    lua_pushstring(L,"numSteps");
    lua_pushunsigned(L,power->turbo.numSteps);
    lua_settable(L,-3);
    lua_pushstring(L,"steps");
    lua_newtable(L);
    for(i=0;i<power->turbo.numSteps;i++)
    {
        lua_pushunsigned(L,i+1);
        lua_pushnumber(L,power->turbo.steps[i]);
        lua_settable(L,-3);
    }
    lua_settable(L,-3);
    lua_settable(L,-3);
    
    return 1;
}

static int lua_likwid_startPower(lua_State* L)
{
    PowerData pwrdata;
    int cpuId = lua_tonumber(L,1);
    luaL_argcheck(L, cpuId >= 0, 1, "CPU ID must be greater than 0");
    PowerType type = (PowerType) lua_tounsigned(L,2);
    luaL_argcheck(L, type >= PKG && type <= DRAM, 2, "Type not valid");
    power_start(&pwrdata, cpuId, type);
    lua_pushnumber(L,pwrdata.before);
    return 1;
}

static int lua_likwid_stopPower(lua_State* L)
{
    PowerData pwrdata;
    int cpuId = lua_tonumber(L,1);
    luaL_argcheck(L, cpuId >= 0, 1, "CPU ID must be greater than 0");
    PowerType type = (PowerType) lua_tounsigned(L,2);
    luaL_argcheck(L, type >= PKG && type <= DRAM, 2, "Type not valid");
    power_stop(&pwrdata, cpuId, type);
    lua_pushnumber(L,pwrdata.after);
    return 1;
}


static int lua_likwid_getCpuClock(lua_State* L)
{
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    lua_pushnumber(L,timer_getCpuClock());
    return 1;
}

static int isleep(lua_State* L)
{
    long interval = lua_tounsigned(L,-1);
    sleep(interval);
    return 0;
}

static int iusleep(lua_State* L)
{
    long interval = lua_tounsigned(L,-1);
    usleep(interval);
    return 0;
}

static int lua_likwid_startClock(lua_State* L)
{
    TimerData timer;
    uint64_t value;
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    timer_start(&timer);
    value = ((uint64_t)timer.start.int32.hi << 32) | (uint64_t)timer.start.int32.lo;
    lua_pushunsigned(L,value);
    return 1;
}

static int lua_likwid_stopClock(lua_State* L)
{
    TimerData timer;
    uint64_t value;
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    timer_stop(&timer);
    value = ((uint64_t)timer.start.int32.hi << 32) | (uint64_t)timer.start.int32.lo;
    lua_pushunsigned(L,value);
    return 1;
}

static int lua_likwid_getClockCycles(lua_State* L)
{
    TimerData timer;
    timer.start.int64 = (uint64_t)lua_tounsigned(L,-1);
    timer.stop.int64 = (uint64_t)lua_tounsigned(L,-2);
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    
    lua_pushunsigned(L, timer_printCycles(&timer));
    return 1;
}

static int lua_likwid_getClock(lua_State* L)
{
    TimerData timer;
    double runtime;
    uint64_t baseline;
    if (timer_isInitialized == 0)
    {
        timer_init();
        timer_isInitialized = 1;
    }
    baseline = timer_getBaseline();
    timer.start.int64 = (uint64_t)lua_tounsigned(L,-1);
    timer.stop.int64 = (uint64_t)lua_tounsigned(L,-2);
    runtime = timer_print(&timer);
    lua_pushnumber(L, runtime);
    return 1;
}

static int lua_likwid_initTemp(lua_State* L)
{
    int cpuid = lua_tounsigned(L,-1);
    thermal_init(cpuid);
    return 0;
}

static int lua_likwid_readTemp(lua_State* L)
{
    int cpuid = lua_tounsigned(L,-1);
    uint32_t data;
    
    if (thermal_read(cpuid, &data)) {
        lua_pushstring(L,"Cannot read thermal data");
        lua_error(L);
    }
    lua_pushnumber(L, data);
    return 1;
}

static int lua_likwid_startDaemon(lua_State* L)
{
    uint64_t switch_interval = (uint64_t)luaL_checknumber(L,-1);
    uint64_t duration = (uint64_t)lua_tonumber(L,-2);
    daemon_start(duration, switch_interval);
    return 0;
}

static int lua_likwid_stopDaemon(lua_State* L)
{
    int signal = lua_tointeger(L,-1);
    daemon_stop(signal);
    return 0;
}


static int lua_likwid_memSweep(lua_State* L)
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
        cpus[ret-1] = lua_tounsigned(L,-1);
        lua_pop(L,1);
    }
    memsweep_threadGroup(cpus, nrThreads);
    return 0;
}

static int lua_likwid_memSweepDomain(lua_State* L)
{
    int ret;
    int domain = luaL_checknumber(L,1);
    luaL_argcheck(L, domain >= 0, 1, "Domain ID must be greater or equal 0");
    memsweep_domain(domain);
    return 0;
}

static int lua_likwid_pinProcess(lua_State* L)
{
    int ret;
    int cpuID = luaL_checknumber(L,-2);
    int silent = luaL_checknumber(L,-1);
    luaL_argcheck(L, cpuID >= 0, 1, "CPU ID must be greater or equal 0");
    if (affinity_isInitialized == 0)
    {
        affinity_init();
        affinity_isInitialized = 1;
    }
    affinity_pinThread(cpuID);
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

static int lua_likwid_setenv(lua_State* L)
{
    const char* element = (const char*)luaL_checkstring(L, -2);
    const char* value = (const char*)luaL_checkstring(L, -1);
    setenv(element, value, 1);
    return 0;
}

static int lua_likwid_getpid(lua_State* L)
{
    lua_pushunsigned(L,getpid());
    return 1;
}

static int lua_likwid_fork(lua_State* L)
{
    int ret = fork();
    lua_pushnumber(L,ret);
    return 1;
}

static int lua_likwid_wait(lua_State* L)
{
    int status = 0;
    pid_t pid = luaL_checknumber(L,-1);
    pid = waitpid(pid, &status, 0);
    lua_pushnumber(L,pid);
    return 1;
}

int luaopen_liblikwid(lua_State* L){
    // Configuration functions
    lua_register(L, "likwid_getConfiguration", lua_likwid_getConfiguration);
    lua_register(L, "likwid_putConfiguration", lua_likwid_putConfiguration);
    // Perfmon functions
    lua_register(L, "accessClient_setaccessmode",lua_accessClient_setaccessmode);
    lua_register(L, "likwid_setAccessClientMode",lua_accessClient_setaccessmode);
    lua_register(L, "likwid_init",lua_likwid_init);
    lua_register(L, "likwid_addEventSet", lua_likwid_addEventSet);
    lua_register(L, "likwid_setupCounters",lua_likwid_setupCounters);
    lua_register(L, "likwid_startCounters",lua_likwid_startCounters);
    lua_register(L, "likwid_stopCounters",lua_likwid_stopCounters);
    lua_register(L, "likwid_readCounters",lua_likwid_readCounters);
    lua_register(L, "likwid_finalize",lua_likwid_finalize);
    lua_register(L, "likwid_getEventsAndCounters", lua_likwid_getEventsAndCounters);
    // Perfmon results functions
    lua_register(L, "likwid_getResult",lua_likwid_getResult);
    lua_register(L, "likwid_getNumberOfGroups",lua_likwid_getNumberOfGroups);
    lua_register(L, "likwid_getRuntimeOfGroup", lua_likwid_getRuntimeOfGroup);
    lua_register(L, "likwid_getIdOfActiveGroup",lua_likwid_getIdOfActiveGroup);
    lua_register(L, "likwid_getNumberOfEvents",lua_likwid_getNumberOfEvents);
    lua_register(L, "likwid_getNumberOfThreads",lua_likwid_getNumberOfThreads);
    // Topology functions
    lua_register(L, "likwid_getCpuInfo",lua_likwid_getCpuInfo);
    lua_register(L, "likwid_getCpuTopology",lua_likwid_getCpuTopology);
    lua_register(L, "likwid_putTopology",lua_likwid_putTopology);
    lua_register(L, "likwid_getNumaInfo",lua_likwid_getNumaInfo);
    lua_register(L, "likwid_putNumaInfo",lua_likwid_putNumaInfo);
    lua_register(L, "likwid_setMemInterleaved", lua_likwid_setMemInterleaved);
    lua_register(L, "likwid_getAffinityInfo",lua_likwid_getAffinityInfo);
    lua_register(L, "likwid_putAffinityInfo",lua_likwid_putAffinityInfo);
    lua_register(L, "likwid_printAffinityDomains",lua_likwid_printAffinityDomains);
    lua_register(L, "likwid_getPowerInfo",lua_likwid_getPowerInfo);
    // Timer functions
    lua_register(L, "likwid_getCpuClock",lua_likwid_getCpuClock);
    lua_register(L, "likwid_startClock",lua_likwid_startClock);
    lua_register(L, "likwid_stopClock",lua_likwid_stopClock);
    lua_register(L, "likwid_getClockCycles",lua_likwid_getClockCycles);
    lua_register(L, "likwid_getClock",lua_likwid_getClock);
    lua_register(L, "sleep",isleep);
    lua_register(L, "usleep",iusleep);
    // Daemon functions
    lua_register(L, "likwid_startDaemon", lua_likwid_startDaemon);
    lua_register(L, "likwid_stopDaemon", lua_likwid_stopDaemon);
    // Power functions
    lua_register(L, "likwid_startPower",lua_likwid_startPower);
    lua_register(L, "likwid_stopPower",lua_likwid_stopPower);
    // Temperature functions
    lua_register(L, "likwid_initTemp",lua_likwid_initTemp);
    lua_register(L, "likwid_readTemp",lua_likwid_readTemp);
    // MemSweep functions
    lua_register(L, "likwid_memSweep", lua_likwid_memSweep);
    lua_register(L, "likwid_memSweepDomain", lua_likwid_memSweepDomain);
    // Pinning functions
    lua_register(L, "likwid_pinProcess", lua_likwid_pinProcess);
    // Helper functions
    lua_register(L, "likwid_setenv", lua_likwid_setenv);
    lua_register(L, "likwid_getpid", lua_likwid_getpid);
    lua_register(L, "likwid_fork", lua_likwid_fork);
    lua_register(L, "likwid_wait", lua_likwid_wait);
    return 0;
}
