#!/usr/bin/env lua

--[[
 * =======================================================================================
 *
 *      Filename:  likwid-topology.lua
 *
 *      Description:  A application to determine the thread and cache topology
 *                    on x86 processors.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Thomas Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Thomas Roehl
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
 * =======================================================================================]]
VERSION = 4
RELEASE = 0

require("liblikwid")
local likwid = require("likwid")

local HLINE = string.rep("-",80)
local SLINE = string.rep("*",80)


function version()
    print(string.format("likwid-topology --  Version %d.%d",VERSION,RELEASE))
end

function usage()
    version()
    print("A tool to print the thread and cache topology on x86 CPUs.\n")
    print("Options:")
    print("-h\t Help message")
    print("-v\t Version information")
    print("-c\t List cache information")
    print("-C\t Measure processor clock")
    print("-o\t Store output to file. (Optional: Apply text filter)")
    print("-g\t Graphical output")
end

print_caches = false
print_graphical = false
measure_clock = false


for opt,arg in likwid.getopt(arg, "hvcCgo:") do
    if (opt == "h") then
        usage()
        os.exit(0)
    elseif (opt == "v") then
        version()
        os.exit(0)
    elseif (opt == "c") then
        print_caches = true
    elseif (opt == "C") then
        measure_clock = true
    elseif (opt == "g") then
        print_graphical = true
    elseif (opt == "o") then
        io.output(arg)
    end
end

local cpuinfo = likwid_getCpuInfo()
local cputopo = likwid_getCpuTopology()
local numainfo = likwid_getNumaInfo()
local affinity = likwid_getAffinityInfo()

print(HLINE)
print(string.format("CPU type:\t%s",cpuinfo["name"]))

if (measure_clock) then
    print(string.format("CPU clock:\t%3.2f GHz", likwid_getCpuClock() * 1.E-09))
end

print(SLINE)
print("Hardware Thread Topology")
print(SLINE)
print(string.format("Sockets:\t\t%u",cputopo["numSockets"]))
print(string.format("Cores per socket:\t%u",cputopo["numCoresPerSocket"]))
print(string.format("Threads per core:\t%u",cputopo["numThreadsPerCore"]))
print(HLINE)
print("HWThread\tThread\t\tCore\t\tSocket")

for cntr=0,cputopo["numHWThreads"]-1 do
    print(string.format("%d\t\t%u\t\t%u\t\t%u",cntr,
                        cputopo["threadPool"][cntr]["threadId"],
                        cputopo["threadPool"][cntr]["coreId"],
                        cputopo["threadPool"][cntr]["packageId"]))
end
print(HLINE)

for socket=0,cputopo["numSockets"]-1 do
    str = string.format("Socket %d: (",socket)
    for core=0,#cputopo["topologyTree"][socket] do
        for cpu=0,#cputopo["topologyTree"][socket][core] do
            str = str .. " " .. cputopo["topologyTree"][socket][core][cpu]
        end
    end
    print(str .. " )")
end
print(HLINE .. "\n")

print(SLINE)
print("Cache Topology")
print(SLINE)

for level=0,cputopo["numCacheLevels"]-1 do
    if (cputopo["cacheLevels"][level]["type"] ~= "INSTRUCTIONCACHE") then
        print(string.format("Level:\t%d",cputopo["cacheLevels"][level]["level"]))
        if (cputopo["cacheLevels"][level]["size"] < 1048576) then
            print(string.format("Size:\t%d kB",cputopo["cacheLevels"][level]["size"]/1024))
        else
            print(string.format("Size:\t%d MB",cputopo["cacheLevels"][level]["size"]/1048576))
        end
        
        if (print_caches) then
            if (cputopo["cacheLevels"][level]["type"] == "DATACACHE") then
                print("Type:\tData cache")
            elseif (cputopo["cacheLevels"][level]["type"] == "UNIFIEDCACHE") then
                print("Type:\tUnified cache")
            end

            print(string.format("Associativity:\t%d",cputopo["cacheLevels"][level]["associativity"]))
            print(string.format("Number of sets:\t%d",cputopo["cacheLevels"][level]["sets"]))
            print(string.format("Cache line size:%d",cputopo["cacheLevels"][level]["lineSize"]))
            
            if (cputopo["cacheLevels"][level]["inclusive"] > 0) then
                print("Non Inclusive cache")
            else
                print("Inclusive cache")
            end
            print(string.format("Shared among %d threads",cputopo["cacheLevels"][level]["threads"]))
            local threads = cputopo["cacheLevels"][level]["threads"]
            str = "Cache groups:\t( "
            for socket=0,cputopo["numSockets"]-1 do
                for core=0,#cputopo["topologyTree"][socket] do
                    for cpu=0,#cputopo["topologyTree"][socket][core] do
                        if (threads ~= 0) then
                            str = str .. cputopo["topologyTree"][socket][core][cpu] .. " "
                            threads = threads - 1
                        else
                            str = str .. string.format(") ( %d ",cputopo["topologyTree"][socket][core][cpu])
                            threads = cputopo["cacheLevels"][level]["threads"]
                            threads = threads - 1
                        end
                    end
                end
                print(str .. ")")
            end
        end
        print(HLINE)
    end
end
print("\n")

print(SLINE)
print("NUMA Topology")
print(SLINE)

if (numainfo["numberOfNodes"] == 0) then
    print("NUMA is not supported on this node!")
else
    print(string.format("NUMA domains: %d",numainfo["numberOfNodes"]))
    print(HLINE)
    -- -2 because numberOfNodes is seen as one entry
    for node=0,numainfo["numberOfNodes"]-1 do
        print(string.format("Domain %d:",numainfo[node]["id"]))
        str = "Processors: "
        for cpu=0,numainfo[node]["numberOfProcessors"]-1 do
            str = str .. " " .. numainfo[node]["processors"][cpu]
        end
        print(str)
        
        str = "Relative distance to nodes: "
        for cpu=0,numainfo[node]["numberOfDistances"]-1 do
            str = str .. " " .. numainfo[node]["distances"][cpu]
        end
        print(str)
        print(string.format("Memory: %g MB free of total %g MB", 
                                numainfo[node]["freeMemory"]/1024,
                                numainfo[node]["totalMemory"]/1024))
        print(HLINE)
    end
end

if (print_graphical) then
    print("Graphical output currently not supported by likwid-topology written in Lua")
end



















