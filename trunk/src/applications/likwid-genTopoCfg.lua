#!/home/rrze/unrz/unrz139/Work/likwid/trunk/ext/lua/lua

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
    print(string.format("likwid-genCfg --  Version %d.%d",VERSION,RELEASE))
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

local cpuinfo = likwid_getCpuInfo()
local cputopo = likwid_getCpuTopology()
local numainfo = likwid_getNumaInfo()
local affinity = likwid_getAffinityInfo()
cpuinfo["clock"] = likwid_getCpuClock()

local threadPool_order = {"threadId", "coreId", "packageId", "apicId"}
local cacheLevels_order = {"type", "associativity", "sets", "lineSize", "size", "threads", "inclusive"}

local file = io.open("./likwid.topo", "w")

for field, value in pairs(cpuinfo) do
    file:write("cpuid_info " .. field .. " = " .. tostring(value).."\n")
end

for field, value in pairs(cputopo) do
    if (field ~= "threadPool" and field ~= "cacheLevels" and field ~= "topologyTree") then
        print("cpuid_topology." .. field .. " = " .. tostring(value))
        file:write("cpuid_topology " .. field .. " = " .. tostring(value).."\n")
    elseif (field == "threadPool") then
        --file:write("cpuid_topology threadPool count = "..tostring(likwid.tablelength(cputopo["threadPool"])).."\n")
        for id, tab in pairs(cputopo["threadPool"]) do
            str = "cpuid_topology threadPool "..tostring(id).." "
            for k,v in pairs(threadPool_order) do
                file:write(str..tostring(v).." = "..tostring(tab[v]).."\n")
            end
            
        end
    elseif (field == "cacheLevels") then
        for id, tab in pairs(cputopo["cacheLevels"]) do
            str = "cpuid_topology cacheLevels "..tostring(id).." "
            for k,v in pairs(cacheLevels_order) do
                file:write(str..tostring(v).." = "..tostring(tab[v]).."\n")
            end
            
        end
    end
end

file:write("numa_info numberOfNodes = "..tostring(numainfo["numberOfNodes"]).."\n")
for field, value in pairs(numainfo["nodes"]) do
    for id, tab in pairs(value) do
        if id ~= "processors" and id ~= "distances" then
            file:write("numa_info nodes "..tostring(field).." "..tostring(id).." = "..tostring(tab).."\n")
        elseif id == "processors" then
            for k,v in pairs(tab) do 
                str = str..","..tostring(v) 
                file:write("numa_info nodes "..tostring(field).." "..tostring(id).." "..tostring(k).." = "..tostring(v).."\n")
            end
        elseif id == "distances" then
            for k,v in pairs(tab) do
                for k1,v1 in pairs(v) do
                    file:write("numa_info nodes "..tostring(field).." "..tostring(id).." "..tostring(k1).." = "..tostring(v1).."\n")
                end
            end
        end
    end
end




file:close()
