#!<PREFIX>/bin/likwid-lua

--[[
 * =======================================================================================
 *
 *      Filename:  likwid-genTopoCfg.lua
 *
 *      Description:  A application to create a file of the underlying system configuration
 *                    that is used by likwid to avoid reading the systems architecture at
 *                    each start.
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
package.path = package.path .. ';<PREFIX>/share/lua/?.lua'

local likwid = require("likwid")

local filename = "/etc/likwid.cfg"

function version()
    print(string.format("likwid-genTopoCfg --  Version %d.%d",likwid.version,likwid.release))
end

function usage()
    version()
    print("A tool to store the system's architecture to a config file for LIKWID.\n")
    print("Options:")
    print("-h, --help\t Help message")
    print("-v, --version\t Version information")
    print("-o, --output <file>\t Use <file> instead of default "..filename)
end

for opt,arg in likwid.getopt(arg, {"h","v","help","version", "o:", "output:"}) do
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif opt == "o" or opt == "output" then
        filename = arg
    end
end

local file = io.open(filename, "w")
if file == nil then
    print("Cannot open file "..filename.." for writing")
    os.exit(1)
end


local cpuinfo = likwid.getCpuInfo()
local cputopo = likwid.getCpuTopology()
local numainfo = likwid.getNumaInfo()
local affinity = likwid.getAffinityInfo()
cpuinfo["clock"] = likwid.getCpuClock()

local threadPool_order = {"threadId", "coreId", "packageId", "apicId"}
local cacheLevels_order = {"type", "associativity", "sets", "lineSize", "size", "threads", "inclusive"}

for field, value in pairs(cpuinfo) do
    file:write("cpuid_info " .. field .. " = " .. tostring(value).."\n")
end

for field, value in pairs(cputopo) do
    if (field ~= "threadPool" and field ~= "cacheLevels" and field ~= "topologyTree") then
        if field ~= "activeHWThreads" then
            file:write("cpuid_topology " .. field .. " = " .. tostring(value).."\n")
        end
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
likwid.putAffinityInfo()
likwid.putNumaInfo()
likwid.putTopology()

