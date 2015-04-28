#!<PREFIX>/bin/likwid-lua

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
package.path = package.path .. ';<PREFIX>/share/lua/?.lua'

local likwid = require("likwid")
stdout_print = print

function version()
    print(string.format("likwid-topology --  Version %d.%d",likwid.version,likwid.release))
end

function usage()
    version()
    print("A tool to print the thread and cache topology on x86 CPUs.\n")
    print("Options:")
    print("-h, --help\t\t Help message")
    print("-v, --version\t\t Version information")
    print("-V, --verbose <level>\t Set verbosity")
    print("-c, --caches\t\t List cache information")
    print("-C, --clock\t\t Measure processor clock")
    print("-o, --output <file>\t Store output to file. (Optional: Apply text filter)")
    print("-g\t\t\t Graphical output")
end

print_caches = false
print_graphical = false
measure_clock = false
outfile = nil

for opt,arg in likwid.getopt(arg, {"h","v","c","C","g","o:","V:","help","version","verbose:","clock","caches","output:"}) do
    if (type(arg) == "string") then
        local s,e = arg:find("-");
        if s == 1 then
            print(string.format("Argmument %s to option -%s starts with invalid character -.", arg, opt))
            print("Did you forget an argument to an option?")
            os.exit(1)
        end
    end
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif opt == "V" or opt == "verbose" then
        likwid.setVerbosity(tonumber(arg))
    elseif opt == "c" or opt == "caches" then
        print_caches = true
    elseif opt == "C" or opt == "clock" then
        measure_clock = true
    elseif opt == "g" then
        print_graphical = true
    elseif opt == "o" or opt == "output" then
        outfile = arg
        io.output(arg:gsub(string.match(arg, ".-[^\\/]-%.?([^%.\\/]*)$"),"tmp"))
        print = function(...) for k,v in pairs({...}) do io.write(v .. "\n") end end
    end
end

local config = likwid.getConfiguration()
local cpuinfo = likwid.getCpuInfo()
local cputopo = likwid.getCpuTopology()
local numainfo = likwid.getNumaInfo()
local affinity = likwid.getAffinityInfo()

print(likwid.hline)
print(string.format("CPU name:\t%s",cpuinfo["osname"]))
print(string.format("CPU type:\t%s",cpuinfo["name"]))
print(string.format("CPU stepping:\t%s",cpuinfo["stepping"]))
if (measure_clock) then
    if cpuinfo["clock"] == 0 then
        print(string.format("CPU clock:\t%3.2f GHz", likwid.getCpuClock() * 1.E-09))
    else
        print(string.format("CPU clock:\t%3.2f GHz", cpuinfo["clock"] * 1.E-09))
    end
end

print(likwid.sline)
print("Hardware Thread Topology")
print(likwid.sline)
print(string.format("Sockets:\t\t%u",cputopo["numSockets"]))
print(string.format("Cores per socket:\t%u",cputopo["numCoresPerSocket"]))
print(string.format("Threads per core:\t%u",cputopo["numThreadsPerCore"]))
print(likwid.hline)
print("HWThread\tThread\t\tCore\t\tSocket")

for cntr=0,cputopo["numHWThreads"]-1 do
    print(string.format("%d\t\t%u\t\t%u\t\t%u",cntr,
                        cputopo["threadPool"][cntr]["threadId"],
                        cputopo["threadPool"][cntr]["coreId"],
                        cputopo["threadPool"][cntr]["packageId"]))
end
print(likwid.hline)

for socket=0,cputopo["numSockets"]-1 do
    str = string.format("Socket %d: (",cputopo["topologyTree"][socket]["ID"])
    for core=0,cputopo["numCoresPerSocket"]-1 do
        for thread=0, cputopo["numThreadsPerCore"]-1 do
            str = str .. " " .. tostring(cputopo["topologyTree"][socket]["Childs"][core]["Childs"][thread])
        end
    end
    print(str .. " )")
end

print(likwid.hline .. "\n")

print(likwid.sline)
print("Cache Topology")
print(likwid.sline)

for level=1,cputopo["numCacheLevels"] do
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
        end
        local threads = cputopo["cacheLevels"][level]["threads"]
        str = "Cache groups:\t( "
        for socket=0,cputopo["numSockets"]-1 do
            for core=0,cputopo["numCoresPerSocket"]-1 do
                for cpu=0,cputopo["numThreadsPerCore"]-1 do
                    if (threads ~= 0) then
                        str = str .. cputopo["topologyTree"][socket]["Childs"][core]["Childs"][cpu] .. " "
                        threads = threads - 1
                    else
                        str = str .. string.format(") ( %d ",cputopo["topologyTree"][socket]["Childs"][core]["Childs"][cpu])
                        threads = cputopo["cacheLevels"][level]["threads"]
                        threads = threads - 1
                    end
                end
            end
        end
        print(str .. ")")
        print(likwid.hline)
    end
end
print("\n")

print(likwid.sline)
print("NUMA Topology")
print(likwid.sline)

if (numainfo["numberOfNodes"] == 0) then
    print("NUMA is not supported on this node!")
else
    print(string.format("NUMA domains: %d",numainfo["numberOfNodes"]))
    print(likwid.hline)
    -- -2 because numberOfNodes is seen as one entry
    for node=1,numainfo["numberOfNodes"] do
        print(string.format("Domain %d:",numainfo["nodes"][node]["id"]))
        str = "Processors: "
        for cpu=1,numainfo["nodes"][node]["numberOfProcessors"] do
            str = str .. " " .. numainfo["nodes"][node]["processors"][cpu]
        end
        print(str)
        str = "Relative distance to nodes: "
        for cpu=1,numainfo["nodes"][node]["numberOfDistances"] do
            str = str .. " " .. numainfo["nodes"][node]["distances"][cpu][cpu-1]
        end
        print(str)
        print(string.format("Memory: %g MB free of total %g MB", 
                                tonumber(numainfo["nodes"][node]["freeMemory"]/1024.0),
                                tonumber(numainfo["nodes"][node]["totalMemory"]/1024.0)))
        print(likwid.hline)
    end
end

if (print_graphical) then
    print("\n")
    print(likwid.sline)
    print("Graphical Topology")
    print(likwid.sline)
    for socket=0,cputopo["numSockets"]-1 do
        print(string.format("Socket %d:",cputopo["topologyTree"][socket]["ID"]))
        container = {}
        for core=0,cputopo["numCoresPerSocket"]-1 do
            local tmpString = ""
            for thread=0,cputopo["numThreadsPerCore"]-1 do
                if thread == 0 then
                    tmpString = tmpString .. tostring(cputopo["topologyTree"][socket]["Childs"][core]["Childs"][thread])
                else
                    tmpString = tmpString .. " " .. tostring(cputopo["topologyTree"][socket]["Childs"][core]["Childs"][thread]).. " "
                end
            end
            likwid.addSimpleAsciiBox(container, 1, core+1, tmpString)
        end
        
        local columnCursor = 1
        local lineCursor = 2
        for cache=1,cputopo["numCacheLevels"] do
            if cputopo["cacheLevels"][cache]["type"] ~= "INSTRUCTIONCACHE" then
                local cachesAtCurLevel = 0
                local sharedCores = cputopo["cacheLevels"][cache]["threads"]/cputopo["numThreadsPerCore"]
                if sharedCores >= cputopo["numCoresPerSocket"] then
                    cachesAtCurLevel = 1
                else
                    cachesAtCurLevel = cputopo["numCoresPerSocket"]/sharedCores
                end
                columnCursor = 1
                for cachesAtLevel=1,cachesAtCurLevel do
                    local tmpString = ""
                    local cacheWidth = 0
                    if cputopo["cacheLevels"][cache]["size"] < 1048576 then
                        tmpString = string.format("%dkB", cputopo["cacheLevels"][cache]["size"]/1024)
                    else
                        tmpString = string.format("%dMB", cputopo["cacheLevels"][cache]["size"]/1048576)
                    end
                    if sharedCores > 1 then
                        if sharedCores > cputopo["numCoresPerSocket"] then
                            cacheWidth = sharedCores
                        else
                            cacheWidth = sharedCores - 1
                        end
                        likwid.addJoinedAsciiBox(container, lineCursor, columnCursor,columnCursor + cacheWidth, tmpString)
                        columnCursor = columnCursor + cacheWidth
                    else
                        likwid.addSimpleAsciiBox(container, lineCursor, columnCursor, tmpString)
                        columnCursor = columnCursor + 1
                    end
                end
                lineCursor = lineCursor + 1
            end
        end
        likwid.printAsciiBox(container);
    end
end

if outfile then
    local suffix = string.match(outfile, ".-[^\\/]-%.?([^%.\\/]*)$")
    local command = "<PREFIX>/share/likwid/filter/" .. suffix 
    if suffix ~= "txt" then
        command = command .." ".. outfile:gsub("."..suffix,".tmp",1) .. " topology"
        local f = io.popen(command)
        local o = f:read("*a")
        if o:len() > 0 then
            print(string.format("Failed to executed filter script %s.",command))
        end
    else
        os.rename(outfile:gsub("."..suffix,".tmp",1), outfile)
    end
end

likwid.putAffinityInfo()
likwid.putNumaInfo()
likwid.putTopology()
likwid.putConfiguration()
