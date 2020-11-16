#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-genTopoCfg.lua
 *
 *      Description:  A application to create a file of the underlying system configuration
 *                    that is used by likwid to avoid reading the systems architecture at
 *                    each start.
 *
 *      Version:   5.1
 *      Released:  16.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
]]
package.path = '<INSTALLED_PREFIX>/share/lua/?.lua;' .. package.path

local likwid = require("likwid")

print_stdout = print
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end

local filename = "<INSTALLED_PREFIX>/etc/likwid_topo.cfg"

function version()
    print_stdout(string.format("likwid-genTopoCfg -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

function usage()
    version()
    print_stdout("A tool to store the system's architecture to a config file for LIKWID.\n")
    print_stdout("Options:")
    print_stdout("-h, --help\t\t Help message")
    print_stdout("-v, --version\t\t Version information")
    print_stdout("-o, --output <file>\t Use <file> instead of default "..filename)
    print_stdout("\t\t\t Likwid searches at startup per default:")
    print_stdout("\t\t\t /etc/likwid_topo.cfg and <INSTALLED_PREFIX>/etc/likwid_topo.cfg")
    print_stdout("\t\t\t Another location can be configured in the configuration file /etc/likwid.cfg,")
    print_stdout("\t\t\t <INSTALLED_PREFIX>/etc/likwid.cfg or the path defined at the build process of Likwid.")
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
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end
local file = io.open(filename, "r")
if file ~= nil then
    print_stderr("File "..filename.." exists, please delete it first.")
    file:close()
    os.exit(1)
end
file = io.open(filename, "w")
if file == nil then
    print_stderr("Cannot open file "..filename.." for writing")
    os.exit(1)
end


local cpuinfo = likwid.getCpuInfo()
local cputopo = likwid.getCpuTopology()
local numainfo = likwid.getNumaInfo()
local affinity = likwid.getAffinityInfo()
if cpuinfo == nil or cputopo == nil or numainfo == nil or affinity == nil then
    print_stderr("Cannot initialize topology module of LIKWID")
    os.exit(1)
end
print_stdout(string.format("Writing new topology file %s", filename))
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

