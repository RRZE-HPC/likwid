#!<PREFIX>/bin/likwid-lua

--[[
 * =======================================================================================
 *
 *      Filename:  likwid-pin.lua
 *
 *      Description:  An application to pin a program including threads
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

local function version()
    print(string.format("likwid-pin.lua --  Version %d.%d",likwid.version,likwid.release))
end

local function examples()
    print("Examples:")
    print("There are three possibilities to provide a thread to processor list:")
    print("1. Thread list with physical or logical thread numberings and physical cores first.")
    print("Example usage thread list: likwid-pin.lua -c N:0,4-6 ./myApp")
    print("You can pin with the following numberings:")
    print("\t1. Physical numbering of OS.")
    print("\t2. Logical numbering inside node. e.g. -c N:0-3")
    print("\t3. Logical numbering inside socket. e.g. -c S0:0-3")
    print("\t4. Logical numbering inside last level cache group. e.g. -c C0:0-3")
    print("\t5. Logical numbering inside NUMA domain. e.g. -c M0:0-3")
    print("\tYou can also mix domains separated by  @, e.g. -c S0:0-3@S1:0-3")
    print("2. Expressions based thread list generation with compact processor numbering.")
    print("Example usage expression: likwid-pin.lua -c E:N:8 ./myApp")
    print("This will generate a compact list of thread to processor mapping for the node domain")
    print("with eight threads.")
    print("The following syntax variants are available:")
    print("\t1. -c E:<thread domain>:<number of threads>")
    print("\t2. -c E:<thread domain>:<number of threads>:<chunk size>:<stride>")
    print("\tFor two SMT threads per core on a SMT 4 machine use e.g. -c E:N:122:2:4")
    print("3. Scatter policy among thread domain type.")
    print("Example usage scatter: likwid-pin.lua -c M:scatter ./myApp")
    print("This will generate a thread to processor mapping scattered among all memory domains")
    print("with physical cores first. If you ommit the -c option likwid will use all processors")
    print("available on the node with physical cores first. likwid-pin will also set ")
    print("OMP_NUM_THREADS with as many threads as specified in your pin expression if")
    print("OMP_NUM_THREADS is not present in your environment.")
end

local function usage()
    version()
    print("An application to pin a program including threads.\n")
    print("Options:")
    print("-h, --help\t\t Help message")
    print("-v, --version\t\t Version information")
    print("-V, --verbose <level>\t Verbose output, 0 (only errors), 1 (info), 2 (details), 3 (developer)")
    print("-i\t\t Set numa interleave policy with all involved numa nodes")
    print("-S, --sweep\t\t Sweep memory in involved NUMA nodes")
    print("-c <list>\t Comma separated processor IDs or expression")
    print("-s, --skip <hex>\t Bitmask with threads to skip")
    print("-p\t\t Print available domains with mapping on physical IDs")
    print("\t\t If used together with -p option outputs a physical processor IDs.")
    print("-d <string>\t Delimiter used for using -p to output physical processor list, default is comma.")
    print("-q, --quiet\t\t Silent without output")
    
    print("\n")
    examples()
end

delimiter = ','
quiet = 0
sweep_sockets = false
interleaved_policy = false
print_domains = false
cpu_list = {}
skip_mask = "0x0"
affinity = nil
num_threads = 0

config = likwid.getConfiguration()
cputopo = likwid.getCpuTopology()
affinity = likwid.getAffinityInfo()

if (#arg == 0) then
    usage()
    os.exit(0)
end

for opt,arg in likwid.getopt(arg, {"c:", "d:", "h", "i", "p", "q", "s:", "S", "t:", "v", "V:", "verbose:", "help", "version", "skip","sweep", "quiet"}) do
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
        likwid.setVerbosity(verbose)
    elseif (opt == "c") then
        if (affinity ~= nil) then
            num_threads,cpu_list = likwid.cpustr_to_cpulist(arg)
        else
            num_threads,cpu_list = likwid.cpustr_to_cpulist_physical(arg)
        end
        if (num_threads == 0) then
            print("Failed to parse cpulist " .. arg)
            os.exit(1)
        end
    elseif (opt == "d") then
        delimiter = arg
    elseif opt == "S" or opt == "sweep" then
        if (affinity == nil) then
            print("Option -S is not supported for unknown processor!")
            os.exit(1)
        end
        sweep_sockets = true
    elseif (opt == "i") then
        interleaved_policy = true
    elseif (opt == "p") then
        print_domains = true
    elseif opt == "s" or opt == "skip" then
        local s,e = arg:find("0x")
        if s == nil then
            print("Skip mask must be given in hex, hence start with 0x")
            os.exit(1)
        end
        skip_mask = arg
    elseif opt == "q" or opt == "quiet" then
        likwid.setenv("LIKWID_SILENT","true")
        quiet = 1
    else
        print("Unknown option -" .. opt .. "\n")
        usage()
        os.exit(0)
    end
end

if print_domains and num_threads > 0 then
    outstr = ""
    for i, cpu in pairs(cpu_list) do
        outstr = outstr .. delimiter .. cpu
    end
    print(outstr:sub(2,outstr:len()))
    os.exit(0)
elseif print_domains then
    for k,v in pairs(affinity["domains"]) do
        print(string.format("Domain %d Tag %s:",k, v["tag"]))
        print("\t" .. table.concat(v["processorList"], ","))
        print()
    end
    os.exit(0)
end

if num_threads == 0 then
    num_threads, cpu_list = likwid.cpustr_to_cpulist("N:0-"..cputopo["numHWThreads"]-1)
end

if interleaved_policy then
    print("Set mem_policy to interleaved")
    likwid.setMemInterleaved(num_threads, cpu_list)
end

if sweep_sockets then
    print("Sweeping memory")
    likwid.memSweep(num_threads, cpu_list)
end

local omp_threads = os.getenv("OMP_NUM_THREADS")
if omp_threads == nil then
    likwid.setenv("OMP_NUM_THREADS",tostring(num_threads))
end


if num_threads > 1 then
    local preload = os.getenv("LD_PRELOAD")
    local pinString = tostring(cpu_list[2])
    for i=3,likwid.tablelength(cpu_list) do
        pinString = pinString .. "," .. cpu_list[i]
    end
    pinString = pinString .. "," .. cpu_list[1]
    skipString = skip_mask

    likwid.setenv("KMP_AFFINITY","disabled")
    likwid.setenv("LIKWID_PIN", pinString)
    likwid.setenv("LIKWID_SKIP",skipString)

    if preload == nil then
        likwid.setenv("LD_PRELOAD",likwid.pinlibpath)
    else
        likwid.setenv("LD_PRELOAD",likwid.pinlibpath .. ":" .. preload)
    end
end

likwid.pinProcess(cpu_list[1], quiet)
local exec = table.concat(arg," ",1, likwid.tablelength(arg)-2)
local err
err = os.execute(exec)
if (err == false) then
    print("Failed to execute command: ".. exec)
    os.exit(1)
end

likwid.putAffinityInfo()
likwid.putTopology()
likwid.putConfiguration()
os.exit(0)
