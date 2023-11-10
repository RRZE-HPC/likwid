#!<INSTALLED_BINPREFIX>/likwid-lua
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
 *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
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
]]
package.path = '<INSTALLED_PREFIX>/share/lua/?.lua;' .. package.path

local likwid = require("likwid")

print_stdout = print
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end

local function version()
    print_stdout(string.format("likwid-pin -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

local function examples()
    print_stdout("Examples:")
    print_stdout("There are three possibilities to provide a thread to processor list:")
    print_stdout("1. Thread list with physical thread IDs")
    print_stdout("Example: likwid-pin.lua -c 0,4-6 ./myApp")
    print_stdout("Pins the application to hardware threads 0,4,5 and 6")
    print_stdout("2. Thread list with logical thread numberings in physical cores first sorted list.")
    print_stdout("Example usage thread list: likwid-pin.lua -c N:0,4-6 ./myApp")
    print_stdout("You can pin with the following numberings:")
    print_stdout("\t2. Logical numbering inside node.\n\t   e.g. -c N:0,1,2,3 for the first 4 physical cores of the node")
    print_stdout("\t3. Logical numbering inside socket.\n\t   e.g. -c S0:0-1 for the first 2 physical cores of the socket")
    print_stdout("\t4. Logical numbering inside last level cache group.\n\t   e.g. -c C0:0-3  for the first 4 physical cores in the first LLC")
    print_stdout("\t5. Logical numbering inside NUMA domain.\n\t   e.g. -c M0:0-3 for the first 4 physical cores in the first NUMA domain")
    print_stdout("\tYou can also mix domains separated by  @,\n\te.g. -c S0:0-3@S1:0-3 for the 4 first physical cores on both sockets.")
    print_stdout("3. Expressions based thread list generation with compact processor numbering.")
    print_stdout("Example usage expression: likwid-pin.lua -c E:N:8 ./myApp")
    print_stdout("This will generate a compact list of thread to processor mapping for the node domain")
    print_stdout("with eight threads.")
    print_stdout("The following syntax variants are available:")
    print_stdout("\t1. -c E:<thread domain>:<number of threads>")
    print_stdout("\t2. -c E:<thread domain>:<number of threads>:<chunk size>:<stride>")
    print_stdout("\tFor two hardware threads per core on a SMT4 machine use e.g. -c E:N:122:2:4")
    print_stdout("4. Scatter policy among thread domain type.")
    print_stdout("Example usage scatter: likwid-pin.lua -c M:scatter ./myApp")
    print_stdout("This will generate a thread to processor mapping scattered among all memory domains")
    print_stdout("with physical hardware threads first.")
    print_stdout("")
    print_stdout("likwid-pin sets OMP_NUM_THREADS with as many threads as specified")
    print_stdout("in your pin expression if OMP_NUM_THREADS is not present in your environment.")
end

local function usage()
    version()
    print_stdout("An application to pin a program including threads.\n")
    print_stdout("Options:")
    print_stdout("-h, --help\t\t Help message")
    print_stdout("-v, --version\t\t Version information")
    print_stdout("-V, --verbose <level>\t Verbose output, 0 (only errors), 1 (info), 2 (details), 3 (developer)")
    print_stdout("-i\t\t\t Set numa interleave policy with all involved numa nodes")
    print_stdout("-m\t\t\t Set numa membind policy with all involved numa nodes")
    print_stdout("-S, --sweep\t\t Sweep memory and LLC of involved NUMA nodes")
    print_stdout("-c/-C <list>\t\t Comma separated processor IDs or expression")
    print_stdout("-s, --skip <hex>\t Bitmask with threads to skip")
    print_stdout("-p\t\t\t Print available domains with mapping on physical IDs")
    print_stdout("\t\t\t If used together with -c option outputs the list of physical processor IDs.")
    print_stdout("-d <string>\t\t Delimiter used for using -p to output physical processor list, default is comma.")
    print_stdout("-q, --quiet\t\t Silent without output")
    print_stdout("\n")
    examples()
end

local function close_and_exit(code)
    likwid.putTopology()
    likwid.putAffinityInfo()
    likwid.putConfiguration()
    likwid.unsetenv("LIKWID_NO_ACCESS")
    os.exit(code)
end

delimiter = ','
quiet = 0
sweep_sockets = false
interleaved_policy = false
membind_policy = false
print_domains = false
cpu_list = {}
skip_mask = nil
affinity = nil
num_threads = 0
cpustr = nil
verbose = 0


if (#arg == 0) then
    usage()
    os.exit(0)
end

for opt,arg in likwid.getopt(arg, {"c:", "C:", "d:", "h", "i", "m", "p", "q", "s:", "S", "t:", "v", "V:", "verbose:", "help", "version", "skip","sweep", "quiet"}) do
    if opt == "h" or opt == "help" then
        usage()
        close_and_exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        close_and_exit(0)
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
        likwid.setVerbosity(verbose)
    elseif (opt == "c") or (opt == "C") then
        cpustr = arg
    elseif (opt == "d") then
        delimiter = arg
    elseif opt == "S" or opt == "sweep" then
        sweep_sockets = true
    elseif (opt == "i") then
        interleaved_policy = true
    elseif (opt == "m") then
        membind_policy = true
    elseif (opt == "p") then
        print_domains = true
    elseif opt == "s" or opt == "skip" then
        if arg:match("0x[0-9A-Fa-f]") then
            skip_mask = arg
        else
            if arg:match("[0-9A-Fa-f]") then
                print_stderr("Given skip mask looks like hex, sanitizing arg to 0x"..arg)
                skip_mask = "0x"..arg
            else
                print_stderr("Skip mask must be given in hex")
                close_and_exit(1)
            end
        end
    elseif opt == "q" or opt == "quiet" then
        likwid.setenv("LIKWID_SILENT","true")
        quiet = 1
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        close_and_exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        close_and_exit(1)
    end
end
local execList = {}
for i=1, likwid.tablelength(arg)-2 do
    if string.find(arg[i], " ") then
        table.insert(execList, "\""..arg[i].."\"")
    else
        table.insert(execList, arg[i])
    end
end

likwid.setenv("LIKWID_NO_ACCESS", "1")
config = likwid.getConfiguration()
cputopo = likwid.getCpuTopology()
numainfo = likwid.getNumaInfo()
affinity = likwid.getAffinityInfo()

if cpustr ~= nil then
    if not (cpustr:match(",$") or cpustr:match("-$") or cpustr:match("^-") or cpustr:match("^,")) then
        if (affinity ~= nil) then
            num_threads,cpu_list = likwid.cpustr_to_cpulist(cpustr)
        else
            num_threads,cpu_list = likwid.cpustr_to_cpulist_physical(cpustr)
        end
    end
    if (num_threads <= 0) then
        print_stderr("Failed to parse cpulist " .. cpustr)
        close_and_exit(1)
    end
    if verbose > 0 and quiet == 0 then
        print_stdout("Evaluated CPU string to CPUs: " .. cpustr)
    end
end


if print_domains and num_threads > 0 then
    outstr = ""
    for i, cpu in pairs(cpu_list) do
        outstr = outstr .. delimiter .. cpu
    end
    print_stdout(outstr:sub(2,outstr:len()))
    close_and_exit(0)
elseif print_domains then
    for k,v in pairs(affinity["domains"]) do
        print_stdout(string.format("Domain %s:", v["tag"]))
        print_stdout("\t" .. table.concat(v["processorList"], delimiter))
        print_stdout("")
    end
    close_and_exit(0)
end

if num_threads == 0 then
    num_threads, cpu_list = likwid.cpustr_to_cpulist("N")
end
if (#arg == 0) then
    print_stderr("Executable must be given on commandline")
    close_and_exit(1)
end

if interleaved_policy then
    if numainfo["numberOfNodes"] > 1 then
        if verbose > 0 and quiet == 0 then
            print_stdout("Set mem_policy to interleaved")
        end
        likwid.setMemInterleaved(num_threads, cpu_list)
    else
        print_stdout("No need to set mem_policy to interleaved, only one NUMA node available")
    end
end
if membind_policy then
    if numainfo["numberOfNodes"] > 1 then
        if verbose > 0 and quiet == 0 then
            print_stdout("Set mem_policy to membind")
        end
        likwid.setMembind(num_threads, cpu_list)
    else
        print_stdout("No need to set mem_policy to membind, only one NUMA node available")
    end
end

if sweep_sockets then
    if verbose > 0 and quiet == 0 then
        print_stdout("Sweeping memory")
    end
    likwid.memSweep(num_threads, cpu_list)
end

local omp_threads = os.getenv("OMP_NUM_THREADS")
if omp_threads == nil then
    likwid.setenv("OMP_NUM_THREADS",tostring(math.tointeger(num_threads)))
elseif num_threads > tonumber(omp_threads) and (quiet == 0 and verbose > 0) then
    print_stdout(string.format("Environment variable OMP_NUM_THREADS already set to %s but %d cpus required", omp_threads,num_threads))
end
if omp_threads and tonumber(omp_threads) < num_threads then
    num_threads = tonumber(omp_threads)
    for i=#cpu_list,num_threads+1,-1 do
        cpu_list[i] = nil
    end
end

likwid.setenv("KMP_AFFINITY","disabled")
local likwid_hwthreads = {}
for i=1,#cpu_list do
    table.insert(likwid_hwthreads, tostring(math.tointeger(cpu_list[i])))
end
likwid.setenv("LIKWID_HWTHREADS", table.concat(likwid_hwthreads, ","))

if os.getenv("CILK_NWORKERS") == nil then
    likwid.setenv("CILK_NWORKERS", tostring(math.tointeger(num_threads)))
end
if skip_mask then
    likwid.setenv("LIKWID_SKIP", skip_mask)
end

if num_threads > 1 then
    local pinString = tostring(math.tointeger(cpu_list[2]))
    for i=3,likwid.tablelength(cpu_list) do
        pinString = pinString .. "," .. tostring(math.tointeger(cpu_list[i]))
    end
    pinString = pinString .. "," .. tostring(math.tointeger(cpu_list[1]))
    likwid.setenv("LIKWID_PIN", pinString)

    local preload = os.getenv("LD_PRELOAD")
    if preload == nil then
        likwid.setenv("LD_PRELOAD",likwid.pinlibpath)
    else
        likwid.setenv("LD_PRELOAD",likwid.pinlibpath .. ":" .. preload)
    end
    local ldpath = os.getenv("LD_LIBRARY_PATH")
    local libpath = likwid.pinlibpath:match("([^%s]+)/[^%s]+.so")
    if ldpath == nil then
        likwid.setenv("LD_LIBRARY_PATH", libpath)
    elseif libpath and not ldpath:match(libpath) then
        likwid.setenv("LD_LIBRARY_PATH", libpath..":"..ldpath)
    end
else
    likwid.setenv("LIKWID_PIN", tostring(math.tointeger(cpu_list[1])))
    likwid.pinProcess(cpu_list[1], quiet)
end

local exec = table.concat(execList," ")
if verbose > 0 and quiet == 0 then
    print_stdout("Running: " .. exec)
    mask = 0
    for _, c in pairs(cpu_list) do
        -- Check whether the Lua version has bit32 module (>= 5.2)
        if not bit32 then
            mask = mask + 2 ^ c
        else
            mask = bit32.bor(mask, bit32.lshift(1, c))
        end
    end
    print_stdout(string.format("Using %d thread(s) (cpuset: 0x%x)", num_threads, mask))
end
local pid = likwid.startProgram(table.concat(execList," "), num_threads, cpu_list)
if (pid == nil) then
    print_stderr("Failed to execute command: ".. exec)
    close_and_exit(1)
end

local exitvalue = likwid.waitpid(pid)

likwid.putAffinityInfo()
likwid.putTopology()
likwid.putConfiguration()
close_and_exit(exitvalue)
