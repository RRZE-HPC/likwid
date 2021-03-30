#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-mpirun.lua
 *
 *      Description: A wrapper script to pin threads spawned by MPI processes and
 *                   measure hardware performance counters
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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
    print_stdout(string.format("likwid-mpirun -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

local function examples()
    print_stdout("Examples:")
    print_stdout("Run 32 processes on hosts in hostlist")
    print_stdout("likwid-mpirun -np 32 ./a.out")
    print_stdout("")
    print_stdout("Run 1 MPI process on each socket")
    print_stdout("likwid-mpirun -nperdomain S:1 ./a.out")
    print_stdout("Total amount of MPI processes is calculated using the number of hosts in the hostfile")
    print_stdout("")
    print_stdout("For hybrid MPI/OpenMP jobs you need to set the -pin option")
    print_stdout("Starts 2 MPI processes on each host, one on socket 0 and one on socket 1")
    print_stdout("Each MPI processes may start 2 OpenMP threads pinned to the first two CPUs on each socket")
    print_stdout("likwid-mpirun -pin S0:0-1_S1:0-1 ./a.out")
    print_stdout("")
    print_stdout("Run 2 processes on each socket and measure the MEM performance group")
    print_stdout("likwid-mpirun -nperdomain S:2 -g MEM ./a.out")
    print_stdout("Only one process on a socket measures the Uncore/RAPL counters, the other one(s) only HWThread-local counters")
    print_stdout("")
end

local function usage()
    version()
    print_stdout("A wrapper script to pin threads spawned by MPI processes and measure hardware performance counters.\n")
    print_stdout("Options:")
    print_stdout("-h, --help\t\t Help message")
    print_stdout("-v, --version\t\t Version information")
    print_stdout("-d, --debug\t\t Debugging output")
    print_stdout("-n/-np <count>\t\t Set the number of processes")
    print_stdout("-nperdomain <domain>\t Set the number of processes per node by giving an affinity domain and count")
    print_stdout("-pin <list>\t\t Specify pinning of threads. CPU expressions like likwid-pin separated with '_'")
    print_stdout("-t/-tpp <count>\t\t Set the number of threads per MPI process")
    print_stdout("--dist <d>(:order)\t Specify the CPU distance between MPI processes. Possible orders are close and spread.")
    print_stdout("-s, --skip <hex>\t Bitmask with threads to skip")
    print_stdout("-mpi <id>\t\t Specify which MPI should be used. Possible values: openmpi, intelmpi, mvapich2 or slurm")
    print_stdout("\t\t\t If not set, module system is checked")
    print_stdout("-omp <id>\t\t Specify which OpenMP should be used. Possible values: gnu and intel")
    print_stdout("\t\t\t Only required for statically linked executables.")
    print_stdout("-hostfile\t\t Use custom hostfile instead of searching the environment")
    print_stdout("-g/-group <perf>\t Set a likwid-perfctr conform event set for measuring on nodes")
    print_stdout("-m/-marker\t\t Activate marker API mode")
    print_stdout("-O\t\t\t Output easily parseable CSV instead of fancy tables")
    print_stdout("-o/--output <file>\t Write output to a file. The file is reformatted according to the suffix.")
    print_stdout("-f\t\t\t Force execution (and measurements). You can also use environment variable LIKWID_FORCE")
    print_stdout("-e, --env <key>=<value>\t Set environment variables for MPI processes")
    print_stdout("--mpiopts <str>\t Hand over options to underlying MPI. Please use proper quoting.")
    print_stdout("")
    print_stdout("Processes are pinned to physical hardware threads first. For syntax questions see likwid-pin")
    print_stdout("")
    print_stdout("For CPU selection and which MPI rank measures Uncore counters the system topology")
    print_stdout("of the current system is used. There is currently no possibility to overcome this")
    print_stdout("limitation by providing a topology file or similar.")
    print_stdout("")
    examples()
end

local np = 0
local ppn = 0
local dist = 1
local tpp = 1
local tpp_orderings = {"close", "spread"}
local tpp_ordering = "spread"
local nperdomain = nil
local npernode = 0
local cpuexprs = {}
local perfexprs = {}
local hostfile = nil
local hosts = {}
local perf = {}
local mpitype = nil
local slurm_involved = false
local omptype = nil
local skipStr = ""
local executable = {}
local envsettings = {}
local mpiopts = nil
local debug = false
local likwiddebug = false
local use_marker = false
local use_csv = false
local outfile = nil
local force = false
local print_stats = false
if os.getenv("LIKWID_FORCE") ~= nil then
    force = true
end

local LIKWID_PIN="<INSTALLED_PREFIX>/bin/likwid-pin"
local LIKWID_PERFCTR="<INSTALLED_PREFIX>/bin/likwid-perfctr"

local readHostfile = nil
local writeHostfile = nil
local getEnvironment = nil
local executeCommand = nil
local mpiexecutable = nil
local hostpattern = "([%.%a%d_-]+)"

local function abspath(cmd)
    local pathlist = os.getenv("PATH")
    for p in pathlist:gmatch("[^:]*") do
        if p then
            local t = string.format("%s/%s", p, cmd)
            if likwid.access(t, "e") == 0 then
                return t
            end
        end
    end
    return nil
end

local function readHostfileOpenMPI(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print_stdout("DEBUG: Reading hostfile in openmpi style")
    end
    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname, slots, maxslots = line:match("^"..hostpattern.."%s+slots=(%d*)%s+max%-slots=(%d*)")
            if not hostname then
                hostname, slots = line:match("^"..hostpattern.."%s+slots=(%d*)")
                if not hostname then
                    hostname = line:match("^"..hostpattern)
                    slots = 1
                    maxslots = 1
                end
            end
            local found = false
            for i, host in pairs(hostlist) do
                if host["hostname"] == hostname then
                    if slots and host["slots"] then
                        host["slots"] = host["slots"] + tonumber(slots)
                    end
                    if maxslots and host["maxslots"] then
                        host["maxslots"] = host["maxslots"] + tonumber(maxslots)
                    end
                    break
                end
            end
            if not found then
                table.insert(hostlist, {hostname=hostname, slots=tonumber(slots), maxslots=tonumber(maxslots)})
            end
        end
    end
    local topo = likwid.getCpuTopology()
    for i,host in pairs(hostlist) do
        if host["slots"] == nil or host["slots"] == 0 then
            host["slots"] = topo.numHWThreads
        end
        if host["maxslots"] == nil or host["maxslots"] == 0 then
            host["maxslots"] = topo.numHWThreads
        end
    end
    if debug then
        print_stdout("Available hosts for scheduling:")
        s = string.format("%-20s\t%s\t%s\t%s", "Host", "Slots", "MaxSlots", "Interface")
        print_stdout(s)
        for i, host in pairs(hostlist) do
            s = string.format("%-20s\t%s\t%s\t%s", host["hostname"], host["slots"], host["maxslots"],"")
            print_stdout (s)
        end
    end
    return hostlist
end

local function writeHostfileOpenMPI(hostlist, filename)
    if filename == nil or filename == "" then
        return
    end

    local f = io.open(filename, "w")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    for i, hostcontent in pairs(hostlist) do
        str = hostcontent["hostname"]
        if hostcontent["slots"] then
            str = str .. string.format(" slots=%d", hostcontent["slots"])
        end
        if hostcontent["maxslots"] then
            str = str .. string.format(" max-slots=%d", hostcontent["maxslots"])
        end
        f:write(str .. "\n")
    end
    f:close()
end

local function getEnvironmentOpenMPI()
    return {}
end

local function executeOpenMPI(wrapperscript, hostfile, env, nrNodes)
    local bindstr = ""
    if wrapperscript.sub(1,1) ~= "/" then
        wrapperscript = os.getenv("PWD").."/"..wrapperscript
    end


    ver1,ver2 = getMpiVersion()
    if ver1 == 1 then
        if ver2 >= 7 then
            bindstr = "--bind-to none"
        elseif ver2 == "6" then
            bindstr = "--bind-to-none"
        end
    elseif ver1 == 2 then
        bindstr = "--bind-to none"
    elseif ver1 == 3 then
        bindstr = "--bind-to none"
    end

    local cmd = string.format("%s -hostfile %s %s -np %d -npernode %d %s %s",
                                mpiexecutable, hostfile, bindstr,
                                np, ppn, mpiopts or "", wrapperscript)
    if debug then
        print_stdout("EXEC: "..cmd)
    end
    local ret = os.execute(cmd)
    return ret
end

local function readHostfileIntelMPI(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print_stdout("DEBUG: Reading hostfile in intelmpi style")
    end
    local topo = likwid.getCpuTopology()
    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname, slots = line:match("^"..hostpattern..":(%d+)")
            if not hostname then
                hostname = line:match("^"..hostpattern)
                slots = topo["numHWThreads"]
            end
            table.insert(hostlist, {hostname=hostname, slots=tonumber(slots), maxslots=tonumber(slots)})
        end
    end
    if debug then
        print_stdout("Available hosts for scheduling:")
        s = string.format("%-20s\t%s\t%s\t%s", "Host", "Slots", "MaxSlots", "Interface")
        print_stdout(s)
        for i, host in pairs(hostlist) do
            s = string.format("%-20s\t%s\t%s\t%s", host["hostname"], host["slots"], host["maxslots"],"")
            print_stdout (s)
        end
    end
    return hostlist
end

local function writeHostfileIntelMPI(hostlist, filename)
    if filename == nil or filename == "" then
        return
    end

    local f = io.open(filename, "w")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    for i, hostcontent in pairs(hostlist) do
        str = hostcontent["hostname"]
        if hostcontent["slots"] then
            str = str .. string.format(":%d", hostcontent["slots"])
        end
        f:write(str .. "\n")
    end
    f:close()
end

local function getEnvironmentIntelMPI()
    local env = {}
    env['I_MPI_PIN']='off'
    env['KMP_AFFINITY']='disabled'
    return env
end

local function executeIntelMPI(wrapperscript, hostfile, env, nrNodes)
    local use_hydra = true
    local mpi_connect = "ssh"
    if wrapperscript.sub(1,1) ~= "/" then
        wrapperscript = os.getenv("PWD").."/"..wrapperscript
    end
    if hostfile.sub(1,1) ~= "/" then
        hostfile = os.getenv("PWD").."/"..hostfile
    end
    local path = ""
    local f = io.popen(string.format("dirname %s", mpiexecutable))
    if f ~= nil then
        path = f:read("*line")
        f:close()
    end
    if likwid.access(string.format("%s/mpdboot", path), "x") == 0 then
        use_hydra = false
    end
    for i, env in pairs({"MPIHOME", "MPI_HOME", "MPI_ROOT", "MPI_BASE"}) do
        if likwid.access(string.format("%s/bin/mpdboot", os.getenv(env)), "x") == 0 then
            use_hydra = false
            path = string.format("%s/bin",os.getenv(env))
            break
        end
    end

    local envstr = ""
    for i, e in pairs(env) do
        if use_hydra then
            envstr = envstr .. string.format("-genv %s %s ", i, e)
        else
            envstr = envstr .. string.format("-env %s %s ", i, e)
        end
    end

    if mpiopts and mpiopts:len() > 0 then
        envstr = envstr .. mpiopts
    end
    if os.getenv("LIKWID_MPI_CONNECT") ~= nil then
        mpi_connect = os.getenv("LIKWID_MPI_CONNECT")
    end

    if debug then
        if use_hydra == false then
            print_stdout(string.format("EXEC: %s/mpdboot -r %s -n %d -f %s", path, mpi_connect, nrNodes, hostfile))
            print_stdout(string.format("EXEC: %s/mpiexec -perhost %d %s -np %d %s", path, ppn, envstr, np, wrapperscript))
            print_stdout(string.format("EXEC: %s/mpdallexit", path))
        else
            print_stdout(string.format("%s %s -f %s -np %d -perhost %d %s",mpiexecutable, envstr, hostfile, np, ppn, wrapperscript))
        end
    end

    --os.execute(string.format("%s -genv I_MPI_PIN 0 -f %s -np %d -perhost %d %s",mpiexecutable, hostfile, np, ppn, wrapperscript))
    local ret = 0
    if use_hydra == false then
        ret = os.execute(string.format("%s/mpdboot -r %s -n %d -f %s", path, mpi_connect, nrNodes, hostfile))
        ret = os.execute(string.format("%s/mpiexec -perhost %d %s -np %d %s", path, ppn, envstr, np, wrapperscript))
        ret = os.execute(string.format("%s/mpdallexit", path))
    else
        ret = os.execute(string.format("%s %s -f %s -np %d -perhost %d %s",mpiexecutable, envstr, hostfile, np, ppn, wrapperscript))
    end
    return ret
end

local function readHostfileMvapich2(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print_stdout("DEBUG: Reading hostfile in mvapich2 style")
    end
    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname, slots, interface = line:match("^"..hostpattern..":(%d+):([%a%d]+)")
            if not hostname then
                hostname, slots = line:match("^"..hostpattern..":(%d+)")
                if not hostname then
                    hostname = line:match("^"..hostpattern)
                    slots = 1
                    interface = nil
                else
                    interface = nil
                end
            end
            table.insert(hostlist, {hostname=hostname, slots=tonumber(slots), maxslots=tonumber(slots), interface=interface})
        end
    end
    if debug then
        if debug then
            print_stdout("Available hosts for scheduling:")
            s = string.format("%-20s\t%s\t%s\t%s", "Host", "Slots", "MaxSlots", "Interface")
            print_stdout(s)
            for i, host in pairs(hostlist) do
                s = string.format("%-20s\t%s\t%s\t%s", host["hostname"], host["slots"], host["maxslots"],"")
                print_stdout (s)
            end
        end
    end
    return hostlist
end

local function writeHostfileMvapich2(hostlist, filename)
    if filename == nil or filename == "" then
        return
    end

    local f = io.open(filename, "w")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    for i, hostcontent in pairs(hostlist) do
        str = hostcontent["hostname"]
        if hostcontent["slots"] then
            str = str .. string.format(":%d", hostcontent["slots"])
        end
        if hostcontent["interface"] then
            str = str .. string.format(":%s", hostcontent["interface"])
        end
        f:write(str .. "\n")
    end
    f:close()
end

local function getEnvironmentMvapich2()
    local env = {}
    env['MV2_ENABLE_AFFINITY'] = "0"
    return env
end

local function executeMvapich2(wrapperscript, hostfile, env, nrNodes)
    if wrapperscript.sub(1,1) ~= "/" then
        wrapperscript = os.getenv("PWD").."/"..wrapperscript
    end
    if hostfile.sub(1,1) ~= "/" then
        hostfile = os.getenv("PWD").."/"..hostfile
    end

    local envstr = ""
    for i, e in pairs(env) do
        envstr = envstr .. string.format("-genv %s %s ", i, e)
    end

    local cmd = string.format("%s -f %s -np %d -ppn %d %s %s %s",
                                mpiexecutable, hostfile,
                                np, ppn, envstr, mpiopts or "", wrapperscript)
    if debug then
        print_stdout("EXEC: "..cmd)
    end
    local ret = os.execute(cmd)
    return ret
end


local function readHostfilePBS(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print_stdout("DEBUG: Reading hostfile from batch system")
    end

    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname = line:match("^"..hostpattern)
            local found = false
            for i, host in pairs(hostlist) do
                if host["hostname"] == hostname then
                    host["slots"] = host["slots"] + 1
                    host["maxslots"] = host["maxslots"] + 1
                    found = true
                    break
                end
            end
            if not found then
                table.insert(hostlist, {hostname=hostname, slots=1, maxslots=1})
            end
        end
    end
    if debug then
        print_stdout("Available hosts for scheduling:")
        s = string.format("%-20s\t%s\t%s\t%s", "Host", "Slots", "MaxSlots", "Interface")
        print_stdout(s)
        for i, host in pairs(hostlist) do
            s = string.format("%-20s\t%s\t%s\t%s", host["hostname"], host["slots"], host["maxslots"],"")
            print_stdout (s)
        end
    end
    return hostlist
end

local function readHostfileSlurm(hostlist)
    local stasks_per_node = 1
    local scpus_per_node = 1
    if os.getenv("SLURM_TASKS_PER_NODE") ~= nil then
        stasks_per_node = tonumber(os.getenv("SLURM_TASKS_PER_NODE"):match("(%d+)"))
    end
    if os.getenv("SLURM_CPUS_ON_NODE") ~= nil then
        scpus_per_node = tonumber(os.getenv("SLURM_CPUS_ON_NODE"):match("(%d+)"))
    end
    nperhost = stasks_per_node * scpus_per_node

    if hostlist and nperhost then
        hostfile = write_hostlist_to_file(hostlist, nperhost)
        hosts = readHostfilePBS(hostfile)
        os.remove(hostfile)
    end
    return hosts
end

function write_hostlist_to_file(hostlist, nperhost)
    if hostlist == "" then
        return {}
    end
    outlist = {}
    cmd = string.format("scontrol show hostname %s", hostlist)
    f = io.popen(cmd, 'r')
    if f ~= nil then
        local s = assert(f:read('*a'))
        f:close()
        for i,line in pairs(likwid.stringsplit(s, "\n")) do
            table.insert(outlist, line)
        end
        fname = string.format("/tmp/hostlist.%d", likwid.getpid())
        f = io.open(fname, "w")
        if f ~= nil then
            for i=1,#outlist do
                for j=1, nperhost do
                    f:write(outlist[i].."\n")
                end
            end
            f:close()
        end
        return fname
    else
        print_stderr("ERROR: Cannot transform SLURM hostlist to list of hosts")
    end
end

local function writeHostfileSlurm(hostlist, filename)
    l = {}
    for i, h in pairs(hostlist) do
        table.insert(l, h["hostname"])
    end
    cmd = string.format("scontrol show hostlist %s", table.concat(l,","))
    f = io.popen(cmd, 'r')
    if f ~= nil then
        likwid.setenv("SLURM_NODELIST", f:read('*l'))
        f:close()
    else
        print_stderr("ERROR: Cannot transform list of hosts to SLURM hostlist format")
    end
end

local function getEnvironmentSlurm()
    return {}
end

local function executeSlurm(wrapperscript, hostfile, env, nrNodes)
    if wrapperscript.sub(1,1) ~= "/" then
        wrapperscript = os.getenv("PWD").."/"..wrapperscript
    end
    local exec = string.format("srun --nodes %d --ntasks %d --ntasks-per-node=%d --cpu_bind=none %s %s",
                                nrNodes, np, ppn, mpiopts or "", wrapperscript)
    if debug then
        print_stdout("EXEC: "..exec)
    end
    local ret = os.execute(exec)
    return ret
end

local function getNumberOfNodes(hostlist)
    local n = 0
    for i, h in pairs(hostlist) do
        hostname = h["hostname"]
        exists = false
        for j=1,i-1 do
            if hostlist[i]["hostname"] == hostlist[j]["hostname"] then
                exists = true
            end
        end
        if not exists then
            n = n + 1
        end
    end
    return n
end

local function getMpiType()
    local mpitype = nil
    if os.getenv("SLURM_JOB_ID") ~= nil then
        mpitype = "slurm"
        slurm_involved = true
    end
    cmd = "bash -c 'tclsh /apps/modules/modulecmd.tcl sh list -t' 2>&1"
    local f = io.popen(cmd, 'r')
    if f == nil then
        cmd = os.getenv("SHELL").." -c 'module -t list' 2>&1"
        f = io.popen(cmd, 'r')
    end
    if f ~= nil then
        local s = assert(f:read('*a'))
        f:close()
        s = string.gsub(s, '^%s+', '')
        s = string.gsub(s, '%s+$', '')
        for i,line in pairs(likwid.stringsplit(s, "\n")) do
            if line:match("[iI]ntel[mM][pP][iI]") or (line:match("[iI]ntel") and line:match("[mM][pP][iI]")) then
                mpitype = "intelmpi"
                --libmpi%a*.so
            elseif line:match("[oO]pen[mM][pP][iI]") or (line:match("[oO]pen") and line:match("[mM][pP][iI]")) then
                mpitype = "openmpi"
                --libmpi.so
            elseif line:match("mvapich2") then
                mpitype = "mvapich2"
                --libmpich.so
            end
        end
    end
    for i, exec in pairs({"mpiexec.hydra", "mpiexec", "mpirun", "srun"}) do
        f = io.popen(string.format("which %s 2>/dev/null", exec), 'r')
        if f ~= nil then
            local s = f:read('*line')
            if s ~= nil then
                f:close()
                f = io.popen(string.format("%s --help 2>/dev/null", s), 'r')
                if f ~= nil then
                    out = f:read("*a")
                    b,e = out:find("Intel")
                    if (b ~= nil) then
                        mpitype = "intelmpi"
                        break
                    end
                    b,e = out:find("OpenRTE")
                    if (b ~= nil) then
                        mpitype = "openmpi"
                        break
                    end
                    b,e = out:find("MPICH")
                    if (b ~= nil) then
                        mpitype = "mvapich2"
                        break
                    else
                        b,e = out:find("MVAPICH2")
                        if (b ~= nil) then
                            mpitype = "mvapich2"
                            break
                        end
                    end
                    b,e = out:find("srun")
                    if (b ~= nil) then
                        mpitype = "slurm"
                        slurm_involved = true
                        break
                    end
                end
            end
        end
    end
    if mpitype then
        local mpi_bootstrap = os.getenv("I_MPI_HYDRA_BOOTSTRAP")
        if mpi_bootstrap == "slurm" then
            slurm_involved = "slurm"
        end
    end
    if not mpitype then
        print_stderr("WARN: No supported MPI loaded in module system")
    end
    return mpitype
end

function getMpiVersion()
    maj = nil
    min = nil
    intel_match = "Version (%d+) Update (%d+)"
    intel_build_match = "Version (%d+) Build (%d+)"
    intel_match_old = "Version (%d+).(%d+).%d+"
    openmpi_match = "(%d+)%.(%d+)%.%d+"
    slurm_match = "slurm (%d+).(%d+).%d+"
    for i, exec in pairs({"mpiexec.hydra", "mpiexec", "mpirun", "srun"}) do
        f = io.popen(string.format("which %s 2>/dev/null", exec), 'r')
        if f ~= nil then
            mpiexec = f:read("*line")
            if mpiexec then
                cmd = mpiexec .. " --version"
                f:close()
                f = io.popen(cmd, 'r')
                if f ~= nil then
                    local t = f:read("*all")
                    f:close()
                    for l in t:gmatch("[^\r\n]+") do
                        if l:match(intel_match) then
                            maj, min = l:match(intel_match)
                            maj = tonumber(maj)
                            min = tonumber(min)
                        elseif l:match(intel_build_match) then
                            maj, min = l:match(intel_build_match)
                            maj = tonumber(maj)
                            min = tonumber(min)
                        elseif l:match(intel_match_old) then
                            maj, min = l:match(intel_match_old)
                            maj = tonumber(maj)
                            min = tonumber(min)
                        elseif l:match(openmpi_match) then
                            maj, min = l:match(openmpi_match)
                            maj = tonumber(maj)
                            min = tonumber(min)
                        elseif l:match(slurm_match) then
                            maj, min = l:match(slurm_match)
                            maj = tonumber(maj)
                            min = tonumber(min)
                        end
                    end
                end
            end
        end
    end
    return maj, min
end

local function getMpiExec(mpitype)
    testing = {}
    if mpitype == "intelmpi" then
        testing = {"mpiexec.hydra", "mpiexec"}
        executeCommand = executeIntelMPI
        readHostfile = readHostfileIntelMPI
        writeHostfile = writeHostfileIntelMPI
        getEnvironment = getEnvironmentIntelMPI
    elseif mpitype == "openmpi" then
        testing = {"mpiexec", "mpirun"}
        executeCommand = executeOpenMPI
        readHostfile = readHostfileOpenMPI
        writeHostfile = writeHostfileOpenMPI
        getEnvironment = getEnvironmentOpenMPI
    elseif mpitype == "mvapich2" then
        testing = {"mpiexec", "mpirun"}
        executeCommand = executeMvapich2
        readHostfile = readHostfileMvapich2
        writeHostfile = writeHostfileMvapich2
        getEnvironment = getEnvironmentMvapich2
    elseif mpitype == "slurm" then
        testing = {"srun"}
        executeCommand = executeSlurm
        readHostfile = readHostfileSlurm
        writeHostfile = writeHostfileSlurm
        getEnvironment = getEnvironmentSlurm
    end
    for i, exec in pairs(testing) do
        f = io.popen(string.format("which %s 2>/dev/null", exec), 'r')
        if f ~= nil then
            local s = f:read('*line')
            if s ~= nil then
                mpiexecutable = s
            end
        end
    end
    if os.getenv("I_MPI_HYDRA_BOOTSTRAP") == "slurm" then
        slurm_involved = true
    end
    if os.getenv("SLURM_JOB_ID") ~= nil then
        slurm_involved = true
    end
    if mpiexecutable == nil then
        local testmpitype = getMpiType()
        print_stderr(string.format("ERROR: Cannot find executable for MPI type %s but %s", mpitype, testmpitype))
        os.exit(1)
    end
end

local function getOmpType()
    local cmd = string.format("ldd %s 2>/dev/null", executable[1])
    local f = io.popen(cmd, 'r')
    if f == nil then
        cmd = string.format("ldd $(basename %s) 2>/dev/null", executable[1])
        f = io.popen(cmd, 'r')
    end
    omptype = nil
    dyn_linked = true
    if f ~= nil then
        local s = f:read('*a')
        f:close()
        for i,line in pairs(likwid.stringsplit(s, "\n")) do
            if line:match("libgomp.so") then
                omptype = "gnu"
                break
            elseif line:match("libiomp%d*.so") then
                omptype = "intel"
                break
            elseif line:match("not a dynamic executable") then
                omptype = "none"
                dyn_linked = false
                break
            end
        end
    end
    if not omptype and dyn_linked == false then
        print_stderr("WARN: Cannot get OpenMP variant from executable, trying module system")
        cmd = "bash -c 'tclsh /apps/modules/modulecmd.tcl sh list -t' 2>&1"
        local f = io.popen(cmd, 'r')
        if f == nil then
            cmd = os.getenv("SHELL").." -c 'module -t list' 2>&1"
            f = io.popen(cmd, 'r')
        end
        if f ~= nil then
            local s = f:read('*a')
            f:close()
            s = string.gsub(s, '^%s+', '')
            s = string.gsub(s, '%s+$', '')
            for i,line in pairs(likwid.stringsplit(s, "\n")) do
                if line:match("[iI]ntel") or line:match("[iI][cC][cC]") then
                    omptype = "intel"
                elseif line:match("[gG][nN][uU]") or line:match("[gG][cC][cC]") then
                    omptype = "gnu"
                end
            end
        end
        if not omptype then
            print_stderr("WARN: No supported OpenMP loaded in module system")
        end
    end
    if omptype == "none" then
        return nil
    end
    return omptype
end

local function assignHosts(hosts, np, ppn, tpp)
    tmp = np
    if tpp > 1 then
        tmp = np * tpp
    end
    newhosts = {}
    current = 0
    if debug then
        print_stdout(string.format("DEBUG: Assign %d processes with %d per node and %d threads per process to %d hosts", np, ppn, tpp, #hosts))
    end
    local break_while = false
    while tmp > 0 and #hosts > 0 do
        for i, host in pairs(hosts) do
            if host["slots"] and host["slots"] >= ppn*tpp then
                if host["maxslots"] and host["maxslots"] < ppn*tpp then
                    table.insert(newhosts, {hostname=host["hostname"],
                                            slots=host["maxslots"],
                                            maxslots=host["maxslots"],
                                            interface=host["interface"]})
                    if debug then
                        print_stdout(string.format("DEBUG: Add Host %s with %d slots to host list", host["hostname"], host["maxslots"]))
                    end
                    current = host["maxslots"]
                    hosts[i] = nil
                else
                    table.insert(newhosts, {hostname=host["hostname"],
                                            slots=ppn*tpp,
                                            maxslots=host["slots"],
                                            interface=host["interface"]})
                    if debug then
                        print_stdout(string.format("DEBUG: Add Host %s with %d slots to host list", host["hostname"], ppn*tpp))
                    end
                    current = ppn*tpp
                    hosts[i] = nil
                end
            elseif host["slots"] and host["slots"] < ppn then
                --[[if host["maxslotsno"] then
                    if host["maxslots"] < ppn then
                        print_stderr(string.format("WARN: Oversubscription for host %s needed, but max-slots set to %d.",
                                                host["hostname"], host["maxslots"]))
                        table.insert(newhosts, {hostname=host["hostname"],
                                                slots=host["maxslots"],
                                                maxslots=host["maxslots"],
                                                interface=host["interface"]})
                        current = host["maxslots"]
                        host["maxslots"] = 0
                        hosts[i] = nil
                    else
                        print_stderr(string.format("WARN: Oversubscription for host %s.", host["hostname"]))
                        table.insert(newhosts, {hostname=host["hostname"],
                                            slots=ppn,
                                            maxslots=host["maxslots"],
                                            interface=host["interface"]})
                        current = ppn
                    end
                else
                    print_stderr(string.format("WARN: Oversubscription for host %s.", host["hostname"]))
                    table.insert(newhosts, {hostname=host["hostname"],
                                        slots=ppn,
                                        maxslots=host["slots"],
                                        interface=host["interface"]})
                    current = ppn
                end]]
                print_stderr(string.format("ERROR: Oversubscription required. Host %s has only %s slots but %d needed per host", host["hostname"], host["slots"], ppn))
                if slurm_involved then
                    print_stderr("ERROR: In SLURM environments, it might be a problem with --ntasks (the slots) and --cpus-per-task options")
                end
                print_stderr("ERROR: Oversubscription may be forced by setting '-f' on the command line.")
                os.exit(1)
            else
                table.insert(newhosts, {hostname=host["hostname"],
                                        slots=ppn*tpp,
                                        maxslots=host["slots"],
                                        interface=host["interface"]})
                if debug then
                    print_stdout(string.format("DEBUG: Add Host %s with %d slots to host list", host["hostname"], ppn*tpp))
                end
                current = ppn*tpp
            end
            tmp = tmp - current
            if tmp < 1 then
                break_while = true
                break
            elseif tmp < ppn*tpp then
                ppn = tmp
            end
        end
        if break_while then
            break
        end
    end
    if current < np then
        print_stdout(string.format("WARN: Only %d processes out of %d can be assigned, running with %d processes", current, np, current))
        np = current
        ppn = np/#newhosts
    end
    for i=1, #newhosts do
        if newhosts[i] then
            for j=i+1,#newhosts do
                if newhosts[j] then
                    if newhosts[i]["hostname"] == newhosts[j]["hostname"] then
                        newhosts[i]["slots"] = newhosts[i]["slots"] + newhosts[j]["slots"]
                        if newhosts[i]["maxslots"] ~= nil and newhosts[j]["maxslots"] ~= nil then
                            newhosts[i]["maxslots"] = newhosts[i]["maxslots"] + newhosts[j]["maxslots"]
                        end
                        if newhosts[i]["slots"] > ppn then
                            ppn = newhosts[i]["slots"]
                        end
                        table.remove(newhosts, j)
                    end
                end
            end
        end
    end
    if debug then
        print_stdout("DEBUG: Scheduling on hosts:")
        for i, h in pairs(newhosts) do
            if h["maxslots"] ~= nil then
                str = string.format("DEBUG: Host %s with %d slots (max. %d slots)",
                                h["hostname"],h["slots"],h["maxslots"])
            else
                str = string.format("DEBUG: Host %s with %d slots", h["hostname"],h["slots"])
            end
            if h["interface"] then
                str = str.. string.format(" using interface %s", h["interface"])
            end
            print_stdout(str)
        end
    end
    return newhosts, ppn
end

local function calculatePinExpr(cpuexprs)
    local newexprs = {}
    for i, expr in pairs(cpuexprs) do
        local strList = {}
        amount, list = likwid.cpustr_to_cpulist(expr)
        for _, c in pairs(list) do
            table.insert(strList, c)
        end
        table.insert(newexprs, strList)
    end
    return newexprs
end

local function calculateCpuExprs(nperdomain, cpuexprs)
    local topo = likwid.getCpuTopology()
    local affinity = likwid.getAffinityInfo()
    local domainlist = {}
    local newexprs = {}
    domainname, count, threads = nperdomain:match("[E]*[:]*([NSCM]*):(%d+)[:]*(%d*)")
    count = math.tointeger(count)
    threads = math.tointeger(threads)
    if threads == nil then threads = 1 end

    for i, domain in pairs(affinity["domains"]) do
        if domain["tag"]:match(domainname.."%d*") then
            table.insert(domainlist, i)
        end
    end
    if debug then
        local str = "DEBUG: NperDomain string "..nperdomain.." covers the domains: "
        for i, idx in pairs(domainlist) do
            str = str .. affinity["domains"][idx]["tag"] .. " "
        end
        print_stdout(str)
    end

    for i, domidx in pairs(domainlist) do
        local sortedlist = {}
        if tpp_ordering == "spread" then
            for off=1,topo["numThreadsPerCore"] do
                for i=0,affinity["domains"][domidx]["numberOfProcessors"]/topo["numThreadsPerCore"] do
                    table.insert(sortedlist, affinity["domains"][domidx]["processorList"][off + (i*topo["numThreadsPerCore"])])
                end
            end
        elseif tpp_ordering == "close" then
            for i=0,affinity["domains"][domidx]["numberOfProcessors"] do
                table.insert(sortedlist, affinity["domains"][domidx]["processorList"][i])
            end
        end
        local tmplist = {}
        for j=1,count do
            local tmplist = {}
            for t=1,threads do
                if sortedlist[1] then
                    table.insert(tmplist, tostring(sortedlist[1]))
                    table.remove(sortedlist, 1)
                end
            end
            table.insert(newexprs, tmplist)
            if dist > threads then
                for t=1, dist-threads do table.remove(sortedlist, 1) end
            end
        end
    end
    if debug then
        local str = "DEBUG: Resolved NperDomain string "..nperdomain.." to CPUs: "
        for i, expr in pairs(newexprs) do
            str = str .. "[" .. table.concat(expr,",") .. "]" .. " "
        end
        print_stdout(str)
    end
    return newexprs
end

local function createEventString(eventlist)
    if eventlist == nil or #eventlist == 0 then
        print_stderr("ERROR: Empty event list. Failed to create event set string")
        return ""
    end
    local str = ""
    if eventlist[1] ~= nil and eventlist[1]["Event"] ~= nil and eventlist[1]["Counter"] ~= nil then
        str = str .. eventlist[1]["Event"]..":"..eventlist[1]["Counter"]
    end
    for i=2,#eventlist do
        if eventlist[i] ~= nil and eventlist[i]["Event"] ~= nil and eventlist[i]["Counter"] ~= nil then
            str = str .. ","..eventlist[i]["Event"]..":"..eventlist[i]["Counter"]
        end
    end
    return str
end

local function splitUncoreEvents(groupdata)
    local core = {}
    local socket = {}
    local numa = {}
    local llc = {}
    local cpuinfo = likwid.getCpuInfo()

    for i, e in pairs(groupdata["Events"]) do
        if  not e["Counter"]:match("FIXC%d") and
            not e["Counter"]:match("^PMC%d") and
            not e["Counter"]:match("TMP%d") then
            local event = e["Event"]..":"..e["Counter"]
            if cpuinfo["architecture"] == "x86_64" then
                if cpuinfo["isIntel"] == 1 then
                    table.insert(socket, event)
                else -- AMD
                    if e["Counter"]:match("^CPMC%d") then
                        table.insert(llc, event)
                    elseif e["Counter"]:match("^UPMC%d") then
                        table.insert(socket, event)
                    elseif e["Counter"]:match("^DFC%d") then
                        table.insert(numa, event)
                    end
                end
            elseif cpuinfo["architecture"] == "armv8" then
                table.insert(socket, event)
            elseif cpuinfo["architecture"] == "armv7" then
                table.insert(socket, event)
            elseif cpuinfo["architecture"]:match("^power%d") then
                table.insert(socket, event)
            end
        else
            local event = e["Event"]..":"..e["Counter"]
            table.insert(core, event)
        end
    end
    sevents = ""
    nevents = ""
    levents = ""
    if #socket > 0 then
        sevents = table.concat(socket,",")
    end
    if #numa > 0 then
        nevents = table.concat(numa, ",")
    end
    if #llc > 0 then
        levents = table.concat(llc, ",")
    end
    return table.concat(core, ","), sevents, nevents, levents
end

local function inList(value, list)
    for _,l in pairs(list) do
        if value == l then
            return true
        end
    end
    return false
end

local function uniqueList(list)
    local newl = {}
    for _,l in pairs(list) do
        found = false
        for _,k in pairs(newl) do
            if l == k then
                found = true
            end
        end
        if not found then
            table.insert(newl, l)
        end
    end
    return newl
end

local function setPerfStrings(perflist, cpuexprs)
    local suncore = false
    local perfexprs = {}
    local grouplist = {}

    local affinity = likwid.getAffinityInfo()
    local socketList = {}
    local socketListFlags = {}
    local numaList = {}
    local numaListFlags = {}
    local llcList = {}
    local llcListFlags = {}
    for i, d in pairs(affinity["domains"]) do
        if d["tag"]:match("S%d+") then
            local tmpList = {}
            for j,cpu in pairs(d["processorList"]) do
                table.insert(tmpList, cpu)
            end
            table.insert(socketList, tmpList)
            table.insert(socketListFlags, 1)
        end
        if d["tag"]:match("M%d+") then
            local tmpList = {}
            for j,cpu in pairs(d["processorList"]) do
                table.insert(tmpList, cpu)
            end
            table.insert(numaList, tmpList)
            table.insert(numaListFlags, 1)
        end
        if d["tag"]:match("C%d+") then
            local tmpList = {}
            for j,cpu in pairs(d["processorList"]) do
                table.insert(tmpList, cpu)
            end
            table.insert(llcList, tmpList)
            table.insert(llcListFlags, 1)
        end
    end

    for k, perfStr in pairs(perflist) do
        local gdata = nil
        gdata = likwid.get_groupdata(perfStr)
        if gdata == nil or gdata["EventString"]:len() == 0 then
            print_stderr("Cannot get data for group "..perfStr..". Skipping...")
        else
            table.insert(grouplist, gdata)
            if perfexprs[k] == nil then
                perfexprs[k] = {}
            end

            local coreevents = ""
            local socketevents = ""
            local numaevents = ""
            local llcevents = ""
            coreevents, socketevents, numaevents, llcevents = splitUncoreEvents(gdata)

            local tmpSocketFlags = {}
            for _,e in pairs(socketListFlags) do
                table.insert(tmpSocketFlags, e)
            end
            local tmpNumaFlags = {}
            for _,e in pairs(numaListFlags) do
                table.insert(tmpNumaFlags, e)
            end
            local tmpCacheFlags = {}
            for _,e in pairs(llcListFlags) do
                table.insert(tmpCacheFlags, e)
            end

            for i,cpuexpr in pairs(cpuexprs) do
                local slist = {}
                for j, cpu in pairs(cpuexpr) do
                    for l, socklist in pairs(socketList) do
                        if inList(tonumber(cpu), socklist) then
                            table.insert(slist, l)
                        end
                    end
                end
                slist = uniqueList(slist)
                local mlist = {}
                for j, cpu in pairs(cpuexpr) do
                    for l, numalist in pairs(numaList) do
                        if inList(tonumber(cpu), numalist) then
                            table.insert(mlist, l)
                        end
                    end
                end
                mlist = uniqueList(mlist)
                local clist = {}
                for j, cpu in pairs(cpuexpr) do
                    for l, llclist in pairs(llcList) do
                        if inList(tonumber(cpu), llclist) then
                            table.insert(clist, l)
                        end
                    end
                end
                clist = uniqueList(clist)
                local suncore = false
                local muncore = false
                local cuncore = false
                for _, s in pairs(slist) do
                    if tmpSocketFlags[s] == 1 then
                        tmpSocketFlags[s] = 0
                        suncore = true
                    end
                end
                for _, s in pairs(mlist) do
                    if tmpNumaFlags[s] == 1 then
                        tmpNumaFlags[s] = 0
                        muncore = true
                    end
                end
                for _, s in pairs(clist) do
                    if tmpCacheFlags[s] == 1 then
                        tmpCacheFlags[s] = 0
                        cuncore = true
                    end
                end
                if perfexprs[k][i] == nil then
                    local elist = {}
                    if coreevents:len() > 0 then
                        table.insert(elist, coreevents)
                    end
                    if cuncore and llcevents:len() > 0 then
                        table.insert(elist, llcevents)
                    end
                    if muncore and numaevents:len() > 0 then
                        table.insert(elist, numaevents)
                    end
                    if suncore and socketevents:len() > 0 then
                        table.insert(elist, socketevents)
                    end
                    if #elist > 0 then
                        perfexprs[k][i] = table.concat(elist, ",")
                    else
                        perfexprs[k][i] = ""
                    end
                end
            end

            if debug then
                for i, expr in pairs(perfexprs[k]) do
                    if expr:len() > 0 then
                        print_stdout(string.format("DEBUG: Process %d measures with event set: %s", i-1, expr))
                    else
                        print_stdout(string.format("DEBUG: Process %d measures with event set: No events", i-1))
                    end
                end
            end
        end
    end
    if #grouplist == 0 then
        print_stderr("No group can be configured for measurements, exiting.")
        os.exit(1)
    end
    return perfexprs, grouplist
end

local function checkLikwid()
    local f = io.popen("which likwid-pin 2>/dev/null", "r")
    if f ~= nil then
        local s = f:read("*line")
        if s ~= nil and s ~= LIKWID_PIN then
            LIKWID_PIN = s
        end
        f:close()
    end
    f = io.popen("which likwid-perfctr 2>/dev/null", "r")
    if f ~= nil then
        local s = f:read("*line")
        if s ~= nil and s ~= LIKWID_PERFCTR then
            LIKWID_PERFCTR = s
        end
        f:close()
    end
end

local function writeWrapperScript(scriptname, execStr, hosts, envsettings, outputname)
    if scriptname == nil or scriptname == "" then
        return
    end
    local oversubscripted = {}
    local commands = {}
    local only_pinned_processes = {}
    tmphosts = {}
    for i, host in pairs(hosts) do
        if tmphosts[host["hostname"]] ~= nil then
            tmphosts[host["hostname"]] = tmphosts[host["hostname"]] + host["slots"]
        else
            tmphosts[host["hostname"]] = host["slots"]
        end
    end

    if mpitype == "openmpi" then
        glsize_var = "$OMPI_COMM_WORLD_SIZE"
        glrank_var = "${OMPI_COMM_WORLD_RANK:-$(($GLOBALSIZE * 2))}"
        losize_var = "$OMPI_COMM_WORLD_LOCAL_SIZE"
    elseif mpitype == "intelmpi" then
        glrank_var = "${PMI_RANK:-$(($GLOBALSIZE * 2))}"
        glsize_var = tostring(math.tointeger(np))
        losize_var = tostring(math.tointeger(ppn))
    elseif mpitype == "mvapich2" then
        glrank_var = "${PMI_RANK:-$(($GLOBALSIZE * 2))}"
        glsize_var = tostring(math.tointeger(np))
        losize_var = tostring(math.tointeger(ppn))
    elseif mpitype == "slurm" then
        glrank_var = "${SLURM_PROCID:-$(($GLOBALSIZE * 2))}"
        glsize_var = tostring(math.tointeger(np))
        losize_var = string.format("${SLURM_NTASKS_PER_NODE:-%d}", math.tointeger(ppn))
    else
        print_stderr("Invalid MPI vendor "..mpitype)
        return
    end

    local taillength = np % ppn
    if taillength ~= 0 then
        local full = tostring(math.tointeger(np -taillength))
        table.insert(oversubscripted, "if [ $GLOBALRANK >= "..tostring(math.tointeger(full)).." ]; then\n")
        table.insert(oversubscripted, "\tLOCALRANK=$($GLOBALRANK - "..tostring(math.tointeger(full))..")\n")
        table.insert(oversubscripted, "fi\n")
    end

    local f = io.open(scriptname, "w")
    if f == nil then
        print_stderr("ERROR: Cannot open hostfile "..scriptname)
        os.exit(1)
    end

    if outputname:sub(1,1) ~= "/" then
        outputname = os.getenv("PWD").."/"..outputname
    end

    for i=1,#cpuexprs do
        local cmd = {}
        local cpuexpr_opt = "-c"
        local eventsets_valid = 0
        local use_perfctr = false
        for j, expr in pairs(perfexprs) do
            if expr[i]:len() > 0 then eventsets_valid = eventsets_valid + 1 end
        end
        if #perf > 0 and eventsets_valid == #perfexprs then
            use_perfctr = true
        end
        if use_perfctr then
            table.insert(cmd, LIKWID_PERFCTR)
            if use_marker then
                table.insert(cmd,"-m")
            end
            cpuexpr_opt = "-C"
        else
            table.insert(cmd, LIKWID_PIN)
            table.insert(cmd,"-q")
            if #perf > 0 then
                table.insert(only_pinned_processes, i)
            end
        end
        if force and use_perfctr then
            table.insert(cmd,"-f")
        end
        if likwiddebug then
            table.insert(cmd,"-V 3")
        end
        table.insert(cmd, skipStr)
        table.insert(cmd, cpuexpr_opt)
        table.insert(cmd, table.concat(cpuexprs[i], ","))
        if use_perfctr then
            for j, expr in pairs(perfexprs) do
                table.insert(cmd, "-g")
                table.insert(cmd, expr[i])
            end
            table.insert(cmd, "-o")
            table.insert(cmd, outputname)
        end
        table.insert(cmd, execStr)
        commands[i] = table.concat(cmd, " ")
    end

    f:write("#!/bin/bash -l\n")
    f:write("GLOBALSIZE="..glsize_var.."\n")
    f:write("GLOBALRANK="..glrank_var.."\n")
    if os.getenv("OMP_NUM_THREADS") == nil then
        f:write(string.format("export OMP_NUM_THREADS=%d\n", tpp))
    else
        f:write(string.format("export OMP_NUM_THREADS=%s\n", os.getenv("OMP_NUM_THREADS")))
    end
    if mpitype == "intelmpi" then
        f:write("export I_MPI_PIN=disable\n")
    end
    for i, e in pairs(envsettings) do
        if debug then
            print_stdout(string.format("DEBUG: Environment variable %s", e))
        end
        f:write(string.format("export %s\n", e))
    end
    f:write("LOCALSIZE="..losize_var.."\n\n")

    if mpitype == "openmpi" then
        f:write("LOCALRANK=$OMPI_COMM_WORLD_LOCAL_RANK\n\n")
    elseif mpitype  == "slurm" then
        f:write("LOCALRANK=${MPI_LOCALRANKID:-$SLURM_LOCALID}\n\n")
    else
        local full = tostring(math.tointeger(np - (np % ppn)))
        f:write("if [ \"$GLOBALRANK\" -lt "..tostring(math.tointeger(full)).." ]; then\n")
        f:write("\tLOCALRANK=$(($GLOBALRANK % $LOCALSIZE))\n")
        f:write("else\n")
        f:write("\tLOCALRANK=$(($GLOBALRANK - ("..tostring(math.tointeger(full)).." - 1)))\n")
        f:write("fi\n\n")
    end

    if #perf > 0 then
        f:write("which `basename "..LIKWID_PERFCTR.."` 1>/dev/null 2>&1\n")
    else
        f:write("which `basename "..LIKWID_PIN.."` 1>/dev/null 2>&1\n")
    end
    f:write("if [ $? -eq 1 ]; then\n")
    f:write("\tmodule load likwid 1>/dev/null 2>&1\n")
    f:write("fi\n\n")

    f:write("if [ \"$LOCALRANK\" -eq 0 ]; then\n")
    if debug then
        print_stdout(string.format("EXEC (Rank 0): %s",commands[1]))
    end
    f:write("\t"..commands[1].."\n")

    for i=2,#commands do
        f:write("elif [ \"$LOCALRANK\" -eq "..tostring(i-1).." ]; then\n")
        if debug then
            print_stdout(string.format("EXEC (Rank %d): %s", i-1,commands[i]))
        end
        f:write("\t"..commands[i].."\n")
    end
    f:write("else\n")
    f:write("\techo \"Unknown local rank $LOCALRANK\"\n")
    f:write("fi\n")
    f:close()
    os.execute("chmod +x "..scriptname)
    return only_pinned_processes
end


local function listdir(dir, infilepart)
    local outlist = {}
    local p = io.popen("find "..dir.." -maxdepth 1 -type f -name \"*"..infilepart.."*\"")
    for file in p:lines() do
        table.insert(outlist, file)
    end
    p:close()
    if #outlist > 0 then
        table.sort(outlist)
    end
    return outlist
end


local function parseOutputFile(filename)
    local rank = 0
    local host = nil
    local cpulist = {}
    local eventlist = {}
    local counterlist = {}
    local idx = 1
    local gidx = 0
    local results = {}
    local f = io.open(filename, "r")
    if f == nil then
        print_stderr("ERROR: Cannot open output file "..filename)
        os.exit(1)
    end
    rank, host = filename:match("output_%d+_(%d+)_([^%s]+).csv")

    local t = f:read("*all")
    f:close()
    if t:len() == 0 then
        print_stderr("Error Output file "..filename.." is empty")
        os.exit(1)
    end
    for i, line in pairs(likwid.stringsplit(t, "\n")) do
        if (not line:match("^-")) and
           (not line:match("^CPU type:")) and
           (not line:match("^CPU name:")) and
           (not line:match("^TABLE")) and
           (not line:match("^STRUCT")) and
           (not line:match("^%s*$")) and
           (not line:match("STAT")) then
            if line:match("^Event") and not line:match("Sum,Min,Max,Avg") then
                linelist = likwid.stringsplit(line,",")
                table.remove(linelist,1)
                table.remove(linelist,1)
                for _, cpustr in pairs(linelist) do
                    local test = tonumber(cpustr:match("HWThread (%d+)"))
                    if test ~= nil then
                        for _cpu in pairs(cpulist) do
                            if tonumber(cpu) == test then test = -1 end
                        end
                        if test >= 0 then
                            table.insert(cpulist, test)
                        end
                    end
                end
                gidx = gidx + 1
                idx = 1
                if results[gidx] == nil then
                    results[gidx] = {}
                    eventlist[gidx] = {}
                    counterlist[gidx] = {}
                    results[gidx]["time"] = {}
                end
            elseif not line:match("^CPU clock:") and not line:match("Sum,Min,Max,Avg") then
                linelist = likwid.stringsplit(line,",")
                event = linelist[1]
                counter = linelist[2]
                table.remove(linelist,1)
                table.remove(linelist,1)
                for j=#linelist,1,-1 do
                    if linelist[j] == "" then
                        table.remove(linelist, j)
                    end
                end
                if results[gidx][counter] == nil then
                    results[gidx][counter] = {}
                end
                for j, value in pairs(linelist) do
                    if event:match("[Rr]untime") then
                        results[gidx]["time"][cpulist[j]] = tonumber(value)
                    else
                        results[gidx][counter][cpulist[j]] = tonumber(value)
                    end
                end
                if not event:match("[Rr]untime") then
                    table.insert(eventlist[gidx], idx, event)
                    table.insert(counterlist[gidx], idx, counter)
                    idx = idx + 1
                end
            elseif line:match("^CPU clock:") then
                results["clock"] = line:match("^CPU clock:,([%d.]+)")
                results["clock"] = tonumber(results["clock"])*1.E09
            end
        end
    end
    return host, tonumber(rank), results, cpulist
end

local function parseMarkerOutputFile(filename)
    local rank = 0
    local host = nil
    local cpulist = {}
    local eventlist = {}
    local counterlist = {}
    local regionlist = {}
    local idx = 1
    local results = {}
    local f = io.open(filename, "r")
    if f == nil then
        print_stderr("ERROR: Cannot open output file "..filename)
        os.exit(1)
    end
    rank, host = filename:match("output_%d+_(%d+)_([^%s]+).csv")
    local t = f:read("*all")
    f:close()
    local parse_reg_info = false
    local parse_reg_output = false
    local current_region = nil
    local gidx = 0
    local gname = ""
    local clock = 0

    for i, line in pairs(likwid.stringsplit(t, "\n")) do
        if (not line:match("^-")) and
           (not line:match("^CPU type:")) and
           (not line:match("^CPU name:")) and
           (not line:match("STAT")) then

            if line:match("^STRUCT,Info") and not parse_reg_info then
                parse_reg_info = true
            elseif line:match("^Event") and not line:match("Sum,Min,Max,Avg") then
                parse_reg_info = false
                parse_reg_output = true
                idx = 1
            elseif line:match("^Event") and line:match("Sum,Min,Max,Avg") then
                parse_reg_output = false
            elseif line:match("^CPU clock:,") then
                clock = line:match("^CPU clock:,([%d.]+)")
                clock = tonumber(clock)*1.E09
            elseif parse_reg_info and line:match("TABLE,Region ([^%s]+),Group (%d+) Raw,([^%s]+),") then
                current_region, gidx, gname  = line:match("TABLE,Region ([^%s]+),Group (%d+) Raw,([^%s]+),")
                gidx = tonumber(gidx)
                if results[current_region] == nil then
                    results[current_region] = {}
                end
                if results[current_region][gidx] == nil then
                    results[current_region][gidx] = {}
                    results[current_region][gidx]["name"] = gname
                    results[current_region][gidx]["time"] = {}
                    results[current_region][gidx]["calls"] = {}
                end
                table.insert(regionlist, current_region)
            elseif parse_reg_info and line:match("^Region Info") then
                linelist = likwid.stringsplit(line,",")
                table.remove(linelist,1)
                for _, cpustr in pairs(linelist) do
                    if cpustr:match("HWThread %d+") then
                        local test = tonumber(cpustr:match("HWThread (%d+)"))
                        if test ~= nil then
                            for _,cpu in pairs(cpulist) do
                                if test == cpu then test = -1 end
                            end
                            if test >= 0 then
                                table.insert(cpulist, test)
                            end
                        end
                    end
                end
            elseif parse_reg_info and line:match("^RDTSC") then
                linelist = likwid.stringsplit(line,",")
                table.remove(linelist,1)
                for i, time in pairs(linelist) do
                    if time ~= "" then
                        results[current_region][gidx]["time"][cpulist[i]] = tonumber(time)
                    end
                end
            elseif parse_reg_info and line:match("^call count") then
                linelist = likwid.stringsplit(line,",")
                table.remove(linelist,1)
                for j, calls in pairs(linelist) do
                    if calls:match("%d+") then
                        if calls ~= "" then
                            results[current_region][gidx]["calls"][cpulist[j]] = tonumber(calls)
                        end
                    end
                end
            elseif parse_reg_output and not line:match("^%s*$") then
                linelist = likwid.stringsplit(line,",")
                if linelist[2] ~= "TSC" then
                    event = linelist[1]
                    counter = linelist[2]
                    table.remove(linelist,1)
                    table.remove(linelist,1)
                    for j=#linelist,1,-1 do
                        if linelist[j] == "" then
                            table.remove(linelist, j)
                        end
                    end
                    if results[current_region][gidx][counter] == nil then
                        results[current_region][gidx][counter] = {}
                    end
                    for j, value in pairs(linelist) do
                        v = tonumber(value)
                        if v then
                            results[current_region][gidx][counter][cpulist[j]] = v
                        else
                            results[current_region][gidx][counter][cpulist[j]] = 0/0
                        end
                    end
                    idx = idx + 1
                end
            end
        end
    end
    for region, data in pairs(results) do
        results[region]["clock"] = clock
    end

    return host, tonumber(rank), results, cpulist, regionlist
end


function percentile_table(inputtable, skip_cols, skip_lines)
    local function percentile(sorted_valuelist, k)
        index = tonumber(k)/100.0 * #sorted_valuelist
        if index - math.floor(index) >= 0.5 then
            index = math.ceil(index)
        else
            index = math.floor(index)
        end
        if index == 0 then
            index = 1
        end
        return tonumber(sorted_valuelist[index])
    end
    local outputtable = {}
    local ncols = #inputtable
    if ncols == 0 then
        return outputtable
    end
    local nlines = #inputtable[1]
    if nlines == 0 then
        return outputtable
    end
    perc25 = {"%ile 25"}
    perc50 = {"%ile 50"}
    perc75 = {"%ile 75"}
    for i=skip_lines+1,nlines do
        perc25[i-skip_lines+1] = 0
        perc50[i-skip_lines+1] = 0
        perc75[i-skip_lines+1] = 0
    end
    for l=skip_lines+1,nlines do
        valuelist = {}
        for c=skip_cols+1, ncols do
            table.insert(valuelist, inputtable[c][l])
        end
        table.sort(valuelist)
        perc25[l-skip_lines+1] = likwid.num2str(percentile(valuelist, 25))
        perc50[l-skip_lines+1] = likwid.num2str(percentile(valuelist, 50))
        perc75[l-skip_lines+1] = likwid.num2str(percentile(valuelist, 75))
    end
    table.insert(outputtable, perc25)
    table.insert(outputtable, perc50)
    table.insert(outputtable, perc75)
    return outputtable
end

function printMpiOutput(group_list, all_results, regionname)
    region = regionname or nil
    if #group_list == 0 or likwid.tablelength(all_results) == 0 then
        return
    end
    for gidx, gdata in pairs(group_list) do
        local firsttab = {}
        local firsttab_combined = {}
        local secondtab = {}
        local secondtab_combined = {}
        local total_threads = 0
        local all_counters = {}
        local groupName = gdata["GroupString"]

        for rank = 0, #all_results do
            total_threads = total_threads + #all_results[rank]["cpus"]
        end

        desc = {"Event"}
        if total_threads == 1 or not gdata["Metrics"] then
            table.insert(desc, "Runtime (RDTSC) [s]")
        end
        if all_results[0]["results"][1]["calls"] then
            table.insert(desc, "Region calls")
        end
        for i=1,#gdata["Events"] do
            table.insert(desc, gdata["Events"][i]["Event"])
        end
        table.insert(firsttab, desc)

        desc = {"Counter"}
        if total_threads == 1 or not gdata["Metrics"] then
            table.insert(desc, "TSC")
        end
        if all_results[0]["results"][1]["calls"] then
            table.insert(desc, "CTR")
        end
        for i=1,#gdata["Events"] do
            table.insert(desc, gdata["Events"][i]["Counter"])
        end
        table.insert(firsttab, desc)

        for rank = 0, #all_results do
            for i, cpu in pairs(all_results[rank]["cpus"]) do
                column = {all_results[rank]["hostname"]..":"..tostring(rank)..":"..tostring(cpu)}
                if total_threads == 1 or not gdata["Metrics"] then
                    table.insert(column, all_results[rank]["results"][gidx]["time"][cpu])
                end
                if all_results[0]["results"][1]["calls"] then
                    table.insert(column, all_results[rank]["results"][gidx]["calls"][cpu])
                end
                for j=1,#gdata["Events"] do
                    local value = "0"
                    cname = gdata["Events"][j]["Counter"]
                    if all_results[rank]["results"][gidx][cname] and
                       all_results[rank]["results"][gidx][cname][cpu] then
                        value = likwid.num2str(all_results[rank]["results"][gidx][cname][cpu])
                    end
                    table.insert(column, value)
                end
                table.insert(firsttab, column)
            end
        end

        if total_threads > 1 or print_stats then
            firsttab_combined = likwid.tableToMinMaxAvgSum(firsttab, 2, 1)
        end
        if gdata["Metrics"] then
            secondtab[1] = {"Metric"}
            for j=1,#gdata["Metrics"] do
                table.insert(secondtab[1], gdata["Metrics"][j]["description"])
            end

            for rank = 0, #all_results do
                for i, cpu in pairs(all_results[rank]["cpus"]) do
                    local counterlist = {}
                    for j=1,#gdata["Events"] do
                        local counter = gdata["Events"][j]["Counter"]
                        counterlist[counter] = 0
                        if all_results[rank]["results"][gidx][counter] ~= nil and
                           all_results[rank]["results"][gidx][counter][cpu] ~= nil then
                            counterlist[counter] = all_results[rank]["results"][gidx][counter][cpu]
                        end
                    end
                    counterlist["time"] = all_results[rank]["results"][gidx]["time"][cpu]
                    counterlist["inverseClock"] = 1.0/all_results[rank]["results"]["clock"]
                    tmpList = {all_results[rank]["hostname"]..":"..tostring(rank)..":"..tostring(cpu)}
                    for j=1,#groupdata["Metrics"] do
                        local f = gdata["Metrics"][j]["formula"]
                        local tmp = likwid.num2str(likwid.calculate_metric(f, counterlist))
                        table.insert(tmpList, tmp)
                    end
                    table.insert(secondtab,tmpList)
                end
            end

            if total_threads > 1 or print_stats then
                secondtab_combined = likwid.tableToMinMaxAvgSum(secondtab, 1, 1)
                local tmp = percentile_table(secondtab, 1, 1)
                for i, col in pairs(tmp) do
                    table.insert(secondtab_combined, col)
                end
            end
        end
        if use_csv then
            local maxLineFields = #firsttab
            if #firsttab_combined > maxLineFields then maxLineFields = #firsttab_combined end
            if gdata["Metrics"] then
                if #secondtab > maxLineFields then maxLineFields = #secondtab end
                if #secondtab_combined > maxLineFields then maxLineFields = #secondtab_combined end
            end
            if region then
                print_stdout(string.format("TABLE,Region %s,Group %d Raw,%s,%d%s",tostring(region), gidx,groupName,#firsttab[1]-1,string.rep(",",maxLineFields-5)))
            else
                print_stdout(string.format("TABLE,Group %d Raw,%s,%d%s",gidx,groupName,#firsttab[1]-1,string.rep(",",maxLineFields-4)))
            end
            --print_stdout("Group,"..tostring(gidx) .. string.rep(",", maxLineFields  - 2))
            likwid.printcsv(firsttab, maxLineFields)
            if total_threads > 1 or print_stats then
                if region == nil then
                    print(string.format("TABLE,Group %d Raw STAT,%s,%d%s",gidx,groupName,#firsttab_combined[1]-1,string.rep(",",maxLineFields-4)))
                else
                    print(string.format("TABLE,Region %s,Group %d Raw STAT,%s,%d%s",tostring(region), gidx,groupName,#firsttab_combined[1]-1,string.rep(",",maxLineFields-5)))
                end
                likwid.printcsv(firsttab_combined, maxLineFields)
            end
            if gdata["Metrics"] then
                if region == nil then
                    print(string.format("TABLE,Group %d Metric,%s,%d%s",gidx,groupName,#secondtab[1]-1,string.rep(",",maxLineFields-4)))
                else
                    print(string.format("TABLE,Region %s,Group %d Metric,%s,%d%s",tostring(region),gidx,groupName,#secondtab[1]-1,string.rep(",",maxLineFields-5)))
                end
                likwid.printcsv(secondtab, maxLineFields)
                if total_threads > 1 or print_stats then
                    if region == nil then
                        print(string.format("TABLE,Group %d Metric STAT,%s,%d%s",gidx,groupName,#secondtab_combined[1]-1,string.rep(",",maxLineFields-4)))
                    else
                        print(string.format("TABLE,Region %s,Group %d Metric STAT,%s,%d%s",tostring(region),gidx,groupName,#secondtab_combined[1]-1,string.rep(",",maxLineFields-5)))
                    end
                    likwid.printcsv(secondtab_combined, maxLineFields)
                end
            end
        else
            if region then
                print_stdout("Region: "..tostring(region))
            end
            print_stdout("Group: "..tostring(gidx))
            likwid.printtable(firsttab)
            if total_threads > 1 or print_stats then likwid.printtable(firsttab_combined) end
            if gdata["Metrics"] then
                likwid.printtable(secondtab)
                if total_threads > 1 or print_stats then likwid.printtable(secondtab_combined) end
            end
        end
    end
end

function cpuCount()
    cputopo = likwid.getCpuTopology()
    local cpus = cputopo["activeHWThreads"]
    return cpus
end

if #arg == 0 then
    usage()
    os.exit(0)
end

local cmd_options = {"h","help", -- default options for help message
                     "v","version", -- default options for version message
                     "d", "debug", -- activate debugging output
                     "n:","np:", -- default options for number of MPI processes
                     "t:","tpp:", -- default options for number of threads per process
                     "mpi:","omp:", -- options to overwrite detection
                     "s:","skip:", -- options to specify custom skip mask for threads
                     "g:","group:", -- options to set group for performance measurements using likwid
                     "m","marker", -- options to activate MarkerAPI
                     "e:", "env:", -- options to forward environment variables
                     "ld",         -- option to activate debugging in likwid-perfctr
                     "dist:",      -- option to specifiy distance between two MPI processes
                     "o:","output:", -- option to specifiy an output file
                     "mpiopts:", -- option to specifiy MPI options forwarded to the underlying MPI
                     "nperdomain:","pin:","hostfile:","O","f", "stats"} -- other options

for opt,arg in likwid.getopt(arg,  cmd_options) do
    if (type(arg) == "string") and opt ~= "mpiopts" then
        local s,e = arg:find("-")
        if s == 1 then
            print_stderr(string.format("ERROR: Argument %s to option -%s starts with invalid character -.", arg, opt))
            print_stderr("ERROR: Did you forget an argument to an option?")
            os.exit(1)
        end
    end

    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version"then
        version()
        os.exit(0)
    elseif opt == "d" or opt == "debug" then
        debug = true
    elseif opt == "m" or opt == "marker" then
        use_marker = true
    elseif opt == "O" then
        use_csv = true
    elseif opt == "f" then
        force = true
    elseif opt == "stats" then
        print_stats = true
    elseif opt == "n" or opt == "np" then
        np = tonumber(arg)
        if np == nil then
            print_stderr("Argument for -n/-np must be a number")
            os.exit(1)
        end
    elseif opt == "t" or opt == "tpp" then
        if arg:match("%d+:%a+") then
            t, order = arg:match("(%d+):(%a+)")
            tpp = tonumber(t)
            if tpp == nil then
                print_stderr("Argument for -t/-tpp must be a number")
                os.exit(1)
            end
            if tpp == 0 then
                print_stderr("Cannot run with 0 threads, at least 1 is required, sanitizing tpp to 1")
                tpp = 1
            end
            local valid_order = false
            for _, o in pairs(tpp_orderings) do
                if o == order then
                    valid_order = true
                    break
                end
            end
            if valid_order then
                tpp_ordering = order
            end
            print_stdout(tpp, tpp_ordering)
        elseif arg:match("%d+") then
            tpp = tonumber(arg)
            if tpp == nil then
                print_stderr("Argument for -t/-tpp must be a number")
                os.exit(1)
            end
            if tpp == 0 then
                print_stderr("Cannot run with 0 threads, at least 1 is required, sanitizing tpp to 1")
                tpp = 1
            end
        else
            print_stderr("Argument for -t/-tpp must be a number")
            os.exit(1)
        end
    elseif opt == "dist" then
        if arg:match("%d+:%a+") then
            t, order = arg:match("(%d+):(%a+)")
            local valid_order = false
            for _, o in pairs(tpp_orderings) do
                if o == order then
                    valid_order = true
                    break
                end
            end
            if valid_order then
                tpp_ordering = order
            end
            dist = tonumber(t)
            if dist == nil then
                print_stderr("Argument for -dist must be a number or number:ordering")
                os.exit(1)
            end
            if dist == 0 then
                print_stderr("Cannot run with distance 0, at least 1 is required, sanitizing dist to 1")
                dist = 1
            end
        elseif arg:match("%d+") then
            dist = tonumber(arg)
            if dist == nil then
                print_stderr("Argument for -dist must be a number or number:ordering")
                os.exit(1)
            end
            if dist == 0 then
                print_stderr("Cannot run with distance 0, at least 1 is required, sanitizing dist to 1")
                dist = 1
            end
        else
            print_stderr("Argument for -dist must be a number or number:ordering")
            os.exit(1)
        end
    elseif opt == "nperdomain" then
        local domain, count, threads = arg:match("([NSCM]):(%d+)[:]*(%d*)")
        if domain == nil or count == nil then
            print_stderr("Invalid option to -nperdomain")
            os.exit(1)
        end
        nperdomain = string.format("%s:%s", domain, count)
        if threads ~= nil then
            nperdomain = nperdomain .. ":" ..threads
        end
    elseif opt == "e" or opt == "env" then
        name, val = arg:match("([%a%d_]+)[=]*([%a%d_\"\"']*)")
        if name == nil and (val == nil or tostring(val):len() == 0) then
            print_stderr("Invalid argument for -e/-env, must be varname=varvalue")
        else
            if (val == nil or tostring(val):len() == 0) then
                val = os.getenv(name) or ''
            end
            if name:len() > 0 and val:len() > 0 then
                table.insert(envsettings, string.format("%s=%s", name, val))
            end
        end
    elseif opt == "hostfile" then
        hostfile = arg
    elseif opt == "pin" then
        cpuexprs = likwid.stringsplit(arg, "_")
    elseif opt == "g" or opt == "group" then
        table.insert(perf, arg)
    elseif opt == "mpi" then
        mpitype = arg
    elseif opt == "omp" then
        omptype = arg
    elseif opt == "ld" then
        likwiddebug = true
    elseif opt == "o" or opt == "output" then
        outfile = arg
        print_stderr("WARN: The output file option is currently ignored. Will be available in upcoming releases")
    elseif opt == "s" or opt == "skip" then
        skipStr = "-s "..arg
    elseif opt == "mpiopts" then
        mpiopts = tostring(arg)
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    elseif opt == "-" then
        break
    end
end


if np == 0 and nperdomain == nil and #cpuexprs == 0 then
    print_stderr("ERROR: No option -n/-np, -nperdomain or -pin")
    os.exit(1)
end

if use_marker and #perf == 0 then
    print_stderr("ERROR: You selected the MarkerAPI feature but didn't set any events on the commandline")
    os.exit(1)
end

for i,x in pairs(arg) do
    if i > 0 then
        table.insert(executable, abspath(x) or x)
    end
end

if #executable == 0 then
    print_stderr("ERROR: No executable given on commandline")
    os.exit(1)
end

if debug then
    print_stdout("DEBUG: Executable given on commandline: "..table.concat(executable, " "))
end
local gotExecutable = false
for i,x in pairs(executable) do
    if likwid.access(x, "x") == 0 then
        gotExecutable = true
        break
    end
end
if not gotExecutable then
    print_stderr("ERROR: Cannot find an executable on commandline")
    print_stderr(table.concat(executable, " "))
    os.exit(1)
end

if mpiopts and mpiopts:len() > 0 and debug then
    print_stdout("DEBUG: MPI options given on commandline: "..mpiopts)
end

if mpitype == nil then
    mpitype = getMpiType()
    if mpitype == nil then
        print_stderr("ERROR: Cannot find MPI implementation")
        os.exit(1)
    end
    if debug then
        print_stdout("DEBUG: Using MPI implementation "..mpitype)
    end
end
if mpitype ~= "intelmpi" and mpitype ~= "mvapich2" and mpitype ~= "openmpi" and mpitype ~= "slurm" then
    print_stderr("ERROR: Cannot determine current MPI implementation. likwid-mpirun checks for openmpi, intelmpi and mvapich2 or if running in a SLURM environment")
    os.exit(1)
end

getMpiExec(mpitype)
if (mpiexecutable == nil) then
    print_stderr(string.format("Cannot find executable for determined MPI implementation %s", mpitype))
    os.exit(1)
end

if omptype == nil then
    omptype = getOmpType()
    if debug and omptype ~= nil then
        print_stdout("DEBUG: Using OpenMP implementation "..omptype)
    end
end
if omptype == nil then
    print_stderr("WARN: Cannot extract OpenMP vendor from executable or commandline, assuming no OpenMP")
end

if not hostfile then
    if os.getenv("PBS_NODEFILE") ~= nil then
        hostfile = os.getenv("PBS_NODEFILE")
        hosts = readHostfilePBS(hostfile)
    elseif os.getenv("LOADL_HOSTFILE") ~= nil then
        hostfile = os.getenv("LOADL_HOSTFILE")
        hosts = readHostfilePBS(hostfile)
    elseif os.getenv("SLURM_NODELIST") ~= nil then
        hostlist = os.getenv("SLURM_NODELIST")
        hosts = readHostfileSlurm(hostlist)
    else
        local cpus = cpuCount()
        table.insert(hosts, {hostname='localhost', slots=cpus, maxslots=cpus})
        if slurm_involved then
            print_stderr("ERROR: Cannot run on localhost with SLURM involved")
            os.exit(1)
        end
    end
else
    hosts = readHostfile(hostfile)
end

local givenNrNodes = getNumberOfNodes(hosts)



if #perf > 0 then
    local sum_maxslots = 0
    local topo = likwid.getCpuTopology()
    if debug then
        print_stdout("DEBUG: Switch to perfctr mode, there are "..tostring(#perf).." eventsets given on the commandline")
    end
    for i, host in pairs(hosts) do
        if debug then
            local str = string.format("DEBUG: Working on host %s with %d slots", host["hostname"], host["slots"])
            if host["maxslots"] ~= nil then
                str = str .. string.format(" and %d slots maximally", host["maxslots"])
            end
            print_stdout(str)
        end
        if host["maxslots"] ~= nil then
            sum_maxslots = sum_maxslots + host["maxslots"]
        elseif host["slots"] ~= nil then
            sum_maxslots = sum_maxslots + host["slots"]
        else
            sum_maxslots = sum_maxslots + topo["numHWThreads"]
            host["slots"] = topo["numHWThreads"]
        end
    end
    if np > sum_maxslots then
        print_stderr("ERROR: Processes requested exceeds maximally available slots of given hosts. Maximal processes: "..sum_maxslots)
        os.exit(1)
    end
end
if #perf == 0 and print_stats then
    print_stderr("WARN: Printing statistics only available when measuring counters (-g option)")
    print_stats = false
end

if #cpuexprs > 0 then
    cpuexprs = calculatePinExpr(cpuexprs)
    if debug then
        str = "["
        for i, expr in pairs(cpuexprs) do
            str = str .. "["..table.concat(expr,",").."], "
        end
        str = str:sub(1,str:len()-2) .. "]"
        print_stdout("DEBUG: Evaluated CPU expressions: ".. str)
    end
    ppn = #cpuexprs
    tpp = #cpuexprs[1]

    if np == 0 then
        if debug then
            print_stdout(string.format("DEBUG: No -np given , setting according to pin expression and number of available hosts"))
        end
        np = givenNrNodes * #cpuexprs
        ppn = #cpuexprs
    elseif np < #cpuexprs*givenNrNodes then
        while np < #cpuexprs*givenNrNodes and #hosts > 1 do
            table.remove(hosts)
            givenNrNodes = getNumberOfNodes(hosts)
        end
        if #hosts == 1 and np < #cpuexprs then
            while np < #cpuexprs do
                table.remove(cpuexprs)
            end
        end
        np = #cpuexprs*givenNrNodes
        ppn = #cpuexprs
    end
    newhosts, ppn = assignHosts(hosts, np, ppn, tpp)
    if np > #cpuexprs*#newhosts and #perf > 0 then
        print_stderr("ERROR: Oversubsribing not allowed.")
        print_stderr(string.format("ERROR: You want %d processes but the pinning expression has only expressions for %d processes. There are only %d hosts in the host list.", np, #cpuexprs*#newhosts, #newhosts))
        os.exit(1)
    end
else
    ppn = math.tointeger(np / givenNrNodes)
    if nperdomain == nil then
        nperdomain = "N:"..tostring(ppn)
        if tpp > 0 then
            nperdomain = nperdomain..":"..tostring(tpp)
            if dist > 1 then
                nperdomain = nperdomain..":"..tostring(dist)
            end
        end
    end
    domainname, count, threads, distance = nperdomain:match("[E]*[:]*([NSCM]*):(%d+)[:]*(%d*)[:]*(%d*)")
    if math.tointeger(threads) == nil then
        if tpp > 1 then
            nperdomain = string.format("E:%s:%d:%d", domainname, count, tpp, dist)
        else
            tpp = 1
            nperdomain = string.format("E:%s:%d:%d", domainname, count, tpp, dist)
        end
    else
        tpp = math.tointeger(threads)
        nperdomain = string.format("E:%s:%d:%d", domainname, count, tpp, dist)
    end
    cpuexprs = calculateCpuExprs(nperdomain, cpuexprs)
    if debug then
        for p, ex in pairs(cpuexprs) do
            print_stdout(string.format("DEBUG: Process %d runs on CPUs %s", p, table.concat(ex, ",")))
        end
    end
    ppn = #cpuexprs
    if np == 0 then
        np = givenNrNodes * ppn
    end
    if np < ppn then
        if debug then
            print_stderr("WARN: Removing additional cpu expressions to get requested number of processes")
        end
        for i=np+1,ppn do
            if debug then
                print_stderr("WARN: Remove cpuexpr: "..table.concat(cpuexprs[#cpuexprs], ","))
            end
            table.remove(cpuexprs, #cpuexprs)
        end
        ppn = np
    elseif np > (givenNrNodes * ppn) and #perf > 0 then
        print_stderr("ERROR: Oversubsribing nodes not allowed!")
        print_stderr(string.format("ERROR: You want %d processes with %d on each of the %d hosts", np, ppn, givenNrNodes))
        os.exit(1)
    end
    newhosts, ppn = assignHosts(hosts, np, ppn, tpp)
end


local grouplist = {}
if #perf > 0 then
    perfexprs, grouplist = setPerfStrings(perf, cpuexprs)
end

local nrNodes = getNumberOfNodes(newhosts)
if np > #cpuexprs*nrNodes then
    np = #cpuexprs*nrNodes
elseif np < #cpuexprs then
    while np < #cpuexprs do
        table.remove(cpuexprs)
    end
    ppn = #cpuexprs
end

if skipStr == "" then
    if mpitype == "intelmpi" then
        maj, min = getMpiVersion()
        if not maj then maj = 2016 end
        if maj < 2000 then maj = maj + 2000 end
        if not min then min = 0 end

        if maj < 2017 and min <= 1 then
            if omptype == "intel" and nrNodes > 1 then
                skipStr = '-s 0x3'
            elseif omptype == "intel" and nrNodes == 1 then
                skipStr = '-s 0x3'
            elseif omptype == "gnu" and nrNodes > 1 then
                skipStr = '-s 0x1'
            elseif omptype == "gnu" and nrNodes == 1 then
                skipStr = '-s 0x0'
            end
        elseif maj >= 2017 then
            if omptype == "intel" then
                skipStr = '-s 0x0'
                if tpp > 1 then
                    skipStr = '-s 0x1'
                end
            elseif omptype == "gnu" then
                skipStr = "-s 0x1"
            end
        end
    elseif mpitype == "mvapich2" then
        if omptype == "intel" and nrNodes > 1 then
            skipStr = '-s 0x7'
        end
    elseif mpitype == "openmpi" then
        if omptype == "intel" and nrNodes > 1 then
            skipStr = '-s 0x7'
        elseif omptype == "intel" and nrNodes == 1 then
            skipStr = '-s 0x1'
            if tpp > 1 then
                skipStr = '-s 0x3'
            end
        elseif omptype == "gnu" and nrNodes > 1 then
            skipStr = '-s 0x7'
        elseif omptype == "gnu" and nrNodes == 1 then
            skipStr = '-s 0x0'
            if tpp > 0 then
                skipStr = '-s 0x3'
            end
        end
    elseif mpitype == "slurm" then
        if omptype == "intel" and nrNodes > 1 then
            if nrNodes == 1 then
                skipStr = '-s 0x1'
            else
                skipStr = '-s 0x3'
            end
        end
    end
end
if debug and skipStr ~= "" then
    print_stdout(string.format("DEBUG: Using skip option %s to skip pinning of shepard threads", skipStr))
end

local pid = likwid.getpid()
local hostfilename = string.format(".hostfile_%s.txt", pid)
local scriptfilename = string.format(".likwidscript_%s.txt", pid)
local outfilename = string.format(os.getenv("PWD").."/.output_%s_%%r_%%h.csv", pid)

checkLikwid()

if writeHostfile == nil or getEnvironment == nil or executeCommand == nil then
    print_stderr("ERROR: Initialization for MPI specific functions failed")
    os.exit(1)
end

writeHostfile(newhosts, hostfilename)
local skipped_ranks = writeWrapperScript(scriptfilename, table.concat(executable, " "), newhosts, envsettings, outfilename)
local env = getEnvironment()
local exitvalue = executeCommand(scriptfilename, hostfilename, env, nrNodes)

os.remove(scriptfilename)
os.remove(hostfilename)

infilepart = ".output_"..pid
filelist = listdir(os.getenv("PWD"), infilepart)
all_results = {}
if not use_marker then
    for i, file in pairs(filelist) do
        local host, rank, results, cpulist = parseOutputFile(file)
        if host ~= nil and rank ~= nil then
            if all_results[rank] == nil then
                all_results[rank] = {}
            end
            all_results[rank]["hostname"] = host
            all_results[rank]["results"] = results
            all_results[rank]["cpus"] = cpulist
            os.remove(file)
        end
    end
    if likwid.tablelength(all_results) > 0 then
        printMpiOutput(grouplist, all_results)
    end
else
    local tmpList = {}
    local cpuCount = 0
    local regionlist = {}
    for i, file in pairs(filelist) do
        host, rank, results, cpulist, rlist = parseMarkerOutputFile(file)
        if host ~= nil and rank ~= nil then
            if all_results[rank] == nil then
                all_results[rank] = {}
            end
            all_results[rank]["hostname"] = host
            all_results[rank]["cpus"] = cpulist
            for _, r in pairs(rlist) do
                table.insert(regionlist, r)
            end
            cpuCount = cpuCount + #cpulist
            tmpList[rank] = results
            os.remove(file)
        end
    end
    regionlist = uniqueList(regionlist)
    if likwid.tablelength(all_results) > 0 then
        for _,region in pairs(regionlist) do
            for rank,_ in pairs(all_results) do
                all_results[rank]["results"] = tmpList[rank][region]
            end
            printMpiOutput(grouplist, all_results, region)
        end
    end
end
os.exit(exitvalue)
