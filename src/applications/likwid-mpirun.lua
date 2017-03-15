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
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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
    print_stdout(string.format("likwid-mpirun --  Version %d.%d",likwid.version,likwid.release))
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
    print_stdout("Only one process on a socket measures the Uncore/RAPL counters, the other one(s) only core-local counters")
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
    print_stdout("-s, --skip <hex>\t Bitmask with threads to skip")
    print_stdout("-mpi <id>\t\t Specify which MPI should be used. Possible values: openmpi, intelmpi and mvapich2")
    print_stdout("\t\t\t If not set, module system is checked")
    print_stdout("-omp <id>\t\t Specify which OpenMP should be used. Possible values: gnu and intel")
    print_stdout("\t\t\t Only required for statically linked executables.")
    print_stdout("-hostfile\t\t Use custom hostfile instead of searching the environment")
    print_stdout("-g/-group <perf>\t Set a likwid-perfctr conform event set for measuring on nodes")
    print_stdout("-m/-marker\t\t Activate marker API mode")
    print_stdout("-O\t\t\t Output easily parseable CSV instead of fancy tables")
    print_stdout("-f\t\t\t Force overwrite of registers if they are in use. You can also use environment variable LIKWID_FORCE")
    print_stdout("")
    print_stdout("Processes are pinned to physical CPU cores first. For syntax questions see likwid-pin")
    print_stdout("")
    print_stdout("For CPU selection and which MPI rank measures Uncore counters the system topology")
    print_stdout("of the current system is used. There is currently no possibility to overcome this")
    print_stdout("limitation by providing a topology file or similar.")
    print_stdout("")
    examples()
end

local np = 0
local ppn = 0
local nperdomain = nil
local npernode = 0
local cpuexprs = {}
local perfexprs = {}
local hostfile = nil
local hosts = {}
local perf = {}
local mpitype = nil
local omptype = nil
local skipStr = ""
local executable = {}
local mpiopts = {}
local debug = false
local use_marker = false
local use_csv = false
local force = false
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
        if debug then
            print_stdout(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
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

    local f = io.popen(string.format("%s -V 2>&1", mpiexecutable), "r")
    if f ~= nil then
        local input = f:read("*a")
        ver1,ver2,ver3 = input:match("(%d+)%.(%d+)%.(%d+)")
        if ver1 == "1" then
            if tonumber(ver2) >= 7 then
                bindstr = "--bind-to none"
            elseif ver2 == "6" then
                bindstr = "--bind-to-none"
            end
        end
        f:close()
    end

    local cmd = string.format("%s -hostfile %s %s -np %d -npernode %d %s %s",
                                mpiexecutable, hostfile, bindstr,
                                np, ppn, table.concat(mpiopts, ' '), wrapperscript)
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
        for i, host in pairs(hostlist) do
            print_stdout(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
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
    for i,e in pairs(mpiopts) do
        envstr = envstr .. string.format("%s ",e)
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
        for i, host in pairs(hostlist) do
            print_stdout(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
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
        envstr = envstr .. string.format("%s=%s ", i, e)
    end

    local cmd = string.format("%s -f %s -np %d -ppn %d %s %s %s",
                                mpiexecutable, hostfile,
                                np, ppn, envstr, table.concat(mpiopts, ' '), wrapperscript)
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
        for i, host in pairs(hostlist) do
            print_stdout(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
        end
    end
    return hostlist
end

local function readHostfileSlurm(hostlist)
    nperhost = tonumber(os.getenv("SLURM_TASKS_PER_NODE"):match("(%d+)"))
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
    list = likwid.stringsplit(hostlist, ",")
    for i, item in pairs(list) do
        if not item:match("%[") then
            table.insert(outlist, item)
        else
            prefixzeros = 0
            host, start, ende,remain = item:match("(%a+)%[(%d+)-(%d+)%]([%w%d%[%]-]*)")
            if host and start and ende then
                if tonumber(start) ~= 0 then
                    for j=1,#start do
                        if start:sub(j,j+1) == '0' then
                            prefixzeros = prefixzeros + 1
                        end
                    end
                end
                if start and ende then
                    for j=start,ende do
                        newh = host..string.rep("0", prefixzeros)..tostring(math.tointeger(j))
                        if remain then
                            newh = newh .. remain
                        end
                        table.insert(outlist, newh)
                    end
                end
            end
        end
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
end

local function writeHostfileSlurm(hostlist, filename)
    l = {}
    for i, h in pairs(hostlist) do
        table.insert(l, h["hostname"])
    end
    likwid.setenv("SLURM_NODELIST", table.concat(l,","))
end

local function getEnvironmentSlurm()
    return {}
end

local function executeSlurm(wrapperscript, hostfile, env, nrNodes)
    if wrapperscript.sub(1,1) ~= "/" then
        wrapperscript = os.getenv("PWD").."/"..wrapperscript
    end
    local exec = string.format("srun -N %d --ntasks-per-node=%d --cpu_bind=none %s %s",
                                nrNodes, ppn, table.concat(mpiopts, ' '), wrapperscript)
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
        return "slurm"
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
    for i, exec in pairs({"mpiexec.hydra", "mpiexec", "mpirun"}) do
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
                end
            end
        end
    end
    if not mpitype then
        print_stderr("WARN: No supported MPI loaded in module system")
    end
    return mpitype
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
end

local function getOmpType()
    local cmd = string.format("ldd `which %s` 2>/dev/null", executable[1])
    local f = io.popen(cmd, 'r')
    if f ~= nil then
        cmd = string.format("ldd %s", executable[1])
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

local function assignHosts(hosts, np, ppn)
    tmp = np
    newhosts = {}
    current = 0
    if debug then
        print_stdout(string.format("Assign %d processes with %d per node to %d hosts", np, ppn, #hosts))
        print_stdout("Available hosts for scheduling:")
        print_stdout("Host", "Slots", "MaxSlots", "Interface")
        for i, h in pairs(hosts) do
            print_stdout (h["hostname"], h["slots"], h["maxslots"],"", h["interface"])
        end
    end
    local break_while = false
    while tmp > 0 and #hosts > 0 do
        for i, host in pairs(hosts) do
            if host["slots"] and host["slots"] >= ppn then
                if host["maxslots"] and host["maxslots"] < ppn then
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
                                            slots=ppn,
                                            maxslots=host["slots"],
                                            interface=host["interface"]})
                    if debug then
                        print_stdout(string.format("DEBUG: Add Host %s with %d slots to host list", host["hostname"], ppn))
                    end
                    current = ppn
                    hosts[i] = nil
                end
            elseif host["slots"] then
                --[[if host["maxslots"] then
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
                os.exit(1)
            else
                table.insert(newhosts, {hostname=host["hostname"],
                                        slots=ppn,
                                        maxslots=host["slots"],
                                        interface=host["interface"]})
                if debug then
                    print_stdout(string.format("DEBUG: Add Host %s with %d slots to host list", host["hostname"], ppn))
                end
                current = ppn
            end
            tmp = tmp - current
            if tmp < 1 then
                break_while = true
                break
            elseif tmp < ppn then
                ppn = tmp
            end
        end
        if break_while then
            break
        end
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
        table.insert(newexprs, table.concat(strList,","))
    end
    return newexprs
end

local function calculateCpuExprs(nperdomain, cpuexprs)
    local topo = likwid.getCpuTopology()
    local affinity = likwid.getAffinityInfo()
    local domainlist = {}
    local newexprs = {}
    domainname, count = nperdomain:match("[E:]*(%g*):(%d+)")

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
        for off=1,topo["numThreadsPerCore"] do
            for i=0,affinity["domains"][domidx]["numberOfProcessors"]/topo["numThreadsPerCore"] do
                table.insert(sortedlist, affinity["domains"][domidx]["processorList"][off + (i*topo["numThreadsPerCore"])])
            end
        end
        local tmplist = {}
        for j=1,count do
            table.insert(newexprs, tostring(sortedlist[1]))
            table.remove(sortedlist, 1)
        end
    end
    if debug then
        local str = "DEBUG: Resolved NperDomain string "..nperdomain.." to CPUs: "
        for i, expr in pairs(newexprs) do
            str = str .. expr .. " "
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

local function setPerfStrings(perflist, cpuexprs)
    local uncore = false
    local perfexprs = {}
    local grouplist = {}
    local cpuinfo = likwid.getCpuInfo()
    local affinity = likwid.getAffinityInfo()
    local socketList = {}
    local socketListFlags = {}
    for i, d in pairs(affinity["domains"]) do
        if d["tag"]:match("S%d+") then
            local tmpList = {}
            for j,cpu in pairs(d["processorList"]) do
                table.insert(tmpList, cpu)
            end
            table.insert(socketList, tmpList)
            table.insert(socketListFlags, 1)
        end
    end

    for k, perfStr in pairs(perflist) do
        local coreevents = {}
        local uncoreevents = {}
        local gdata = nil
        gdata = likwid.get_groupdata(perfStr)
        if gdata == nil then
            print_stderr("Cannot get data for group "..perfStr..". Skipping...")
        else
            table.insert(grouplist, gdata)
            if perfexprs[k] == nil then
                perfexprs[k] = {}
            end

            for i, e in pairs(gdata["Events"]) do
                if  not e["Counter"]:match("FIXC%d") and
                    not e["Counter"]:match("^PMC%d") and
                    not e["Counter"]:match("TMP%d") then
                    table.insert(uncoreevents, e)
                else
                    table.insert(coreevents, e)
                end
            end
            local tmpSocketFlags = {}
            for _,e in pairs(socketListFlags) do
                table.insert(tmpSocketFlags, e)
            end

            for i,cpuexpr in pairs(cpuexprs) do
                for j, cpu in pairs(likwid.stringsplit(cpuexpr,",")) do
                    local uncore = false
                    for sidx, socket in pairs(socketList) do
                        local switchedFlag = false
                        for _,c in pairs(socket) do
                            if c == tonumber(cpu) then
                                if tmpSocketFlags[sidx] == 1 then
                                    local eventStr = createEventString(coreevents)
                                    if #uncoreevents > 0 then
                                        eventStr = eventStr .. ","..createEventString(uncoreevents)
                                    end
                                    table.insert(perfexprs[k], eventStr)
                                    tmpSocketFlags[sidx] = 0
                                    switchedFlag = true
                                    uncore = true
                                    break
                                else
                                    table.insert(perfexprs[k], createEventString(coreevents))
                                    switchedFlag = true
                                    uncore = true
                                end
                            end
                        end
                        if switchedFlag then break end
                    end
                    if uncore then break end
                end
            end

            if debug then
                for i, expr in pairs(perfexprs[k]) do
                    print_stdout(string.format("DEBUG: Process %d measures with event set: %s", i-1, expr))
                end
            end
        end
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

local function writeWrapperScript(scriptname, execStr, hosts, outputname)
    if scriptname == nil or scriptname == "" then
        return
    end
    local oversubscripted = {}
    local commands = {}
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
        glrank_var = "${PMI_RANK:-$(($GLOBALSIZE * 2))}"
        glsize_var = tostring(math.tointeger(np))
        losize_var = "${MPI_LOCALNRANKS:-$SLURM_NTASKS_PER_NODE}"
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
        if #perf > 0 then
            table.insert(cmd,LIKWID_PERFCTR)
            if use_marker then
                table.insert(cmd,"-m")
            end
            cpuexpr_opt = "-C"
        else
            table.insert(cmd,LIKWID_PIN)
            table.insert(cmd,"-q")
        end
        if force and #perf > 0 then
            table.insert(cmd,"-f")
        end
        table.insert(cmd,skipStr)
        table.insert(cmd,cpuexpr_opt)
        table.insert(cmd,cpuexprs[i])
        if #perf > 0 then
            for j, expr in pairs(perfexprs) do
                table.insert(cmd,"-g")
                table.insert(cmd,expr[i])
            end
            table.insert(cmd,"-o")
            table.insert(cmd,outputname)
        end
        table.insert(cmd,execStr)
        commands[i] = table.concat(cmd, " ")
    end

    f:write("#!/bin/bash -l\n")
    f:write("GLOBALSIZE="..glsize_var.."\n")
    f:write("GLOBALRANK="..glrank_var.."\n")
    if os.getenv("OMP_NUM_THREADS") == nil then
        f:write("unset OMP_NUM_THREADS\n")
    else
        f:write(string.format("export OMP_NUM_THREADS=%s\n", os.getenv("OMP_NUM_THREADS")))
    end
    if mpitype == "intelmpi" then
        f:write("export I_MPI_PIN=disable\n")
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
        print_stdout("NODE_EXEC: "..commands[1])
    end
    f:write("\t"..commands[1].."\n")

    for i=2,#commands do
        f:write("elif [ \"$LOCALRANK\" -eq "..tostring(i-1).." ]; then\n")
        if debug then
            print_stdout("NODE_EXEC: "..commands[i])
        end
        f:write("\t"..commands[i].."\n")
    end
    f:write("else\n")
    f:write("\techo \"Unknown local rank $LOCALRANK\"\n")
    f:write("fi\n")
    f:close()
    os.execute("chmod +x "..scriptname)
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
    rank, host = filename:match("output_%d+_(%d+)_(%g+).csv")

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
                    local test = tonumber(cpustr:match("Core (%d+)"))
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
                if results[gidx][idx] == nil then
                    results[gidx][idx] = {}
                end
                for j, value in pairs(linelist) do
                    if event:match("[Rr]untime") then
                        results[gidx]["time"][cpulist[j]] = tonumber(value)
                    else
                        results[gidx][idx][cpulist[j]] = tonumber(value)
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
    local idx = 1
    local results = {}
    local f = io.open(filename, "r")
    if f == nil then
        print_stderr("ERROR: Cannot open output file "..filename)
        os.exit(1)
    end
    rank, host = filename:match("output_%d+_(%d+)_(%g+).csv")
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
            elseif parse_reg_info and line:match("TABLE,Region (%g+),Group (%d+) Raw,(%g+),") then
                current_region, gidx, gname  = line:match("TABLE,Region (%g+),Group (%d+) Raw,(%g+),")
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
            elseif parse_reg_info and line:match("^Region Info") then
                linelist = likwid.stringsplit(line,",")
                table.remove(linelist,1)
                for _, cpustr in pairs(linelist) do
                    if cpustr:match("Core %d+") then
                        local test = tonumber(cpustr:match("Core (%d+)"))
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
            elseif parse_reg_output then
                linelist = likwid.stringsplit(line,",")
                if linelist[2] ~= "TSC" then
                    table.remove(linelist,1)
                    table.remove(linelist,1)
                    for j=#linelist,1,-1 do
                        if linelist[j] == "" then
                            table.remove(linelist, j)
                        end
                    end
                    if results[current_region][gidx][idx] == nil then
                        results[current_region][gidx][idx] = {}
                    end
                    for j, value in pairs(linelist) do
                        results[current_region][gidx][idx][cpulist[j]] = tonumber(value)
                    end
                    idx = idx + 1
                end
            end
        end
    end
    for region, data in pairs(results) do
        results[region]["clock"] = clock
    end

    return host, tonumber(rank), results, cpulist
end


function percentile_table(inputtable, skip_cols, skip_lines)
    local function percentile(sorted_valuelist, k)
        index = tonumber(k)/100.0 * #sorted_valuelist
        if index - math.floor(index) >= 0.5 then
            index = math.ceil(index)
        else
            index = math.floor(index)
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
                    if all_results[rank]["results"][gidx][j] and
                       all_results[rank]["results"][gidx][j][cpu] then
                        value = likwid.num2str(all_results[rank]["results"][gidx][j][cpu])
                    end
                    table.insert(column, value)
                end
                table.insert(firsttab, column)
            end
        end

        if total_threads > 1 then
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
                        if all_results[rank]["results"][gidx][j] ~= nil and
                           all_results[rank]["results"][gidx][j][cpu] ~= nil then
                            counterlist[counter] = all_results[rank]["results"][gidx][j][cpu]
                        end
                    end
                    counterlist["time"] = all_results[rank]["results"][gidx]["time"][cpu]
                    counterlist["inverseClock"] = 1.0/all_results[rank]["results"]["clock"]
                    tmpList = {all_results[rank]["hostname"]..":"..tostring(rank)..":"..tostring(cpu)}
                    for j=1,#groupdata["Metrics"] do
                        local tmp = likwid.num2str(likwid.calculate_metric(gdata["Metrics"][j]["formula"], counterlist))
                        table.insert(tmpList, tmp)
                    end
                    table.insert(secondtab,tmpList)
                end
            end

            if total_threads > 1 then
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
                print_stdout("Region,"..tostring(region).. string.rep(",", maxLineFields  - 2))
            end
            print_stdout("Group,"..tostring(gidx) .. string.rep(",", maxLineFields  - 2))
            likwid.printcsv(firsttab, maxLineFields)
            if total_threads > 1 then likwid.printcsv(firsttab_combined, maxLineFields) end
            if gdata["Metrics"] then
                likwid.printcsv(secondtab, maxLineFields)
                if total_threads > 1 then likwid.printcsv(secondtab_combined, maxLineFields) end
            end
        else
            if region then
                print_stdout("Region: "..tostring(region))
            end
            print_stdout("Group: "..tostring(gidx))
            likwid.printtable(firsttab)
            if total_threads > 1 then likwid.printtable(firsttab_combined) end
            if gdata["Metrics"] then
                likwid.printtable(secondtab)
                if total_threads > 1 then likwid.printtable(secondtab_combined) end
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

for opt,arg in likwid.getopt(arg, {"n:","np:", "nperdomain:","pin:","hostfile:","h","help","v","g:","group:","mpi:","omp:","d","m","O","debug","marker","version","s:","skip:","f"}) do
    if (type(arg) == "string") then
        local s,e = arg:find("-")
        if s == 1 then
            print_stderr(string.format("ERROR: Argmument %s to option -%s starts with invalid character -.", arg, opt))
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
    elseif opt == "n" or opt == "np" then
        np = tonumber(arg)
        if np == nil then
            print_stderr("Argument for -n/-np must be a number")
            os.exit(1)
        end
    elseif opt == "nperdomain" then
        nperdomain = arg
        local domain, count = nperdomain:match("([NSCM]%d*):(%d+)")
        if domain == nil then
            print_stderr("Invalid option to -nperdomain")
            os.exit(1)
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
    elseif opt == "s" or opt == "skip" then
        skipStr = "-s "..arg
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
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

local test_mpiOpts = false
for i=1,#arg do
    if arg[i] == '--' then
        test_mpiOpts = true
    end
    if not test_mpiOpts then
        table.insert(executable, arg[i])
    elseif arg[i] ~= '--' then
        table.insert(mpiopts, arg[i])
    end
end

if #executable == 0 then
    print_stderr("ERROR: No executable given on commandline")
    os.exit(1)
else
    local do_which = false
    local found = false
    if likwid.access(executable[1], "x") == -1 then
        do_which = true
    else
        found = true
    end
    if not found then
        if do_which then
            local f = io.popen(string.format("which %s 2>/dev/null", executable[1]))
            if f ~= nil then
                executable[1] = f:read("*line")
                f:close()
                found = true
            end
            if debug then
                print_stdout("DEBUG: Executable given on commandline: "..table.concat(executable, " "))
            end
        end
    end
    if not found then
        print_stderr("ERROR: Cannot find executable given on commandline")
        os.exit(1)
    end
end
if #mpiopts > 0 and debug then
    print_stdout("DEBUG: MPI options given on commandline: "..table.concat(mpiopts, " "))
end

if mpitype == nil then
    mpitype = getMpiType()
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
    elseif mpitype == "slurm" and os.getenv("SLURM_NODELIST") ~= nil then
        hostlist = os.getenv("SLURM_NODELIST")
        hosts = readHostfileSlurm(hostlist)
    else
        local cpus = cpuCount()
        table.insert(hosts, {hostname='localhost', slots=cpus, maxslots=cpus})
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

if #cpuexprs > 0 then
    cpuexprs = calculatePinExpr(cpuexprs)
    if debug then
        str = "["
        for i, expr in pairs(cpuexprs) do
            str = str .. "["..expr.."], "
        end
        str = str:sub(1,str:len()-2) .. "]"
        print_stdout("DEBUG: Evaluated CPU expressions: ".. str)
    end
    ppn = #cpuexprs
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
    newhosts = assignHosts(hosts, np, ppn)
    if np > #cpuexprs*#newhosts and #perf > 0 then
        print_stderr("ERROR: Oversubsribing not allowed.")
        print_stderr(string.format("ERROR: You want %d processes but the pinning expression has only expressions for %d processes. There are only %d hosts in the host list.", np, #cpuexprs*#newhosts, #newhosts))
        os.exit(1)
    end
elseif nperdomain ~= nil then
    cpuexprs = calculateCpuExprs(nperdomain, cpuexprs)
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
                print_stderr("WARN: Remove cpuexpr: "..cpuexprs[#cpuexprs])
            end
            table.remove(cpuexprs, #cpuexprs)
        end
        ppn = np
    elseif np > (givenNrNodes * ppn) and #perf > 0 then
        print_stderr("ERROR: Oversubsribing nodes not allowed!")
        print_stderr(string.format("ERROR: You want %d processes with %d on each of the %d hosts", np, ppn, givenNrNodes))
        os.exit(1)
    end
    newhosts, ppn = assignHosts(hosts, np, ppn)
elseif ppn == 0 and np > 0 then
    maxnp = 0
    maxppn = 0
    for i, host in pairs(hosts) do
        maxnp = maxnp + host["slots"]
        if host["slots"] > maxppn then
            maxppn = host["slots"]
        end
    end
    if ppn == 0 then
        ppn = 1
    end
    if ppn > maxppn and np > maxppn then
        ppn = maxppn
    elseif np < maxppn then
        ppn = np
    elseif maxppn == np then
        ppn = maxppn
    end
    if (ppn * givenNrNodes) < np then
        if #perf == 0 then
            print_stderr("ERROR: Processes cannot be equally distributed")
            print_stderr(string.format("WARN: You want %d processes on %d hosts.", np, givenNrNodes))
            ppn = np/givenNrNodes
            print_stderr(string.format("WARN: Sanitizing number of processes per node to %d", ppn))
        else
            ppn = 0
            os.exit(1)
        end
    end
    local newexprs = calculateCpuExprs("E:N:"..tostring(ppn), cpuexprs)
    local copynp = np
    while copynp > 0 do
        for i, expr in pairs(newexprs) do
            local exprlist = likwid.stringsplit(expr, ",")
            local seclength = math.ceil(#exprlist/ppn)
            local offset = 0
            for p=1, ppn do
                local str = ""
                for j=1, seclength do
                    if exprlist[((p-1)*seclength) + j] then
                        str = str .. exprlist[((p-1)*seclength) + j] ..","
                    end
                end
                if str ~= "" then
                    str = str:sub(1,#str-1)
                    table.insert(cpuexprs, str)
                    copynp = copynp - seclength
                else
                    break
                end
            end
        end
    end
    newhosts, ppn = assignHosts(hosts, np, ppn)
    if np < ppn*#newhosts then
        np = 0
        for i, host in pairs(newhosts) do
            np = np + host["slots"]
        end
    end
else
    print_stderr("ERROR: Commandline settings are not supported.")
    os.exit(1)
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
        if omptype == "intel" and nrNodes > 1 then
            skipStr = '-s 0x3'
        elseif omptype == "intel" and nrNodes == 1 then
            skipStr = '-s 0x3'
        elseif omptype == "gnu" and nrNodes > 1 then
            skipStr = '-s 0x1'
        elseif omptype == "gnu" and nrNodes == 1 then
            skipStr = '-s 0x0'
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
        elseif omptype == "gnu" and nrNodes > 1 then
            skipStr = '-s 0x7'
        elseif omptype == "gnu" and nrNodes == 1 then
            skipStr = '-s 0x0'
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
writeWrapperScript(scriptfilename, table.concat(executable, " "), newhosts, outfilename)
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
    for i, file in pairs(filelist) do
        host, rank, results, cpulist = parseMarkerOutputFile(file)
        if host ~= nil and rank ~= nil then
            if all_results[rank] == nil then
                all_results[rank] = {}
            end
            all_results[rank]["hostname"] = host
            all_results[rank]["cpus"] = cpulist
            cpuCount = cpuCount + #cpulist
            tmpList[rank] = results
            os.remove(file)
        end
    end
    if likwid.tablelength(all_results) > 0 then
        for region, _ in pairs(tmpList[0]) do
            for rank,_ in pairs(all_results) do
                all_results[rank]["results"] = tmpList[rank][region]
            end
            printMpiOutput(grouplist, all_results, region)
        end
    end
end
os.exit(exitvalue)
