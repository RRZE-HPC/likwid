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

local function version()
    print(string.format("likwid-mpirun --  Version %d.%d",likwid.version,likwid.release))
end

local function examples()
    print("Examples:")
    print("Run 32 processes on hosts in hostlist")
    print("likwid-mpirun -np 32 ./a.out")
    print("")
    print("Run 1 MPI process on each socket")
    print("likwid-mpirun -nperdomain S:1 ./a.out")
    print("Total amount of MPI processes is calculated using the number of hosts in the hostfile")
    print("")
    print("For hybrid MPI/OpenMP jobs you need to set the -pin option")
    print("Starts 2 MPI processes on each host, one on socket 0 and one on socket 1")
    print("Each MPI processes starts 2 OpenMP threads pinned to the same socket")
    print("likwid-mpirun -pin S0:2_S1:2 ./a.out")
    print("")
    print("Run 2 processes on each socket and measure the MEM performance group")
    print("likwid-mpirun -nperdomain S:2 -g MEM ./a.out")
    print("Only one process on a socket measures the Uncore/RAPL counters, the other one(s) only core-local counters")
    print("")
end

local function usage()
    version()
    print("A wrapper script to pin threads spawned by MPI processes and measure hardware performance counters.\n")
    print("Options:")
    print("-h, --help\t\t Help message")
    print("-v, --version\t\t Version information")
    print("-d, --debug\t\t Debugging output")
    print("-n/-np <count>\t\t Set the number of processes")
    print("-nperdomain <domain>\t Set the number of processes per node by giving an affinity domain and count")
    print("-pin <list>\t\t Specify pinning of threads. CPU expressions like likwid-pin separated with '_'")
    print("-s, --skip <hex>\t Bitmask with threads to skip")
    print("-mpi <id>\t\t Specify which MPI should be used. Possible values: openmpi, intelmpi and mvapich2")
    print("\t\t\t If not set, module system is checked")
    print("-omp <id>\t\t Specify which OpenMP should be used. Possible values: gnu and intel")
    print("\t\t\t Only required for statically linked executables.")
    print("-hostfile\t\t Use custom hostfile instead of searching the environment")
    print("-g/-group <perf>\t Set a likwid-perfctr conform event set for measuring on nodes")
    print("-m/-marker\t\t Activate marker API mode")
    print("")
    print("Processes are pinned to physical CPU cores first.")
    print("")
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
local debug = false
local use_marker = false
local use_csv = false
local force = false

local LIKWID_PIN="<INSTALLED_PREFIX>/bin/likwid-pin"
local LIKWID_PERFCTR="<INSTALLED_PREFIX>/bin/likwid-perfctr"
local MPIINFO = {}
local MPIROOT = os.getenv("MPIHOME")
if MPIROOT == nil then
    MPIROOT = os.getenv("MPI_ROOT")
end
if MPIROOT == nil then
    MPIROOT = ""
end

local MPIEXEC = { openmpi=MPIROOT.."/bin/mpiexec", intelmpi=MPIROOT.."/bin/mpiexec.hydra", mvapich2="mpirun"}


local readHostfile = nil
local writeHostfile = nil
local getEnvironment = nil


local function readHostfileOpenMPI(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print("DEBUG: Reading hostfile in openmpi style")
    end
    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname, slots, maxslots = line:match("^([%.%a%d]+)%s+slots=(%d*)%s+max%-slots=(%d*)")
            if not hostname then
                hostname, slots = line:match("^([%.%a%d]+)%s+slots=(%d*)")
                if not hostname then
                    hostname = line:match("^([%.%a%d]+)")
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
            print(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
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
        print("ERROR: Cannot open hostfile "..filename)
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

    local f = io.popen(string.format("%s -V", MPIINFO["openmpi"]["MPIEXEC"]), "r")
    if f ~= nil then
        local input = f:read("*a")
        ver1,ver2,ver3 = input:match("(%d+)%.(%d+)%.(%d+)")
        if ver1 == "1" then
            if ver2 == "7" then
                bindstr = "--bind-to none"
            elseif ver2 == "6" then
                bindstr = "--bind-to-none"
            end
        end
        f:close()
    end

    local cmd = string.format("%s -hostfile %s %s -np %d -npernode %d %s",
                                MPIINFO["openmpi"]["MPIEXEC"], hostfile, bindstr,
                                np, ppn, wrapperscript)
    if debug then
        print("EXEC: "..cmd)
    end
    os.execute(cmd)
end

local function readHostfileIntelMPI(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print("DEBUG: Reading hostfile in intelmpi style")
    end
    local topo = likwid.getCpuTopology()
    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname, slots = line:match("^([%.%a%d]+):(%d+)")
            if not hostname then
                hostname = line:match("^([%.%a%d]+)")
                slots = topo["numHWThreads"]
            end
            table.insert(hostlist, {hostname=hostname, slots=slots, maxslots=slots})
        end
    end
    if debug then
        for i, host in pairs(hostlist) do
            print(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
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
        print("ERROR: Cannot open hostfile "..filename)
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
    if wrapperscript.sub(1,1) ~= "/" then
        wrapperscript = os.getenv("PWD").."/"..wrapperscript
    end
    if hostfile.sub(1,1) ~= "/" then
        hostfile = os.getenv("PWD").."/"..hostfile
    end

    if debug then
        print(string.format("EXEC: %s/bin/mpdboot -r ssh -n %d -f %s", MPIROOT, nrNodes, hostfile))
        print(string.format("EXEC: %s/bin/mpiexec -perhost %d -env I_MPI_PIN 0 -np %d %s", MPIROOT, ppn, np, wrapperscript))
        print(string.format("EXEC: %s/bin/mpdallexit", MPIROOT))
    end

    os.execute(string.format("%s/bin/mpdboot -r ssh -n %d -f %s", MPIROOT, nrNodes, hostfile))
    os.execute(string.format("%s/bin/mpiexec -perhost %d -env I_MPI_PIN 0 -np %d %s", MPIROOT, ppn, np, wrapperscript))
    os.execute(string.format("%s/bin/mpdallexit", MPIROOT))
end

local function readHostfileMvapich2(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print("DEBUG: Reading hostfile in mvapich2 style")
    end
    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname, slots, interface = line:match("^([%.%a%d]+):(%d+):([%a%d]+)")
            if not hostname then
                hostname, slots = line:match("^([%.%a%d]+):(%d+)")
                if not hostname then
                    hostname = line:match("^([%.%a%d]+)")
                    slots = 1
                    interface = nil
                else
                    interface = nil
                end
            end
            table.insert(hostlist, {hostname=hostname, slots=slots, maxslots=slots, interface=interface})
        end
    end
    if debug then
        for i, host in pairs(hostlist) do
            print(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
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
        print("ERROR: Cannot open hostfile "..filename)
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
    local cmd = string.format("%s -f %s -np %d -ppn %d %s",
                                MPIINFO["mvapich2"]["MPIEXEC"], hostfile,
                                np, ppn, wrapperscript)
    if debug then
        print("EXEC: "..cmd)
    end
    os.execute(cmd)
end

MPIINFO =      { openmpi={ MPIEXEC=MPIROOT.."/bin/mpiexec",
                            readHostfile = readHostfileOpenMPI,
                            writeHostfile = writeHostfileOpenMPI,
                            getEnvironment = getEnvironmentOpenMPI,
                            executeCommand = executeOpenMPI},
                  intelmpi={MPIEXEC=MPIROOT.."/bin/mpiexec",
                            readHostfile = readHostfileIntelMPI,
                            writeHostfile = writeHostfileIntelMPI,
                            getEnvironment = getEnvironmentIntelMPI,
                            executeCommand = executeIntelMPI},
                  mvapich2={MPIEXEC=MPIROOT.."/bin/mpiexec.hydra",
                            readHostfile = readHostfileMvapich2,
                            writeHostfile = writeHostfileMvapich2,
                            getEnvironment = getEnvironmentMvapich2,
                            executeCommand = executeMvapich2}
                }

local function readHostfilePBS(filename)
    local hostlist = {}
    if filename == nil or filename == "" then
        return {}
    end
    local f = io.open(filename, "r")
    if f == nil then
        print("ERROR: Cannot open hostfile "..filename)
        os.exit(1)
    end
    if debug then
        print("DEBUG: Reading hostfile from batch system")
    end
    local t = f:read("*all")
    f:close()
    for i, line in pairs(likwid.stringsplit(t,"\n")) do
        if line:match("^#") == nil and line:match("^%s*$") == nil then
            hostname = line:match("^([%.%a%d]+)")
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
            print(string.format("DEBUG: Read host %s with %d slots and %d slots maximally", host["hostname"], host["slots"], host["maxslots"]))
        end
    end
    return hostlist
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
    cmd = "tclsh /apps/modules/modulecmd.tcl sh list -t 2>&1"
    local f = io.popen(cmd, 'r')
    if f == nil then
        cmd = os.getenv("SHELL").." -c 'module -t 2>&1'"
        f = io.popen(cmd, 'r')
    end
    if f ~= nil then
        local s = assert(f:read('*a'))
        f:close()
        s = string.gsub(s, '^%s+', '')
        s = string.gsub(s, '%s+$', '')
        for i,line in pairs(likwid.stringsplit(s, "\n")) do
            if line:match("^intelmpi") then
                mpitype = "intelmpi"
                --libmpi%a*.so
            elseif line:match("^openmpi") then
                mpitype = "openmpi"
                --libmpi.so
            elseif line:match("^mvapich2") then
                mpitype = "mvapich2"
                --libmpich.so
            end
        end
    end
    if not mpitype then
        print("WARN: No supported MPI loaded in module system")
    end
    return mpitype
end

local function getOmpType()
    local cmd = string.format("ldd `which %s`", executable[1])
    local f = assert(io.popen(cmd, 'r'))
    local s = assert(f:read('*a'))
    f:close()
    for i,line in pairs(likwid.stringsplit(s, "\n")) do
        if line:match("libgomp.so") then
            omptype = "gnu"
            break
        elseif line:match("libiomp%d*.so") then
            omptype = "intel"
            break
        end
    end
    if not omptype then
        print("WARN: Cannot get OpenMP variant from executable, trying module system")
        cmd = "tclsh /apps/modules/modulecmd.tcl sh list -t 2>&1"
        local f = io.popen(cmd, 'r')
        if f ~= nil then
            f:close()
            cmd = os.getenv("SHELL").." -c 'module -t list 2>&1'"
            f = io.popen(cmd, 'r')
        end
        if f ~= nil then
            local s = assert(f:read('*a'))
            f:close()
            s = string.gsub(s, '^%s+', '')
            s = string.gsub(s, '%s+$', '')
            for i,line in pairs(likwid.stringsplit(s, "\n")) do
                if line:match("^intel") then
                    omptype = "intel"
                elseif line:match("^gnu") then
                    omptype = "gnu"
                end
            end
        end
        if not omptype then
            print("WARN: No supported OpenMP loaded in module system")
        end
    end
    return omptype
end

local function assignHosts(hosts, np, ppn)
    tmp = np
    newhosts = {}
    current = 0
    local break_while = false
    while tmp > 0 and #hosts > 0 do
        for i, host in pairs(hosts) do
            if host["slots"] and host["slots"] >= ppn then
                if host["maxslots"] and host["maxslots"] < ppn then
                    table.insert(newhosts, {hostname=host["hostname"],
                                            slots=host["maxslots"],
                                            maxslots=host["maxslots"],
                                            interface=host["interface"]})
                    current = host["maxslots"]
                    hosts[i] = nil
                else
                    table.insert(newhosts, {hostname=host["hostname"],
                                            slots=ppn,
                                            maxslots=host["slots"],
                                            interface=host["interface"]})
                    current = ppn
                    hosts[i] = nil
                end
            elseif host["slots"] then
                if host["maxslots"] then
                    if host["maxslots"] < ppn then
                        print(string.format("WARN: Oversubscription for host %s needed, but max-slots set to %d.",
                                                host["hostname"], host["maxslots"]))
                        table.insert(newhosts, {hostname=host["hostname"],
                                                slots=host["maxslots"],
                                                maxslots=host["maxslots"],
                                                interface=host["interface"]})
                        current = host["maxslots"]
                        host["maxslots"] = 0
                        hosts[i] = nil
                    else
                        print(string.format("WARN: Oversubscription for host %s.", host["hostname"]))
                        table.insert(newhosts, {hostname=host["hostname"],
                                            slots=ppn,
                                            maxslots=host["maxslots"],
                                            interface=host["interface"]})
                        current = ppn
                        
                    end
                else
                    print(string.format("WARN: Oversubscription for host %s.", host["hostname"]))
                    table.insert(newhosts, {hostname=host["hostname"],
                                        slots=ppn,
                                        maxslots=host["slots"],
                                        interface=host["interface"]})
                    current = ppn
                end
            else
                table.insert(newhosts, {hostname=host["hostname"],
                                        slots=ppn,
                                        maxslots=host["slots"],
                                        interface=host["interface"]})
                current = ppn
            end
            tmp = tmp - current
            if tmp <= 1 then
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
        print("DEBUG: Scheduling on hosts:")
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
            print(str)
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
        print(str)
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
        print(str)
    end
    return newexprs
end

local function createEventString(eventlist)
    if eventlist == nil or #eventlist == 0 then
        print("ERROR: Empty event list. Failed to create event set string")
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
    if debug then
        print("DEBUG: Created event set string "..str)
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
            print("Cannot get data for group "..perfStr..". Skipping...")
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
                    print(string.format("DEBUG: Process %d measures with event set: %s", i-1, expr))
                end
            end
        end
    end
    return perfexprs, grouplist
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
        glsize_var = tostring(np)
        losize_var = tostring(ppn)
    elseif mpitype == "mvapich2" then
        glrank_var = "${PMI_RANK:-$(($GLOBALSIZE * 2))}"
        glsize_var = tostring(np)
        losize_var = tostring(ppn)
    else
        print("Invalid MPI vendor "..mpitype)
        return
    end

    local taillength = np % ppn
    if taillength ~= 0 then
        local full = tostring(np -taillength)
        table.insert(oversubscripted, "if [ $GLOBALRANK >= "..full.." ]; then\n")
        table.insert(oversubscripted, "\tLOCALRANK=$($GLOBALRANK - "..full..")\n")
        table.insert(oversubscripted, "fi\n")
    end

    local f = io.open(scriptname, "w")
    if f == nil then
        print("ERROR: Cannot open hostfile "..scriptname)
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
        if force then
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

    f:write("#!/bin/bash\n")
    f:write("GLOBALSIZE="..glsize_var.."\n")
    f:write("GLOBALRANK="..glrank_var.."\n")
    f:write("unset OMP_NUM_THREADS\n")
    if mpitype == "intelmpi" then
        f:write("export I_MPI_PIN=disable\n")
    end
    f:write("LOCALSIZE="..losize_var.."\n\n")

    if mpitype == "openmpi" then
        f:write("LOCALRANK=$OMPI_COMM_WORLD_LOCAL_RANK\n\n")
    else
        local full = tostring(np - (np % ppn))
        f:write("if [ \"$GLOBALRANK\" -lt "..full.." ]; then\n")
        f:write("\tLOCALRANK=$(($GLOBALRANK % $LOCALSIZE))\n")
        f:write("else\n")
        f:write("\tLOCALRANK=$(($GLOBALRANK - ("..full.." - 1)))\n")
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
        print("NODE_EXEC: "..commands[1])
    end
    f:write("\t"..commands[1].."\n")

    for i=2,#commands do
        f:write("elif [ \"$LOCALRANK\" -eq "..tostring(i-1).." ]; then\n")
        if debug then
            print("NODE_EXEC: "..commands[i])
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
        print("ERROR: Cannot open output file "..filename)
        os.exit(1)
    end
    rank, host = filename:match("output_%d+_(%d+)_(%g+).csv")

    local t = f:read("*all")
    f:close()
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
        print("ERROR: Cannot open output file "..filename)
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
           (not line:match("^TABLE")) and
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
            elseif parse_reg_info and line:match("^%d+,%g+") then
                gidx, gname, current_region = line:match("^(%d+),(%g+),(%g+)")
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
    for region, data in pairs(results) do
        results[region]["clock"] = clock
    end

    return host, tonumber(rank), results, cpulist
end

function printMpiOutput(group_list, all_results)

    if #group_list == 0 or likwid.tablelength(all_results) == 0 then
        return
    end
    for gidx, gdata in pairs(group_list) do
        local firsttab = {}
        local firsttab_combined = {}
        local secondtab = {}
        local secondtab_combined = {}
        local total_threads = 0
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
            table.insert(desc, "")
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
                    if all_results[rank]["results"][gidx][j] ~= nil then
                        table.insert(column, all_results[rank]["results"][gidx][j][cpu])
                    else
                        table.insert(column, 0)
                    end
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
                        if all_results[rank]["results"][gidx][j] ~= nil then
                            counterlist[counter] = all_results[rank]["results"][gidx][j][cpu]
                        else
                            counterlist[counter] = 0
                        end
                    end
                    counterlist["time"] = all_results[rank]["results"][gidx]["time"][cpu]
                    counterlist["inverseClock"] = 1.0/all_results[rank]["results"]["clock"]
                    tmpList = {all_results[rank]["hostname"]..":"..tostring(rank)..":"..tostring(cpu)}
                    for j=1,#groupdata["Metrics"] do
                        local tmp = likwid.calculate_metric(gdata["Metrics"][j]["formula"], counterlist)
                        if tostring(tmp):len() > 12 then
                            tmp = string.format("%e",tmp)
                        end
                        table.insert(tmpList, tostring(tmp))
                    end
                    table.insert(secondtab,tmpList)
                end
            end

            if total_threads > 1 then
                secondtab_combined = likwid.tableToMinMaxAvgSum(secondtab, 1, 1)
            end
        end
        if use_csv then
            print("Group,"..tostring(gidx))
            likwid.printcsv(firsttab)
            if total_threads > 1 then likwid.printcsv(firsttab_combined) end
            if gdata["Metrics"] then
                likwid.printcsv(secondtab)
                if total_threads > 1 then likwid.printcsv(secondtab_combined) end
            end
        else
            print("Group: "..tostring(gidx))
            likwid.printtable(firsttab)
            if total_threads > 1 then likwid.printtable(firsttab_combined) end
            if gdata["Metrics"] then
                likwid.printtable(secondtab)
                if total_threads > 1 then likwid.printtable(secondtab_combined) end
            end
        end
    end
end

if #arg == 0 then
    usage()
    os.exit(0)
end

for opt,arg in likwid.getopt(arg, {"n:","np:", "nperdomain:","pin:","hostfile:","h","help","v","g:","group:","mpi:","omp:","d","m","O","debug","marker","version","s:","skip:","f"}) do
    if (type(arg) == "string") then
        local s,e = arg:find("-")
        if s == 1 then
            print(string.format("ERROR: Argmument %s to option -%s starts with invalid character -.", arg, opt))
            print("ERROR: Did you forget an argument to an option?")
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
    elseif opt == "nperdomain" then
        nperdomain = arg
        local domain, count = nperdomain:match("([NSCM]%d*):(%d+)")
        if domain == nil then
            print("Invalid option to -nperdomain")
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
        print("Invalid commandline option -"..arg)
        os.exit(1)
    end
end

if MPIROOT == "" then
    print("Please load a MPI module or set path to MPI solder in MPIHOME environment variable")
    print("$MPIHOME/bin/<MPI launcher> should be valid")
    os.exit(1)
end

if np == 0 and nperdomain == nil and #cpuexprs == 0 then
    print("ERROR: No option -n/-np, -nperdomain or -pin")
    os.exit(1)
end

for i=1,#arg do
    table.insert(executable, arg[i])
end
if #executable == 0 then
    print("ERROR: No executable given on commandline")
    os.exit(1)
else
    if debug then
        print("DEBUG: Executable given on commandline: "..table.concat(executable, " "))
    end
end

if mpitype == nil then
    mpitype = getMpiType()
    if debug then
        print("DEBUG: Using MPI implementation "..mpitype)
    end
end
if mpitype ~= "intelmpi" and mpitype ~= "mvapich2" and mpitype ~= "openmpi" then
    print("ERROR: Unknown MPI given. Possible values: openmpi, intelmpi, mvapich2")
    os.exit(1)
end

if omptype == nil then
    omptype = getOmpType()
    if debug and omptype ~= nil then
        print("DEBUG: Using OpenMP implementation "..omptype)
    end
end
if omptype == nil then
    print("WARN: Cannot extract OpenMP vendor from executable or commandline, assuming no OpenMP")
end

if not hostfile then
    hostfile = os.getenv("PBS_NODEFILE")
    if not hostfile or hostfile == "" then
        hostfile = os.getenv("LOADL_HOSTFILE")
    end
    if not hostfile or hostfile == "" then
        hostfile = os.getenv("SLURM_HOSTFILE")
    end
    if not hostfile or hostfile == "" then
        print("ERROR: No hostfile given and not in batch environment")
        os.exit(1)
    end
    hosts = readHostfilePBS(hostfile)
else
    hosts = MPIINFO[mpitype]["readHostfile"](hostfile)
end

local givenNrNodes = getNumberOfNodes(hosts)

if skipStr == "" then
    if mpitype == "intelmpi" then
        if omptype == "intel" and givenNrNodes > 1 then
            skipStr = '-s 0x3'
        elseif omptype == "intel" and givenNrNodes == 1 then
            skipStr = '-s 0x1'
        elseif omptype == "gnu" and givenNrNodes > 1 then
            skipStr = '-s 0x1'
        elseif omptype == "gnu" and givenNrNodes == 1 then
            skipStr = '-s 0x0'
        end
    elseif mpitype == "mvapich2" then
        if omptype == "intel" and givenNrNodes > 1 then
            skipStr = '-s 0x7'
        end
    elseif mpitype == "openmpi" then
        if omptype == "intel" and givenNrNodes > 1 then
            skipStr = '-s 0x7'
        elseif omptype == "intel" and givenNrNodes == 1 then
            skipStr = '-s 0x1'
        elseif omptype == "gnu" and givenNrNodes > 1 then
            skipStr = '-s 0x7'
        elseif omptype == "gnu" and givenNrNodes == 1 then
            skipStr = '-s 0x0'
        end
    end
end
if debug and skipStr ~= "" then
    print(string.format("DEBUG: Using skip option %s to skip pinning of shepard threads", skipStr))
end

if #perf > 0 then
    local sum_maxslots = 0
    local topo = likwid.getCpuTopology()
    if debug then
        print("DEBUG: Switch to perfctr mode, there are "..tostring(#perf).." eventsets given on the commandline")
    end
    for i, host in pairs(hosts) do
        if debug then
            local str = string.format("DEBUG: Working on host %s with %d slots", host["hostname"], host["slots"])
            if host["maxslots"] ~= nil then
                str = str .. string.format(" and %d slots maximally", host["maxslots"])
            end
            print(str)
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
        print("ERROR: Processes requested exceeds maximally available slots of given hosts. Maximal processes: "..sum_maxslots)
        os.exit(1)
    end
end

if #cpuexprs > 0 then
    cpuexprs = calculatePinExpr(cpuexprs)
    ppn = #cpuexprs
    if np == 0 then
        if debug then
            print(string.format("DEBUG: No -np given , setting according to pin expression and number of available hosts"))
        end
        np = givenNrNodes * #cpuexprs
        ppn = #cpuexprs
    elseif np < #cpuexprs*givenNrNodes then
        while np < #cpuexprs*givenNrNodes and #cpuexprs > 1 do
            table.remove(cpuexprs)
        end
        ppn = #cpuexprs
    end
    newhosts = assignHosts(hosts, np, ppn)
    if np > #cpuexprs*#newhosts and #perf > 0 then
        print("ERROR: Oversubsribing not allowed.")
        print(string.format("ERROR: You want %d processes but the pinning expression has only expressions for %d processes. There are only %d hosts in the host list.", np, #cpuexprs*#newhosts, #newhosts))
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
            print("WARN: Removing additional cpu expressions to get requested number of processes")
        end
        for i=np+1,ppn do
            if debug then
                print("WARN: Remove cpuexpr: "..cpuexprs[#cpuexprs])
            end
            table.remove(cpuexprs, #cpuexprs)
        end
        ppn = np
    elseif np > (givenNrNodes * ppn) and #perf > 0 then
        print("ERROR: Oversubsribing nodes not allowed!")
        print(string.format("ERROR: You want %d processes with %d on each of the %d hosts", np, ppn, givenNrNodes))
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
    end
    if (ppn * givenNrNodes) < np then
        if #perf == 0 then
            print("ERROR: Processes cannot be equally distributed")
            print(string.format("WARN: You want %d processes on %d hosts.", np, givenNrNodes))
            --np = givenNrNodes * ppn
            ppn = np/givenNrNodes
            print(string.format("WARN: Sanitizing number of processes per node to %d", ppn))
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
    print("ERROR: Commandline settings are not supported.")
    os.exit(1)
end

local grouplist = {}
if #perf > 0 then
    perfexprs, grouplist = setPerfStrings(perf, cpuexprs)
end

local nrNodes = getNumberOfNodes(newhosts)

local pid = likwid.getpid()
local hostfilename = string.format(".hostfile_%s.txt", pid)
local scriptfilename = string.format(".likwidscript_%s.txt", pid)
local outfilename = string.format(os.getenv("PWD").."/.output_%s_%%r_%%h.csv", pid)

MPIINFO[mpitype]["writeHostfile"](newhosts, hostfilename)
writeWrapperScript(scriptfilename, table.concat(executable, " "), newhosts, outfilename)
local env = MPIINFO[mpitype]["getEnvironment"]()
MPIINFO[mpitype]["executeCommand"](scriptfilename, hostfilename, env, nrNodes)

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
    if #all_results > 0 then
        printMpiOutput(grouplist, all_results)
    end
else
    local tmpList = {}
    for i, file in pairs(filelist) do
        host, rank, results, cpulist = parseMarkerOutputFile(file)
        if host ~= nil and rank ~= nil then
            if all_results[rank] == nil then
                all_results[rank] = {}
            end
            all_results[rank]["hostname"] = host
            all_results[rank]["cpus"] = cpulist
            tmpList[rank] = results
            os.remove(file)
        end
    end
    if #all_results > 0 then
        for reg, _ in pairs(tmpList[0]) do
            print("Region: "..reg)
            for rank,_ in pairs(all_results) do
                all_results[rank]["results"] = tmpList[rank][reg]
            end
            printMpiOutput(grouplist, all_results)
        end
    end
end
