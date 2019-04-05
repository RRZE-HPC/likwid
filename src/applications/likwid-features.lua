#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-features.lua
 *
 *      Description:  A application to retrieve and manipulate CPU features.
 *
 *      Version:   4.0
 *      Released:  28.04.2015
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@gmail.com
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
local cpuinfo = likwid.getCpuInfo()

print_stdout = print
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end

GENERAL_FEATURES = {"HW_PREFETCHER", "CL_PREFETCHER", "DCU_PREFETCHER", "IP_PREFETCHER"}
KNL_FEATURES = {"HW_PREFETCHER", "DCU_PREFETCHER"}
FEATURES = GENERAL_FEATURES
if cpuinfo["short_name"] == "knl" then
    FEATURES = KNL_FEATURES
end

function version()
    print_stdout(string.format("likwid-features -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

function usage()
    version()
    print_stdout("A tool list and modify the states of CPU features.\n")
    print_stdout("Options:")
    print_stdout("-h, --help\t\t Help message")
    print_stdout("-v, --version\t\t Version information")
    print_stdout("-a, --all\t\t List all available features")
    print_stdout("-l, --list\t\t List features and state for given CPUs")
    print_stdout("-c, --cpus <list>\t Perform operations on given CPUs")
    print_stdout("-e, --enable <list>\t List of features that should be enabled")
    print_stdout("-d, --disable <list>\t List of features that should be disabled")
    print_stdout()
    print_stdout("Currently modifiable features:")
    print_stdout(table.concat(FEATURES, ", "))
end

if #arg == 0 then
    usage()
    os.exit(0)
end

listFeatures = false
num_cpus = 0
cpulist = {}
enableList = {}
disableList = {}
skipList = {}

for opt,arg in likwid.getopt(arg, {"h","v","l","c:","e:","d:","a","help","version","list", "enable:", "disable:","all", "cpus:"}) do
    if (type(arg) == "string") then
        local s,e = arg:find("-");
        if s == 1 then
            print_stderr(string.format("Argument %s to option -%s starts with invalid character -.", arg, opt))
            print_stderr("Did you forget an argument to an option?")
            os.exit(1)
        end
    end
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif opt == "c" or opt == "cpus"then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
    elseif opt == "l" or opt == "list" then
        listFeatures = true
    elseif opt == "a" or opt == "all" then
        if cpuinfo["isIntel"] == 0 then
            print_stdout("INFO: Manipulation of CPU features is only available on Intel platforms")
        end
        print_stdout("Available features:")
        for i=0,likwid.tablelength(likwid.cpuFeatures)-1 do
            local found = false
            for j, f in pairs(FEATURES) do
                if likwid.cpuFeatures[i] == f then
                    found = true
                    break
                end
            end
            if found then
                print_stdout(string.format("\t%s*",likwid.cpuFeatures[i]))
            else
                print_stdout(string.format("\t%s",likwid.cpuFeatures[i]))
            end
        end
        print_stdout("Modifiable features are marked with *")
        os.exit(0)
    elseif opt == "e" or opt == "enable" then
        local tmp = likwid.stringsplit(arg, ",")
        for i, f in pairs(tmp) do
            for i=0,likwid.tablelength(likwid.cpuFeatures)-1 do
                if likwid.cpuFeatures[i] == f then
                    table.insert(enableList, i)
                end
            end
        end
    elseif opt == "d" or opt == "disable" then
        local tmp = likwid.stringsplit(arg, ",")
        for i, f in pairs(tmp) do
            for i=0,likwid.tablelength(likwid.cpuFeatures)-1 do
                if likwid.cpuFeatures[i] == f then
                    table.insert(disableList, i)
                end
            end
        end
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end

if cpuinfo["isIntel"] == 0 then
    print_stdout("INFO: Manipulation of CPU features is only available on Intel platforms")
    os.exit(0)
end

likwid.initCpuFeatures()

if listFeatures and #cpulist > 0 then
    if likwid.getCpuFeatures(c, 0) >= 0 then
        local str = "Feature"..string.rep(" ",string.len("BRANCH_TRACE_STORAGE")-string.len("Feature")+2)
        for j, c in pairs(cpulist) do
            str = str..string.format("CPU %d\t",c)
        end
        print_stdout(str)
        str = ""
        for i=0,likwid.tablelength(likwid.cpuFeatures)-1 do
            str = likwid.cpuFeatures[i]..string.rep(" ",string.len("BRANCH_TRACE_STORAGE")-string.len(likwid.cpuFeatures[i])+2)
            for j, c in pairs(cpulist) do
                if (likwid.getCpuFeatures(c, i) == 1) then
                    str = str .. "on\t"
                else
                    str = str .. "off\t"
                end
            end
            print_stdout(str)
        end
    end
    os.exit(1)
elseif #cpulist == 0 then
    print_stderr("Need CPU to list current feature state")
    os.exit(1)
end

if #enableList > 0 and #disableList > 0 then
    for i,e in pairs(enableList) do
        for j, d in pairs(disableList) do
            if (e == d) then
                print_stderr(string.format("Feature %s is in enable and disable list, doing nothing for feature", e))
                table.insert(skipList, e)
            end
        end
    end
    for i, s in pairs(skipList) do
        for j, e in pairs(enableList) do
            if (s == e) then table.remove(enableList, j) end
        end
        for j, e in pairs(disableList) do
            if (s == e) then table.remove(disableList, j) end
        end
    end
end

if #enableList > 0 then
    for i, c in pairs(cpulist) do
        for j, f in pairs(enableList) do
            local ret = likwid.enableCpuFeatures(c, f, 1)
            if ret == 0 then
                print_stdout(string.format("Enabled %s for CPU %d", likwid.cpuFeatures[f], c))
            else
                print_stdout(string.format("Failed %s for CPU %d", likwid.cpuFeatures[f], c))
            end
        end
    end
end
if #disableList > 0 then
    for i, c in pairs(cpulist) do
        for j, f in pairs(disableList) do
            local ret = likwid.disableCpuFeatures(c, f, 1)
            if ret == 0 then
                print_stdout(string.format("Disabled %s for CPU %d", likwid.cpuFeatures[f], c))
            else
                print_stdout(string.format("Failed %s for CPU %d", likwid.cpuFeatures[f], c))
            end
        end
    end
end
