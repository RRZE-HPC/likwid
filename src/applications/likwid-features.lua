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
local cpuinfo = likwid.getCpuInfo()

print_stdout = print
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end


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
    print_stdout("-l, --list\t\t List features and state for given hardware threads")
    print_stdout("-c, --cpus <list>\t Perform operations on given hardware threads")
    print_stdout("-e, --enable <list>\t List of features that should be enabled")
    print_stdout("-d, --disable <list>\t List of features that should be disabled")
    print_stdout("-O\t\t Output results in CSV")
end

if #arg == 0 then
    usage()
    os.exit(0)
end

local listFeatures = false
local allFeatures = false
local num_hwts = 0
local hwtlist = {}
local enableList = {}
local disableList = {}

local output_csv = false

for opt,arg in likwid.getopt(arg, {"h","v","l","c:","e:","d:","a", "O","help","version","list", "enable:", "disable:","all", "cpus:"}) do
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
        num_hwts, hwtlist = likwid.cpustr_to_cpulist(arg)
    elseif opt == "l" or opt == "list" then
        listFeatures = true
    elseif opt == "O" then
        output_csv = true
    elseif opt == "a" or opt == "all" then
        allFeatures = true
    elseif opt == "e" or opt == "enable" then
        enableList = likwid.stringsplit(arg, ",")
    elseif opt == "d" or opt == "disable" then
        disableList = likwid.stringsplit(arg, ",")
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end



if (not listFeatures) and (not allFeatures) and (#enableList == 0) and (#disableList == 0) then
    print_stderr("No operations specified, exiting...")
    os.exit(1)
end
if (listFeatures or allFeatures) and (#enableList > 0 or #disableList > 0) then
    print_stderr("Cannot list features and enable/disable at the same time")
    os.exit(1)
end
if listFeatures and #hwtlist == 0 then
    print_stderr("HWThread selection (-c) required for listing the state of all features")
    os.exit(1)
end
local access_mode = likwid.getAccessClientMode()
if access_mode < 0 or access_mode > 1 then
    print_stderr("Manipulation of HW features only for access mode 'direct' or 'accessdaemon'")
    os.exit(1)
end
local err = likwid.initHWFeatures()
if err < 0 then
    print_stderr("Cannot initialize HW features module")
    os.exit(1)
end

local list = likwid.hwFeatures_list()

if allFeatures then
    local all = {}
    local names = {}
    local scopes = {}
    local access = {}
    local descs = {}
    table.insert(names, "Feature")
    table.insert(scopes, "Scope")
    table.insert(access, "Access")
    table.insert(descs, "Description")
    for _,f in pairs(list) do
        table.insert(names, f["Name"])
        table.insert(scopes, f["Scope"])
        table.insert(descs, f["Description"])
        if f["ReadOnly"] then
            table.insert(access, "ro")
        elseif f["WriteOnly"] then
            table.insert(access, "wo")
        else
            table.insert(access, "rw")
        end
    end
    table.insert(all, names)
    table.insert(all, scopes)
    table.insert(all, access)
    table.insert(all, descs)
    if output_csv then
        likwid.printcsv(all, #all)
    else
        print_stdout("Available features:")
        likwid.printtable(all)
    end

end


if (not allFeatures) and listFeatures and #hwtlist > 0 then
    local all = {}
    local first = {}
    table.insert(first, "Feature/HWT")
    for _,f in pairs(list) do
        table.insert(first, f["Name"])
    end
    table.insert(all, first)
    for i, c in pairs(hwtlist) do
        local tab = {}
        table.insert(tab, string.format("%d", c))
        for _,f in pairs(list) do
            local v = likwid.hwFeatures_get(f["Name"], c)
            if v == nil then
                table.insert(tab, "wronly")
            elseif v == 1 then
                table.insert(tab, "on")
            else
                table.insert(tab, "off")
            end
        end
        table.insert(all, tab)
    end
    
    if output_csv then
        likwid.printcsv(all, num_hwts + 1)
    else
        likwid.printtable(all)
    end
end

if (not (listFeatures or allFeatures)) or #enableList > 0 or #disableList > 0 then
    -- Check whether there are similar entries in enable and distable list and remove them (first add to skip list, then remove from tables)
    if #enableList > 0 and #disableList > 0 then
        local skipList = {}
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

    -- Filter enable and disable lists to contain only valid and writable features
    local realEnableList = {}
    local realDisableList = {}
    for _, f in pairs(list) do
        for _, e in pairs(enableList) do
            if f["Name"] == e and not f["ReadOnly"] then
                table.insert(realEnableList, f["Name"])
            end
        end
        for _, e in pairs(disableList) do
            if f["Name"] == e and not f["ReadOnly"] then
                table.insert(realDisableList, f["Name"])
            end
        end
    end

    -- First enable all features for all selected hardware threads
    if #realEnableList > 0 then
        for i, c in pairs(hwtlist) do
            for j, f in pairs(realEnableList) do
                local ret = likwid.hwFeatures_set(f, c, 1)
                if ret == true then
                    print_stdout(string.format("Enabled %s for HWThread %d", f, c))
                else
                    print_stdout(string.format("Failed %s for HWThread %d", f, c))
                end
            end
        end
    end
    -- Next disable all features for all selected hardware threads
    if #realDisableList > 0 then
        for i, c in pairs(hwtlist) do
            for j, f in pairs(realDisableList) do
                local ret = likwid.hwFeatures_set(f, c, 0)
                if ret == true then
                    print_stdout(string.format("Disabled %s for HWThread %d", f, c))
                else
                    print_stdout(string.format("Failed %s for HWThread %d", f, c))
                end
            end
        end
    end
end
likwid.finalizeHWFeatures()
os.exit(0)
