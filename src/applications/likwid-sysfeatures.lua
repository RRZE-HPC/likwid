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


print_stdout = print
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end


function version()
    print_stdout(string.format("likwid-features -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

function example()
    print_stdout("Options:")
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
    print_stdout("-g, --get <feature(s)>\t Get the value of the given feature(s)")
    print_stdout("                      \t feature format: <category>.<name> or just <name> if unique")
    print_stdout("                      \t can be a list of features")
    print_stdout("-s, --set <feature(s)>\t Set feature(s) to the given value")
    print_stdout("                      \t format: <category>.<name>=<value> or just <name>=<value> if unique")
    print_stdout("                      \t can be a list of modifications")
    print_stdout("-O\t\t\t Output results in CSV")
    print_stdout("-V, --verbose <level>\t Set verbosity\n")
end

if #arg == 0 then
    usage()
    os.exit(0)
end

-- main variables with defaults
local listFeatures = false
local allFeatures = false
local num_hwts = 0
local hwtlist = {}
local getList = {}
local setList = {}
local cpuinfo = likwid.getCpuInfo()
local cputopo = likwid.getCpuTopology()
local affinity = likwid.getAffinityInfo()
local verbose = 0
local output_csv = false

-- parse the command line
for opt,arg in likwid.getopt(arg, {"h","v","l","c:","g:","s:","a", "O","help","version","list", "set:", "get:","all", "cpus:", "V:", "verbose:"}) do
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
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
    elseif opt == "a" or opt == "all" then
        allFeatures = true
    elseif opt == "g" or opt == "get" then
        getList = likwid.stringsplit(arg, ",")
    elseif opt == "s" or opt == "set" then
        setList = likwid.stringsplit(arg, ",")
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end


-- validate command line input
if (not listFeatures) and (not allFeatures) and (#getList == 0) and (#setList == 0) then
    print_stderr("No operations specified, exiting...")
    os.exit(1)
end
if (listFeatures or allFeatures) and (#getList > 0 or #setList > 0) then
    print_stderr("Cannot list features and get/set at the same time")
    os.exit(1)
end
if #hwtlist == 0 then
    if listFeatures then
        print_stderr("HWThread selection (-c) required for listing the state of all features")
        os.exit(1)
    elseif #getList > 0 then
        print_stderr("HWThread selection (-c) required for getting the state of given features")
        os.exit(1)
    elseif #setList > 0 then
        print_stderr("HWThread selection (-c) required for setting the state of given features")
        os.exit(1)
    end
end

--[[local access_mode = likwid.getAccessClientMode()
if access_mode < 0 or access_mode > 1 then
    print_stderr("Manipulation of HW features only for access mode 'direct' or 'accessdaemon'")
    os.exit(1)
end]]

-- set verbosity
if verbose > 0 and verbose <= 3 then
    likwid.setVerbosity(verbose)
end

-- initialize the sysfeatures module
local err = likwid.initSysFeatures()
if err < 0 then
    print_stderr("Cannot initialize HW features module")
    os.exit(1)
end

-- get a list of all features for the system
local list = likwid.sysFeatures_list()

-- print the list
if allFeatures then
    local all = {}
    local names = {}
    local types = {}
    local access = {}
    local descs = {}
    local cats = {}
    -- create a table of categories for sorting
    for _,f in pairs(list) do
        found = false
        for _, c in pairs(cats) do
            if c == f.Category then
                found = true
            end
        end
        if not found then
            table.insert(cats, f.Category)
        end
    end
    table.sort(cats)

    -- prepare table for output
    -- first we create tables for the four columns
    table.insert(names, "Feature")
    table.insert(types, "Scope")
    table.insert(access, "Access")
    table.insert(descs, "Description")
    for _,c in pairs(cats) do
        for _,f in pairs(list) do
            if f.Category == c then
                table.insert(names, string.format("%s.%s", f.Category, f.Name))
                table.insert(types, f.Type)
                table.insert(descs, f.Description)
                if f.ReadOnly then
                    table.insert(access, "rdonly")
                elseif f.WriteOnly then
                    table.insert(access, "wronly")
                else
                    table.insert(access, "rw")
                end
            end
        end
    end
    -- add all columns to the table
    setmetatable(names, {align = "left"})
    table.insert(all, names)
    table.insert(all, types)
    table.insert(all, access)
    setmetatable(descs, {align = "left"})
    table.insert(all, descs)

    -- print the table in selected format
    if output_csv then
        likwid.printcsv(all, #all)
    else
        print_stdout("Available features:")
        likwid.printtable(all)
    end

    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end


-- sysfeatures requires exact specification of devices
-- some exist per hw thread others only for the whole node
-- or the socket
-- this creates all devices based on the given hw threads
local deviceTree = {}

-- there is only a single entry for the whole node
-- the first entry in the list of hw threads is responsible
deviceTree[likwid.node] = {}
table.insert(deviceTree[likwid.node], {device = likwid.createDevice(likwid.node, hwtlist[1]), id = hwtlist[1]})
-- create the devices for all hw threads in the list
deviceTree[likwid.hwthread] = {}
for i, c in pairs(hwtlist) do
    table.insert(deviceTree[likwid.hwthread], {device = likwid.createDevice(likwid.hwthread, c), id = c})
end
-- create the devices for all sockets which are covered by the list of given hw threads
deviceTree[likwid.socket] = {}
for sid=0, cputopo.numSockets do
    local sdone = false
    for _, c in pairs(hwtlist) do
        for _, t in pairs(cputopo.threadPool) do
            if t.apicId == c and t.packageId == sid then
                table.insert(deviceTree[likwid.socket], {device = likwid.createDevice(likwid.socket, sid), id = c})
                sdone = true
                break
            end
        end
        if sdone then
            break
        end
    end
end

-- I don't know what I'm doing but I'm trying to populate the device tree according to cores
deviceTree[likwid.core] = {}
coresAdded = {}
for _, c in pairs(hwtlist) do
    for _, t in pairs(cputopo.threadPool) do
        if t.apicId == c and not (coresAdded[t.coreId] ~= Nil) then
            table.insert(deviceTree[likwid.core], {device = likwid.createDevice(likwid.core, t.coreId), id = c})
            coresAdded[t.coreId] = true
        end
    end
end

local function getDevice(level, id)
    if deviceTree[level] then
        for _, entry in pairs(deviceTree[level]) do
            if entry.id == id then
                return entry.device
            end
        end
    end
    return nil
end

--[[deviceTree[likwid.core] = {}
for cid=0, tonumber(cputopo.numCores) do
    for _, c in pairs(hwtlist) do
        for _, t in pairs(cputopo.threadPool) do
            if t.apicId == c and t.coreId == cid then
                table.insert(deviceTree[likwid.core], {device = likwid.createDevice(likwid.core, cid), id = c})
            end
        end
    end
end]]


if listFeatures and #hwtlist > 0 then
    -- prepare output table
    local all = {}
    -- first column contains the feature category and name
    local first = {}
    table.insert(first, "Feature/HWT")
    for _,f in pairs(list) do
        table.insert(first, string.format("%s.%s", f.Category, f.Name))
    end
    setmetatable(first, {align = "left"})
    table.insert(all, first)
    -- create one column per given hw thread with the current value of the feature
    for i, c in pairs(hwtlist) do
        local tab = {}
        table.insert(tab, string.format("HWThread %d", c))
        for _,f in pairs(list) do
            local dev = getDevice(f.TypeID, c)
            if dev then
                local v = likwid.sysFeatures_get(f.Name, dev)
                if v == nil then
                    table.insert(tab, "fail")
                else
                    table.insert(tab, v)
                end
            else
                table.insert(tab, "-")
            end
        end
        -- add the hw thread column to the table
        table.insert(all, tab)
    end

    -- print the table
    if output_csv then
        likwid.printcsv(all, num_hwts + 1)
    else
        likwid.printtable(all)
    end
    -- cleanup device tree before exiting
    for l, ltab in pairs(deviceTree) do
        for _, e in pairs(ltab) do
            likwid.destroyDevice(e.device)
        end
    end
    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end

if #getList > 0 and #hwtlist > 0 then
    -- filter the full feature list to have only the selected ones
    -- the new list entry contains all entries of the original feature entry plus
    -- the user-given value
    local featList = {}
    for _, f in pairs(getList) do
        for _, l in pairs(list) do
            if f == l.Name or f == string.format("%s.%s", l.Category, l.Name) then
                table.insert(featList, {
                    TypeID = l.TypeID,
                    Type = l.Type,
                    Name = l.Name,
                    Category = l.Category,
                    ReadOnly = l.ReadOnly,
                    WriteOnly = l.WriteOnly,
                })
            end
        end
    end

    -- get all features in the new list
    for i, c in pairs(hwtlist) do
        local tab = {}
        for _,f in pairs(featList) do
            -- get device from the device tree
            local dev = getDevice(f.TypeID, c)
            if dev then
                local v = likwid.sysFeatures_get(f.Name, dev)
                if not v then
                    print_stderr(string.format("Failed to get feature '%s.%s' (Type %s, Resp %d)", f.Category, f.Name, f.Type, c))
                else
                    print_stdout(v)
                end
            end
        end
    end
    -- cleanup device tree before exiting
    for l, ltab in pairs(deviceTree) do
        for _, e in pairs(ltab) do
            likwid.destroyDevice(e.device)
        end
    end
    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end

if #setList > 0 and #hwtlist > 0 then
    -- filter the full feature list to have only the selected ones
    -- the new list entry contains all entries of the original feature entry plus
    -- the user-given value
    local featList = {}
    for _, f in pairs(setList) do
        local t = likwid.stringsplit(f, "=")
        if #t == 2 then
            for _, l in pairs(list) do
                if t[1] == l.Name or t[1] == string.format("%s.%s", l.Category, l.Name) then
                    table.insert(featList, {
                        TypeID = l.TypeID,
                        Type = l.Type,
                        Name = l.Name,
                        Category = l.Category,
                        ReadOnly = l.ReadOnly,
                        WriteOnly = l.WriteOnly,
                        Value = t[2]
                    })
                end
            end
        else
            print_stderr(string.format("Invalid format of '%s' in set list", f))
        end
    end

    -- set all features in the new list
    for i, c in pairs(hwtlist) do
        local tab = {}
        for _,f in pairs(featList) do
            -- get device from the device tree
            local dev = getDevice(f.TypeID, c)
            if dev then
                if verbose > 0 then
                    print_stdout(string.format("Setting '%s.%s' to '%s' (Type %s, Resp %d)", f.Category, f.Name, f.Value, f.Type, c))
                end
                local v = likwid.sysFeatures_set(f.Name, dev, f.Value)
                if not v then
                    print_stderr(string.format("Failed to set feature '%s.%s' to '%s' (Type %s, Resp %d)", f.Category, f.Name, f.Value, f.Type, c))
                end
            end
        end
    end
    -- cleanup device tree before exiting
    for l, ltab in pairs(deviceTree) do
        for _, e in pairs(ltab) do
            likwid.destroyDevice(e.device)
        end
    end
    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end


--[[if (not (listFeatures or allFeatures)) or #enableList > 0 or #disableList > 0 then
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
            if f.Name == e and not f.ReadOnly then
                table.insert(realEnableList, f.Name)
            end
        end
        for _, e in pairs(disableList) do
            if f.Name == e and not f.ReadOnly then
                table.insert(realDisableList, f.Name)
            end
        end
    end

    -- First enable all features for all selected hardware threads
    if #realEnableList > 0 then
        for i, c in pairs(hwtlist) do
            local dev = likwid.createDevice(likwid.hwthread, c)
            for j, f in pairs(realEnableList) do
                local ret = likwid.sysFeatures_set(f, dev, 1)
                if ret == true then
                    print_stdout(string.format("Enabled %s for HWThread %d", f, c))
                else
                    print_stdout(string.format("Failed %s for HWThread %d", f, c))
                end
            end
            likwid.destroyDevice(dev)
        end
    end
    -- Next disable all features for all selected hardware threads
    if #realDisableList > 0 then
        for i, c in pairs(hwtlist) do
            local dev = likwid.createDevice(likwid.hwthread, c)
            for j, f in pairs(realDisableList) do
                local ret = likwid.sysFeatures_set(f, dev, 0)
                if ret == true then
                    print_stdout(string.format("Disabled %s for HWThread %d", f, c))
                else
                    print_stdout(string.format("Failed %s for HWThread %d", f, c))
                end
            end
            likwid.destroyDevice(dev)
        end
    end
end]]
likwid.finalizeSysFeatures()
os.exit(0)
