--[[
 * =======================================================================================
 *
 *      Filename:  likwid.lua
 *
 *      Description:  Lua LIKWID interface library
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@gmail.com
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

local likwid = {}
package.cpath = '<PREFIX>/lib/?.so;' .. package.cpath
require("liblikwid")
require("math")

likwid.groupfolder = "<PREFIX>/share/likwid/perfgroups"

likwid.version = <VERSION>
likwid.release = <RELEASE>
likwid.pinlibpath = "<PREFIX>/lib/liblikwidpin.so"
likwid.dline = string.rep("=",80)
likwid.hline =  string.rep("-",80)
likwid.sline = string.rep("*",80)



likwid.getConfiguration = likwid_getConfiguration
likwid.putConfiguration = likwid_putConfiguration
likwid.setAccessClientMode = likwid_setAccessClientMode
likwid.init = likwid_init
likwid.addEventSet = likwid_addEventSet
likwid.setupCounters = likwid_setupCounters
likwid.startCounters = likwid_startCounters
likwid.stopCounters = likwid_stopCounters
likwid.readCounters = likwid_readCounters
likwid.switchGroup = likwid_switchGroup
likwid.finalize = likwid_finalize
likwid.getEventsAndCounters = likwid_getEventsAndCounters
likwid.getResult = likwid_getResult
likwid.getNumberOfGroups = likwid_getNumberOfGroups
likwid.getRuntimeOfGroup = likwid_getRuntimeOfGroup
likwid.getIdOfActiveGroup = likwid_getIdOfActiveGroup
likwid.getNumberOfEvents = likwid_getNumberOfEvents
likwid.getNumberOfThreads = likwid_getNumberOfThreads
likwid.getCpuInfo = likwid_getCpuInfo
likwid.getCpuTopology = likwid_getCpuTopology
likwid.putTopology = likwid_putTopology
likwid.getNumaInfo = likwid_getNumaInfo
likwid.putNumaInfo = likwid_putNumaInfo
likwid.setMemInterleaved = likwid_setMemInterleaved
likwid.getAffinityInfo = likwid_getAffinityInfo
likwid.putAffinityInfo = likwid_putAffinityInfo
likwid.getPowerInfo = likwid_getPowerInfo
likwid.putPowerInfo = likwid_putPowerInfo
likwid.getOnlineDevices = likwid_getOnlineDevices
likwid.printSupportedCPUs = likwid_printSupportedCPUs
likwid.getCpuClock = likwid_getCpuClock
likwid.startClock = likwid_startClock
likwid.stopClock = likwid_stopClock
likwid.getClockCycles = likwid_getClockCycles
likwid.getClock = likwid_getClock
likwid.sleep = sleep
likwid.usleep = usleep
likwid.startPower = likwid_startPower
likwid.stopPower = likwid_stopPower
likwid.calcPower = likwid_printEnergy
likwid.getPowerLimit = likwid_powerLimitGet
likwid.setPowerLimit = likwid_powerLimitSet
likwid.statePowerLimit = likwid_powerLimitState
likwid.initTemp = likwid_initTemp
likwid.readTemp = likwid_readTemp
likwid.memSweep = likwid_memSweep
likwid.memSweepDomain = likwid_memSweepDomain
likwid.pinProcess = likwid_pinProcess
likwid.setenv = likwid_setenv
likwid.getpid = likwid_getpid
likwid.setVerbosity = likwid_setVerbosity
likwid.access = likwid_access
likwid.startProgram = likwid_startProgram
likwid.checkProgram = likwid_checkProgram
likwid.killProgram = likwid_killProgram
likwid.catchSignal = likwid_catchSignal
likwid.getSignalState = likwid_getSignalState

infinity = math.huge


local function getopt(args, ostrlist)
    local arg, place,placeend = nil, 0, 0;
    return function ()
        if place == 0 then -- update scanning pointer
            place = 1
            if #args == 0 or args[1]:sub(1, 1) ~= '-' then place = 0; return nil end
            if #args[1] >= 2 then
                if args[1]:sub(2, 2) == '-' then
                    if #args[1] == 2 then -- found "--"
                        place = 0
                        table.remove(args, 1)
                        return args[1], nil
                    end
                    place = place + 1
                end
                if args[1]:sub(3, 3) == '-' then
                    place = 0
                    table.remove(args, 1)
                    return args[1], nil
                end
                place = place + 1
                placeend = #args[1]
            end
        end
        local optopt = args[1]:sub(place, placeend)
        place = place + 1;
        local givopt = ""
        local needarg = false
        for _, ostr in pairs(ostrlist) do
            local matchstring = "^"..ostr.."$"
            placeend = place + #ostr -1
            if ostr:sub(#ostr,#ostr) == ":" then
                matchstring = "^"..ostr:sub(1,#ostr-1).."$"
                needarg = true
                placeend = place + #ostr -2
            end
            if optopt:match(matchstring) then
                givopt = ostr
                break
            end
            needarg = false
        end
        if givopt == "" then -- unknown option
            if optopt == '-' then return nil end
            if place > #args[1] then
                table.remove(args, 1)
                place = 0;
            end
            return '?',  optopt;
        end

        if not needarg then -- do not need argument
            arg = true;
            table.remove(args, 1)
            place = 0;
        else -- need an argument
            if placeend < #args[1] then -- no white space
                arg = args[1]:sub(placeend,#args[1])
            else
                table.remove(args, 1);
                if #args == 0 then -- an option requiring argument is the last one
                    place = 0
                    if givopt:sub(placeend, placeend) == ':' then return ':' end
                    return '?', optopt
                else arg = args[1] end
            end
            table.remove(args, 1)
            place = 0;
        end
        return optopt, arg
    end
end


likwid.getopt = getopt

local function tablelength(T)
    local count = 0
    if T == nil then return count end
    if type(T) ~= "table" then return count end
    for _ in pairs(T) do count = count + 1 end
    return count
end

likwid.tablelength = tablelength

local function tableprint(T, long)
    if T == nil or type(T) ~= "table" or tablelength(T) == 0 then
        print("[]")
        return
    end
    local start_index = 0
    local end_index = #T
    if T[start_index] == nil then
        start_index = 1
        end_index = #T
    end
    outstr = ""
    if T[start_index] ~= nil then
        for i=start_index,end_index do
            if not long then
                outstr = outstr .. "," .. tostring(T[i])
            else
                outstr = outstr .. "," .. "[" .. tostring(i) .. "] = ".. tostring(T[i])
            end
        end
    else
        for k,v in pairs(T) do
            if not long then
                outstr = outstr .. "," .. tostring(v)
            else
                outstr = outstr .. "," .. "[" .. tostring(k) .. "] = ".. tostring(v)
            end
        end
    end
    print("["..outstr:sub(2,outstr:len()).."]")
end

likwid.tableprint = tableprint

local function get_spaces(str, min_space, max_space)
    local length = str:len()
    local back = 0
    local front = 0
    back = math.ceil((max_space-str:len()) /2)
    front = max_space - back - str:len()

    if (front < back) then
        local tmp = front
        front = back
        back = tmp
    end
    return string.rep(" ", front),string.rep(" ", back)
end

local function calculate_metric(formula, counters_to_values)
    local function cmp(a,b)
        if a:len() > b:len() then return true end
        return false
    end
    local result = "Nan"
    local err = false
    local clist = {}
    for counter,value in pairs(counters_to_values) do
        table.insert(clist, counter)
    end
    table.sort(clist, cmp)
    for _,counter in pairs(clist) do
        formula = string.gsub(formula, tostring(counter), tostring(counters_to_values[counter]))
    end
    for c in formula:gmatch"." do
        if c ~= "+" and c ~= "-" and  c ~= "*" and  c ~= "/" and c ~= "(" and c ~= ")" and c ~= "." and c:lower() ~= "e" then
            local tmp = tonumber(c)
            if type(tmp) ~= "number" then
                print(c,tmp)
                print("Not all formula entries can be substituted with measured values")
                print("Current formula: "..formula)
                err = true
                break
            end
        end
    end
    if not err then
        if formula then
            result = assert(loadstring("return (" .. formula .. ")")())
            if (result == nil or result ~= result or result == infinity or result == -infinity) then
                result = 0
            end
        else
            result = 0
        end
    end
    return result
end

likwid.calculate_metric = calculate_metric

local function printtable(tab)
    local nr_columns = tablelength(tab)
    if nr_columns == 0 then
        print("Table has no columns. Empty table?")
        return
    end
    local nr_lines = tablelength(tab[1])
    local min_lengths = {}
    local max_lengths = {}
    for i, col in pairs(tab) do
        if tablelength(col) ~= nr_lines then
            print("Not all columns have the same row count, nr_lines"..tostring(nr_lines)..", current "..tablelength(col))
            return
        end
        if min_lengths[i] == nil then
            min_lengths[i] = 10000000
            max_lengths[i] = 0
        end
        for j, field in pairs(col) do
            if tostring(field):len() > max_lengths[i] then
                max_lengths[i] = tostring(field):len()
            end
            if tostring(field):len() < min_lengths[i] then
                min_lengths[i] = tostring(field):len()
            end
        end
    end
    hline = ""
    for i=1,#max_lengths do
        hline = hline .. "+-"..string.rep("-",max_lengths[i]).."-"
    end
    hline = hline .. "+"
    print(hline)
    
    str = "| "
    for i=1,nr_columns do
        front, back = get_spaces(tostring(tab[i][1]), min_lengths[i],max_lengths[i])
        str = str .. front.. tostring(tab[i][1]) ..back.. " | "
    end
    print(str)
    print(hline)
    
    for j=2,nr_lines do
        str = "| "
        for i=1,nr_columns do
            front, back = get_spaces(tostring(tab[i][j]), min_lengths[i],max_lengths[i])
            str = str .. front.. tostring(tab[i][j]) ..back.. " | "
        end
        print(str)
    end
    if nr_lines > 1 then
        print(hline)
    end
    print()
end

likwid.printtable = printtable

local function printcsv(tab, linelength)
    local nr_columns = tablelength(tab)
    if nr_columns == 0 then
        print("Table has no columns. Empty table?")
        return
    end
    local nr_lines = tablelength(tab[1])
    local str = ""
    for j=1,nr_lines do
        str = ""
        for i=1,nr_columns do
            str = str .. tostring(tab[i][j])
            if (i ~= nr_columns) then
                str = str .. ","
            end
        end
        if nr_columns < linelength then
            str = str .. string.rep(",", linelength-nr_columns)
        end
        print(str)
    end
    
end

likwid.printcsv = printcsv

local function stringsplit(astr, sSeparator, nMax, bRegexp)
    assert(sSeparator ~= '')
    assert(nMax == nil or nMax >= 1)
    if astr == nil then return {} end
    local aRecord = {}

    if astr:len() > 0 then
        local bPlain = not bRegexp
        nMax = nMax or -1

        local nField=1 nStart=1
        local nFirst,nLast = astr:find(sSeparator, nStart, bPlain)
        while nFirst and nMax ~= 0 do
            aRecord[nField] = astr:sub(nStart, nFirst-1)
            nField = nField+1
            nStart = nLast+1
            nFirst,nLast = astr:find(sSeparator, nStart, bPlain)
            nMax = nMax-1
            end
        aRecord[nField] = astr:sub(nStart)
    end

    return aRecord
end

likwid.stringsplit = stringsplit

local function cpulist_sort(cpulist)
    local newlist = {}
    if #cpulist == 0 then
        return newlist
    end
    local topo = likwid_getCpuTopology()
    for offset=1,topo["numThreadsPerCore"] do
        for i=0, #cpulist/topo["numThreadsPerCore"] do
            table.insert(newlist, cpulist[(i*topo["numThreadsPerCore"])+offset])
        end
    end
    return newlist
end

local function cpulist_concat(cpulist, addlist)
    for i, add in pairs(addlist) do
        table.insert(cpulist, add)
    end
    return cpulist
end

local function cpustr_valid(cpustr)
    invalidlist = {"%.", "_", ";", "!", "§", "%$", "%%", "%&", "/", "\\",  "%(","%)","=", "?","`","´" ,"~","°","|","%^","<",">", "{","}","%[","%]","#","\'","\"", "*"}
    for i, inval in pairs(invalidlist) do
        local s,e = cpustr:find(inval)
        if s ~= nil then
            return false
        end
    end
    return true
end

local function cpustr_to_cpulist_scatter(cpustr)
    local cpulist = {}
    local domain_list = {}
    local domain_cpus = {}
    if not cpustr_valid(cpustr) then
        print("ERROR: Expression contains invalid characters")
        return {}
    end
    local s,e = cpustr:find(":")
    if s ~= nil then
        local domain = cpustr:sub(1,s-1)
        local expression = cpustr:sub(s+1,cpustr:len())
        local affinity = likwid_getAffinityInfo()
        local topo = likwid_getCpuTopology()

        for dom,content in pairs(affinity["domains"]) do
            s,e = content["tag"]:find(domain)
            if s ~= nil then 
                table.insert(domain_list, dom)
                table.insert(domain_cpus, cpulist_sort(affinity["domains"][dom]["processorList"]))
            end
        end

        local num_domains = tablelength(domain_list)
        local domain_idx = 1
        local threadID = 1
        -- Adding physical cores
        for i=1,topo["activeHWThreads"]/num_domains do
            for idx, _ in pairs(domain_list) do
                table.insert(cpulist, domain_cpus[idx][i])
            end
        end
    else
        print("ERROR: Cannot parse scatter expression, should look something like <domain>:scatter")
        return {}
    end
    return cpulist
end


local function cpustr_to_cpulist_expression(cpustr)
    local cpulist = {}
    if not cpustr_valid(cpustr) then
        print("ERROR: Expression contains invalid characters")
        return {}
    end
    local affinity = likwid_getAffinityInfo()
    local exprlist = stringsplit(cpustr, ":")
    table.remove(exprlist, 1)
    local domain = 0

    local tag = "X"
    local count = 0
    local chunk = 1
    local stride = 1

    if #exprlist == 2 then
        tag = exprlist[1]
        count = tonumber(exprlist[2])
    elseif #exprlist == 4 then
        tag = exprlist[1]
        count = tonumber(exprlist[2])
        chunk = tonumber(exprlist[3])
        stride = tonumber(exprlist[4])
    end
    if tag == "X" or count == nil or chunk == nil or stride == nil then
        print("ERROR: Invalid expression, cannot parse all needed values")
        return {}
    end
    for domidx, domcontent in pairs(affinity["domains"]) do
        if domcontent["tag"] == tag then
            domain = domidx
            break
        end
    end
    if domain == 0 then
        print(string.format("ERROR: Invalid affinity domain %s", tag))
        return {}
    end

    index = 1
    selected = 0
    for i=1,count do
        for j=0, chunk-1 do
            table.insert(cpulist, affinity["domains"][domain]["processorList"][index+j])
            selected = selected+1
            if (selected >= count) then break end
        end
        index = index + stride
        if (index > affinity["domains"][domain]["numberOfProcessors"]) then
            index = 1
        end
        if (selected >= count) then break end
    end
    return cpulist
end


local function cpustr_to_cpulist_logical(cpustr)
    local cpulist = {}
    local sorted_list = {}
    if not cpustr_valid(cpustr) then
        print("ERROR: Expression contains invalid characters")
        return {}
    end
    local affinity = likwid_getAffinityInfo()
    local exprlist = stringsplit(cpustr, ":")
    table.remove(exprlist, 1)
    local domain = 0
    if #exprlist ~= 2 then
        print("ERROR: Invalid expression, should look like L:<domain>:<indexlist> or be in a cpuset")
        return {}
    end
    local tag = exprlist[1]
    local indexstr = exprlist[2]
    for domidx, domcontent in pairs(affinity["domains"]) do
        if domcontent["tag"] == tag then
            domain = domidx
            break
        end
    end
    if domain == 0 then
        print(string.format("ERROR: Invalid affinity domain %s", tag))
        return {}
    end
    sorted_list = cpulist_sort(affinity["domains"][domain]["processorList"])

    indexlist = stringsplit(indexstr, ",")
    for i, item in pairs(indexlist) do
        local s,e = item:find("-")
        if s == nil then
            local index = tonumber(item)+1
            if index > affinity["domains"][domain]["numberOfProcessors"] then
                print(string.format("CPU index %s larger than number of processors in affinity group %s", item, tag))
                return {}
            end
            table.insert(cpulist, sorted_list[index])
        else
            start, ende = item:match("(%d*)-(%d*)")
            start = start + 1
            ende = ende + 1
            if tonumber(start) == nil then
                print("ERROR: CPU indices smaller than 0 are not allowed")
                return {}
            end
            if tonumber(start) > tonumber(ende) then
                print(string.format("ERROR: CPU list %s invalid, start %s is larger than end %s", item, start, ende))
                return {}
            end
            if tonumber(ende) > #sorted_list then
                print(string.format("ERROR: CPU list end %d larger than number of processors in affinity group %s", ende, tag))
                return {}
            end
            for i=tonumber(start),tonumber(ende) do
                table.insert(cpulist, sorted_list[i])
            end
        end
    end
    return cpulist
end

local function cpustr_to_cpulist_physical(cpustr)
    local function present(list, check)
        for i, item in pairs(list) do
            if item == check then
                return true
            end
        end
        return false
    end
    local cpulist = {}
    if not cpustr_valid(cpustr) then
        print("ERROR: Expression contains invalid characters")
        return {}
    end
    local affinity = likwid_getAffinityInfo()
    local domain = 0
    tag, indexstr = cpustr:match("^(%g+):(%g+)")
    if tag == nil then
        tag = "N"
        indexstr = cpustr:match("^(%g+)")
    end
    for domidx, domcontent in pairs(affinity["domains"]) do
        if domcontent["tag"] == tag then
            domain = domidx
            break
        end
    end
    if domain == 0 then
        print(string.format("ERROR: Invalid affinity domain %s", tag))
        return {}
    end
    indexlist = stringsplit(indexstr, ",")
    for i, item in pairs(indexlist) do
        local s,e = item:find("-")
        if s == nil then
            if present(affinity["domains"][domain]["processorList"], tonumber(item)) then
                table.insert(cpulist, tonumber(item))
            else
                print(string.format("ERROR: CPU %s not in affinity domain %s", item, tag))
                return {}
            end
        else
            start, ende = item:match("^(%d*)-(%d*)")
            if tonumber(start) == nil then
                print("ERROR: CPU indices smaller than 0 are not allowed")
                return {}
            end
            if tonumber(ende) >= affinity["domains"][domain]["numberOfProcessors"] then
                print(string.format("ERROR: CPU list end %d larger than number of processors in affinity group %s", ende, tag))
                return {}
            end
            for i=tonumber(start),tonumber(ende) do
                if present(affinity["domains"][domain]["processorList"], i) then
                    table.insert(cpulist, i)
                else
                    print(string.format("ERROR: CPU %s not in affinity domain %s", i, tag))
                    return {}
                end
            end
        end
    end
    return cpulist
end

likwid.cpustr_to_cpulist_physical = cpustr_to_cpulist_physical


local function cpustr_to_cpulist(cpustr)
    local strlist = stringsplit(cpustr, "@")
    local topo = likwid_getCpuTopology()
    local cpulist = {}
    for pos, str in pairs(strlist) do
        if str:match("^%a*:scatter") then
            cpulist = cpulist_concat(cpulist, cpustr_to_cpulist_scatter(str))
        elseif str:match("^E:%a") then
            cpulist = cpulist_concat(cpulist, cpustr_to_cpulist_expression(str))
        elseif str:match("^L:%a") then
            cpulist = cpulist_concat(cpulist, cpustr_to_cpulist_logical(str))
        elseif topo["activeHWThreads"] < topo["numHWThreads"] then
            print(string.format("INFO: You are running LIKWID in a cpuset with %d CPUs, only logical numbering allowed",topo["activeHWThreads"]))
            if str:match("^N:") or str:match("^S%d*:") or str:match("^C%d*:") or str:match("^M%d*:") then
                cpulist = cpulist_concat(cpulist, cpustr_to_cpulist_logical("L:"..str))
            else
                cpulist = cpulist_concat(cpulist, cpustr_to_cpulist_logical("L:N:"..str))
            end
        elseif str:match("^N:") or str:match("^S%d*:") or str:match("^C%d*:") or str:match("^M%d*:") then
            cpulist = cpulist_concat(cpulist, cpustr_to_cpulist_logical("L:"..str))
        else
            local tmplist = cpustr_to_cpulist_physical(str)
            if tmplist == {} then
                print(string.format("ERROR: Cannot analyze string %s", str))
            else
                cpulist = cpulist_concat(cpulist, tmplist)
            end
        end
    end
    return tablelength(cpulist),cpulist
end

likwid.cpustr_to_cpulist = cpustr_to_cpulist

local function cpuexpr_to_list(cpustr, prefix)
    local cpulist = {}
    if not cpustr_valid(cpustr) then
        print("ERROR: Expression contains invalid characters")
        return 0, {}
    end
    local affinity = likwid_getAffinityInfo()
    local domain = 0
    local exprlist = stringsplit(cpustr,",")
    for i, expr in pairs(exprlist) do
        local added = false
        for domidx, domcontent in pairs(affinity["domains"]) do
            if domcontent["tag"] == prefix..expr then
                table.insert(cpulist, tonumber(expr))
                added = true
                break
            end
        end
        if not added then
            print(string.format("ERROR: No affinity domain with index %s%s", prefix, expr))
            return 0, {}
        end
    end
    return tablelength(cpulist),cpulist
end

local function nodestr_to_nodelist(cpustr)
    return cpuexpr_to_list(cpustr, "M")
end

likwid.nodestr_to_nodelist = nodestr_to_nodelist

local function sockstr_to_socklist(cpustr)
    return cpuexpr_to_list(cpustr, "S")
end

likwid.sockstr_to_socklist = sockstr_to_socklist

local function get_groups()
    groups = {}
    local cpuinfo = likwid.getCpuInfo()
    if cpuinfo == nil then return 0, {} end
    local f = io.popen("ls " .. likwid.groupfolder .. "/" .. cpuinfo["short_name"] .."/*.txt 2>/dev/null")
    if f == nil then
        print("Cannot read groups for architecture "..cpuinfo["short_name"])
        return 0, {}
    end
    t = stringsplit(f:read("*a"),"\n")
    f:close()
    for i, a in pairs(t) do
        if a ~= "" then
            table.insert(groups,a:sub((a:match'^.*()/')+1,a:len()-4))
        end
    end
    return #groups,groups
end

likwid.get_groups = get_groups

local function new_groupdata(eventString, fix_ctrs)
    local gdata = {}
    local num_events = 1
    gdata["Events"] = {}
    gdata["EventString"] = ""
    gdata["GroupString"] = ""
    local s,e = eventString:find(":")
    if s == nil then
        return gdata
    end
    if fix_ctrs > 0 then
        if not eventString:match("FIXC2") and fix_ctrs == 3 then
            eventString = "CPU_CLK_UNHALTED_REF:FIXC2,"..eventString
        end
        if not eventString:match("FIXC1") and fix_ctrs >= 2 then
            eventString = "CPU_CLK_UNHALTED_CORE:FIXC1,"..eventString
        end
        if not eventString:match("FIXC0") and fix_ctrs >= 1 then
            eventString = "INSTR_RETIRED_ANY:FIXC0,"..eventString
        end
    end
    gdata["EventString"] = eventString
    gdata["GroupString"] = eventString
    local eventslist = likwid.stringsplit(eventString,",")
    for i,e in pairs(eventslist) do
        eventlist = likwid.stringsplit(e,":")
        gdata["Events"][num_events] = {}
        gdata["Events"][num_events]["Event"] = eventlist[1]
        gdata["Events"][num_events]["Counter"] = eventlist[2]
        if #eventlist > 2 then
            table.remove(eventlist, 2)
            table.remove(eventlist, 1)
            gdata["Events"][num_events]["Options"] = eventlist
        end
        num_events = num_events + 1
    end
    return gdata
end


local function get_groupdata(group)
    groupdata = {}
    local group_exist = 0
    local cpuinfo = likwid.getCpuInfo()
    if cpuinfo == nil then return nil end

    num_groups, groups = get_groups()
    for i, a in pairs(groups) do
        if (a == group) then group_exist = 1 end
    end
    if (group_exist == 0) then return new_groupdata(group, cpuinfo["perf_num_fixed_ctr"]) end
    
    local f = assert(io.open(likwid.groupfolder .. "/" .. cpuinfo["short_name"] .. "/" .. group .. ".txt", "r"))
    local t = f:read("*all")
    f:close()
    local parse_eventset = false
    local parse_metrics = false
    local parse_long = false
    groupdata["EventString"] = ""
    groupdata["Events"] = {}
    groupdata["Metrics"] = {}
    groupdata["LongDescription"] = ""
    groupdata["GroupString"] = group
    nr_events = 1
    nr_metrics = 1
    for i, line in pairs(stringsplit(t,"\n")) do
        
        if (parse_eventset or parse_metrics or parse_long) and line:len() == 0 then
            parse_eventset = false
            parse_metrics = false
            parse_long = false
        end

        if line:match("^SHORT%a*") ~= nil then
            linelist = stringsplit(line, "%s+", nil, "%s+")
            table.remove(linelist, 1)
            groupdata["ShortDescription"] = table.concat(linelist, " ")  
        end

        if line:match("^EVENTSET$") ~= nil then
            parse_eventset = true
        end

        if line:match("^METRICS$") ~= nil then
            parse_metrics = true
        end

        if line:match("^LONG$") ~= nil then
            parse_long = true
        end

        if parse_eventset and line:match("^EVENTSET$") == nil then
            linelist = stringsplit(line:gsub("^%s*(.-)%s*$", "%1"), "%s+", nil, "%s+")
            eventstring = linelist[2] .. ":" .. linelist[1]
            if #linelist > 2 then
                table.remove(linelist,2)
                table.remove(linelist,1)
                eventstring = eventstring .. ":".. table.concat(":",linelist)
            end
            groupdata["EventString"] = groupdata["EventString"] .. "," .. eventstring
            groupdata["Events"][nr_events] = {}
            groupdata["Events"][nr_events]["Event"] = linelist[2]:gsub("^%s*(.-)%s*$", "%1")
            groupdata["Events"][nr_events]["Counter"] = linelist[1]:gsub("^%s*(.-)%s*$", "%1")
            nr_events = nr_events + 1
        end
        
        if parse_metrics and line:match("^METRICS$") == nil then
            linelist = stringsplit(line:gsub("^%s*(.-)%s*$", "%1"), "%s+", nil, "%s+")
            formula = linelist[#linelist]
            table.remove(linelist)
            groupdata["Metrics"][nr_metrics] = {}
            groupdata["Metrics"][nr_metrics]["description"] = table.concat(linelist, " ")
            groupdata["Metrics"][nr_metrics]["formula"] = formula
            nr_metrics = nr_metrics + 1
        end
        
        if parse_long and line:match("^LONG$") == nil then
            groupdata["LongDescription"] = groupdata["LongDescription"] .. "\n" .. line
        end
    end
    groupdata["LongDescription"] = groupdata["LongDescription"]:sub(2)
    groupdata["EventString"] = groupdata["EventString"]:sub(2)
    
    return groupdata
    
end

likwid.get_groupdata = get_groupdata




local function parse_time(timestr)
    local duration = 0
    local s1,e1 = timestr:find("ms")
    local s2,e2 = timestr:find("us")
    if s1 ~= nil then
        duration = tonumber(timestr:sub(1,s1-1)) * 1.E03
    elseif s2 ~= nil then
        duration = tonumber(timestr:sub(1,s2-1))
    else
        s1,e1 = timestr:find("s")
        if s1 == nil then
            print("Cannot parse time, '" .. timestr .. "' not well formatted, we need a time unit like s, ms, us")
            os.exit(1)
        end
        duration = tonumber(timestr:sub(1,s1-1)) * 1.E06
    end
    return duration
end

likwid.parse_time = parse_time



local function min_max_avg(values)
    min = math.huge
    max = 0.0
    sum = 0.0
    count = 0
    for _, value in pairs(values) do
        if value ~= nil then
            if (value < min) then min = value end
            if (value > max) then max = value end
            sum = sum + value
            count = count + 1
        end
    end
    return min, max, sum/count
end

local function tableMinMaxAvgSum(inputtable, skip_cols, skip_lines)
    local outputtable = {}
    local nr_columns = #inputtable
    if nr_columns == 0 then
        return {}
    end
    local nr_lines = #inputtable[1]
    if nr_lines == 0 then
        return {}
    end
    minOfLine = {"Min"}
    maxOfLine = {"Max"}
    sumOfLine = {"Sum"}
    avgOfLine = {"Avg"}
    for i=skip_lines+1,nr_lines do
        minOfLine[i-skip_lines+1] = math.huge
        maxOfLine[i-skip_lines+1] = 0
        sumOfLine[i-skip_lines+1] = 0
        avgOfLine[i-skip_lines+1] = 0
    end
    for j=skip_cols+1,nr_columns do
        for i=skip_lines+1, nr_lines do
            local res = tonumber(inputtable[j][i])
            if res ~= nil then
                minOfLine[i-skip_lines+1] = math.min(res, minOfLine[i-skip_lines+1])
                maxOfLine[i-skip_lines+1] = math.max(res, maxOfLine[i-skip_lines+1])
                sumOfLine[i-skip_lines+1] = sumOfLine[i-skip_lines+1] + res
            else
                minOfLine[i-skip_lines+1] = 0
                maxOfLine[i-skip_lines+1] = 0
                sumOfLine[i-skip_lines+1] = 0
            end
            avgOfLine[i-skip_lines+1] = sumOfLine[i-skip_lines+1]/(nr_columns-skip_cols)
        end
    end

    local tmptable = {}
    table.insert(tmptable, inputtable[1][1])
    for j=2,#inputtable[1] do
        table.insert(tmptable, inputtable[1][j].." STAT")
    end
    table.insert(outputtable, tmptable)
    for i=2,skip_cols do
        local tmptable = {}
        table.insert(tmptable, inputtable[i][1])
        for j=2,#inputtable[i] do
            table.insert(tmptable, inputtable[i][j])
        end
        table.insert(outputtable, tmptable)
    end
    table.insert(outputtable, sumOfLine)
    table.insert(outputtable, minOfLine)
    table.insert(outputtable, maxOfLine)
    table.insert(outputtable, avgOfLine)
    return outputtable
end

likwid.tableToMinMaxAvgSum = tableMinMaxAvgSum

local function printOutput(groups, results, groupData, cpulist)
    local nr_groups = #groups
    local maxLineFields = 0
    local cpuinfo = likwid_getCpuInfo()
    local clock = likwid.getCpuClock()
    for g, group in pairs(groups) do
        local groupID = group["ID"]
        local num_events = likwid_getNumberOfEvents(groupID);
        local num_threads = likwid_getNumberOfThreads(groupID-1);
        local runtime = likwid_getRuntimeOfGroup(groupID)
        local groupName = groupData[groupID]["GroupString"]
        if groupName == groupData[groupID]["EventString"] then
            groupName = "Custom"
        end
        local firsttab =  {}
        local firsttab_combined = {}
        local secondtab = {}
        local secondtab_combined = {}
        firsttab[1] = {"Event"}
        firsttab_combined[1] = {"Event"}
        firsttab[2] = {"Counter"}
        firsttab_combined[2] = {"Counter"}
        if not groupData[groupID]["Metrics"] then
            table.insert(firsttab[1],"Runtime (RDTSC) [s]")
            table.insert(firsttab[2],"TSC")
        end
        
        for i=1,num_events do
            table.insert(firsttab[1],groupData[groupID]["Events"][i]["Event"])
            table.insert(firsttab_combined[1],groupData[groupID]["Events"][i]["Event"] .. " STAT")
        end

        for i=1,num_events do
            table.insert(firsttab[2],groupData[groupID]["Events"][i]["Counter"])
            table.insert(firsttab_combined[2],groupData[groupID]["Events"][i]["Counter"])
        end
        

        for j=1,num_threads do
            tmpList = {"Core "..tostring(cpulist[j])}
            if not groupData[groupID]["Metrics"] then
                table.insert(tmpList, string.format("%e",runtime))
            end
            for i=1,num_events do
                local tmp = tostring(results[groupID][i][j])
                if tostring(results[groupID][i][j]):len() > 12 then
                    tmp = string.format("%e", results[groupID][i][j])
                end
                table.insert(tmpList, tmp)
            end
            table.insert(firsttab, tmpList)
        end
        
        if #cpulist > 1 then
            firsttab_combined = tableMinMaxAvgSum(firsttab, 2, 1)
        end

        if groupData[groupID]["Metrics"] then
            local counterlist = {}
            counterlist["time"] = runtime
            counterlist["inverseClock"] = 1.0/clock;

            secondtab[1] = {"Metric"}
            secondtab_combined[1] = {"Metric"}
            for m=1,#groupData[groupID]["Metrics"] do
                table.insert(secondtab[1],groupData[groupID]["Metrics"][m]["description"] )
                table.insert(secondtab_combined[1],groupData[groupID]["Metrics"][m]["description"].." STAT" )
            end
            for j=1,num_threads do
                tmpList = {"Core "..tostring(cpulist[j])}
                for i=1,num_events do
                    counterlist[groupData[groupID]["Events"][i]["Counter"]] = results[groupID][i][j]
                end
                for m=1,#groupData[groupID]["Metrics"] do
                    local tmp = calculate_metric(groupData[groupID]["Metrics"][m]["formula"], counterlist)
                    if tostring(tmp):len() > 12 then
                        tmp = string.format("%e",tmp)
                    end
                    table.insert(tmpList, tostring(tmp))
                end
                table.insert(secondtab,tmpList)
            end

            if #cpulist > 1 then
                secondtab_combined = tableMinMaxAvgSum(secondtab, 1, 1)
            end
        end
        maxLineFields = math.max(#firsttab, #firsttab_combined,
                                 #secondtab, #secondtab_combined)
        if use_csv then
            print(string.format("STRUCT,Info,3%s",string.rep(",",maxLineFields-3)))
            print(string.format("CPU name:,%s%s", cpuinfo["osname"],string.rep(",",maxLineFields-2)))
            print(string.format("CPU type:,%s%s", cpuinfo["name"],string.rep(",",maxLineFields-2)))
            print(string.format("CPU clock:,%s GHz%s", clock*1.E-09,string.rep(",",maxLineFields-2)))
            print(string.format("TABLE,Group %d Raw,%s,%d%s",groupID,groupName,#firsttab[1]-1,string.rep(",",maxLineFields-4)))
            likwid.printcsv(firsttab, maxLineFields)
        else
            if outfile ~= nil then
                print(likwid.hline)
                print(string.format("CPU name:\t%s",cpuinfo["osname"]))
                print(string.format("CPU type:\t%s",cpuinfo["name"]))
                print(string.format("CPU clock:\t%3.2f GHz",clock * 1.E-09))
                print(likwid.hline)
            end
            print("Group "..tostring(groupID)..": "..groupName)
            likwid.printtable(firsttab)
        end
        if #cpulist > 1 then
            if use_csv then
                print(string.format("TABLE,Group %d Raw Stat,%s,%d%s",groupID,groupName,#firsttab_combined[1]-1,string.rep(",",maxLineFields-4)))
                likwid.printcsv(firsttab_combined, maxLineFields)
            else
                likwid.printtable(firsttab_combined)
            end
        end
        if groupData[groupID]["Metrics"] then
            if use_csv then
                print(string.format("TABLE,Group %d Metric,%s,%d%s",groupID,groupName,#secondtab[1]-1,string.rep(",",maxLineFields-4)))
                likwid.printcsv(secondtab, maxLineFields)
            else
                likwid.printtable(secondtab)
            end
            if #cpulist > 1 then
                if use_csv then
                    print(string.format("TABLE,Group %d Metric Stat,%s,%d%s",groupID,groupName,#secondtab_combined[1]-1,string.rep(",",maxLineFields-4)))
                    likwid.printcsv(secondtab_combined, maxLineFields)
                else
                    likwid.printtable(secondtab_combined)
                end
            end
        end
    end
end


likwid.printOutput = printOutput

local function printMarkerOutput(groups, results, groupData, cpulist)
    local nr_groups = #groups
    local maxLineFields = 0
    local clock = likwid_getCpuClock();
    for g, group in pairs(groups) do
        local groupName = groupData[g]["GroupString"]
        if groupName == groupData[g]["EventString"] then
            groupName = "Custom"
        end
        for r, region in pairs(groups[g]) do
            local nr_threads = likwid.tablelength(groups[g][r]["Time"])
            local nr_events = likwid.tablelength(groupData[g]["Events"])
            if tablelength(groups[g][r]["Count"]) > 0 then

                local infotab = {}
                local firsttab = {}
                local firsttab_combined = {}
                local secondtab = {}
                local secondtab_combined = {}

                infotab[1] = {"Region Info","RDTSC Runtime [s]","call count"}
                for thread=1, nr_threads do
                    local tmpList = {}
                    table.insert(tmpList, "Core "..tostring(cpulist[thread]))
                    table.insert(tmpList, string.format("%.6f", groups[g][r]["Time"][thread]))
                    table.insert(tmpList, tostring(groups[g][r]["Count"][thread]))
                    table.insert(infotab, tmpList)
                end

                firsttab[1] = {"Event"}
                firsttab_combined[1] = {"Event"}
                for e=1,nr_events do
                    table.insert(firsttab[1],groupData[g]["Events"][e]["Event"])
                    table.insert(firsttab_combined[1],groupData[g]["Events"][e]["Event"].." STAT")
                end
                firsttab[2] = {"Counter"}
                firsttab_combined[2] = {"Counter"}
                for e=1,nr_events do
                    table.insert(firsttab[2],groupData[g]["Events"][e]["Counter"])
                    table.insert(firsttab_combined[2],groupData[g]["Events"][e]["Counter"])
                end
                for t=1,nr_threads do
                    local tmpList = {}
                    table.insert(tmpList, "Core "..tostring(cpulist[t]))
                    for e=1,nr_events do
                        local index = 0
                        local tmp = results[g][r][e][t]["Value"]
                        if tmp == nil then
                            tmp = 0
                        end
                        table.insert(tmpList, string.format("%e",tmp))
                    end
                    table.insert(firsttab, tmpList)
                end

                if #cpulist > 1 then
                    firsttab_combined = tableMinMaxAvgSum(firsttab, 2, 1)
                end


                if likwid.tablelength(groupData[g]["Metrics"]) > 0 then

                    tmpList = {"Metric"}
                    for m=1,#groupData[g]["Metrics"] do
                        table.insert(tmpList, groupData[g]["Metrics"][m]["description"])
                    end
                    table.insert(secondtab, tmpList)
                    for t=1,nr_threads do
                        counterlist = {}
                        for e=1,nr_events do
                            counterlist[ results[g][r][e][t]["Counter"] ] = results[g][r][e][t]["Value"]
                        end
                        counterlist["inverseClock"] = 1.0/clock
                        counterlist["time"] = groups[g][r]["Time"][t]
                        tmpList = {}
                        table.insert(tmpList, "Core "..tostring(cpulist[t]))
                        for m=1,#groupData[g]["Metrics"] do
                            local tmp = likwid.calculate_metric(groupData[g]["Metrics"][m]["formula"],counterlist)
                            if tmp == nil or tostring(tmp) == "-nan" then
                                tmp = "0"
                            elseif tostring(tmp):len() > 12 then
                                tmp = string.format("%e",tmp)
                            end
                            table.insert(tmpList, tmp)
                        end
                        table.insert(secondtab,tmpList)
                    end

                    if #cpulist > 1 then
                        secondtab_combined = tableMinMaxAvgSum(secondtab, 1, 1)
                    end
                end
                maxLineFields = math.max(#infotab, #firsttab, #firsttab_combined,
                                         #secondtab, #secondtab_combined, 2)
                
                if use_csv then
                    str = tostring(g)..","..groupName..","..groups[g][r]["Name"]
                    if maxLineFields > 3 then
                        str = str .. string.rep(",", maxLineFields-3)
                    end
                    if outfile ~= nil and g == 1 and r == 1 then
                        print(string.format("STRUCT,Info,3%s",string.rep(",",maxLineFields-3)))
                        print(string.format("CPU name:,%s%s", cpuinfo["osname"],string.rep(",",maxLineFields-2)))
                        print(string.format("CPU type:,%s%s", cpuinfo["name"],string.rep(",",maxLineFields-2)))
                        print(string.format("CPU clock:,%s GHz%s", clock*1.E-09,string.rep(",",maxLineFields-2)))
                    end
                else
                    if outfile ~= nil and g == 1 and r == 1 then
                        print(likwid.hline)
                        print(string.format("CPU name:\t%s",cpuinfo["osname"]))
                        print(string.format("CPU type:\t%s",cpuinfo["name"]))
                        print(string.format("CPU clock:\t%3.2f GHz",clock * 1.E-09))
                        print(likwid.hline)
                    end
                    print(likwid.dline)
                    str = "Group "..tostring(g).." "..groupName..": Region "..groups[g][r]["Name"]
                    print(str)
                    print(likwid.dline)
                end
                
                if use_csv then
                    print(string.format("STRUCT,Info,5%s",string.rep(",",maxLineFields-3)))
                    print(str)
                    likwid.printcsv(infotab, maxLineFields)
                    print(string.format("CPU clock,%f MHz%s",clock*1.E-9,string.rep(",",maxLineFields-2)))
                else
                    likwid.printtable(infotab)
                end
                if use_csv then
                    print(string.format("TABLE,Group %d Raw,%s,%d%s",g,groupName,#firsttab[1]-1,string.rep(",",maxLineFields-3)))
                    likwid.printcsv(firsttab, maxLineFields)
                else
                    likwid.printtable(firsttab)
                end
                if #cpulist > 1 then
                    if use_csv then
                        print(string.format("TABLE,Group %d Raw Stat,%s,%d%s",g,groupName,#firsttab_combined[1]-1,string.rep(",",maxLineFields-3)))
                        likwid.printcsv(firsttab_combined, maxLineFields)
                    else
                        likwid.printtable(firsttab_combined)
                    end
                end
                if likwid.tablelength(groupData[g]["Metrics"]) > 0 then
                    if use_csv then
                        print(string.format("TABLE,Group %d Metric,%s,%d%s",g,groupName,#secondtab[1]-1,string.rep(",",maxLineFields-3)))
                        likwid.printcsv(secondtab, maxLineFields)
                    else
                        likwid.printtable(secondtab)
                    end
                    if #cpulist > 1 then
                        if use_csv then
                            print(string.format("TABLE,Group %d Metric Stat,%s,%d%s",g,groupName,#secondtab_combined[1]-1,string.rep(",",maxLineFields-3)))
                            likwid.printcsv(secondtab_combined, maxLineFields)
                        else
                            likwid.printtable(secondtab_combined)
                        end
                    end
                end
            end
        end
    end
end


likwid.print_markerOutput = printMarkerOutput

local function getResults()
    local results = {}
    local nr_groups = likwid_getNumberOfGroups()
    local nr_threads = likwid_getNumberOfThreads()
    for i=1,nr_groups do
        results[i] = {}
        local nr_events = likwid_getNumberOfEvents(i)
        for j=1,nr_events do
            results[i][j] = {}
            for k=1, nr_threads do
                results[i][j][k] = likwid_getResult(i,j,k)
            end
        end
    end
    return results
end

likwid.getResults = getResults

local function getMarkerResults(filename, group_list, num_cpus)
    local cpuinfo = likwid_getCpuInfo()
    local ctr_and_events = likwid_getEventsAndCounters()
    local group_data = {}
    local results = {}
    local f = io.open(filename, "r")
    if f == nil then
        print("Have you called LIKWID_MARKER_CLOSE?")
        print(string.format("Cannot find intermediate results file %s", filename))
        return {}, {}
    end
    local lines = stringsplit(f:read("*all"),"\n")
    f:close()

    -- Read first line with general counts
    local tmpList = stringsplit(lines[1]," ")
    if #tmpList ~= 3 then
        print(string.format("Marker file %s not in proper format",filename))
        return {}, {}
    end
    local nr_threads = tonumber(tmpList[1])
    if tonumber(nr_threads) ~= tonumber(num_cpus) then
        print(string.format("Marker file lists only %d cpus, but perfctr configured %d cpus", nr_threads, num_cpus))
        return {},{}
    end
    local nr_regions = tonumber(tmpList[2])
    if tonumber(nr_regions) == 0 then
        print("No region results can be found in marker API output file")
        return {},{}
    end
    local nr_groups = tonumber(tmpList[3])
    if tonumber(nr_groups) == 0 then
        print("No group listed in the marker API output file")
        return {},{}
    end
    table.remove(lines,1)

    -- Read Region IDs and names from following lines
    for l=1, #lines do
        r, gname, g = string.match(lines[1],"(%d+):([%a%g]*)-(%d+)")
        if (r ~= nil and g ~= nil) then
            g = tonumber(g)+1
            r = tonumber(r)+1
            
            if group_data[g] == nil then
                group_data[g] = {}
            end
            if group_data[g][r] == nil then
                group_data[g][r] = {}
            end
            group_data[g][r]["ID"] = g
            group_data[g][r]["Name"] = gname
            group_data[g][r]["Time"] = {}
            group_data[g][r]["Count"] = {}
            if results[g] == nil then
                results[g] = {}
            end
            if results[g][r] == nil then
                results[g][r]= {}
            end
            table.remove(lines, 1 )
        else
            break
        end
    end

    for l, line in pairs(lines) do
        if line:len() > 0 then
            r, g, t, count = string.match(line,"(%d+) (%d+) (%d+) (%d+) %a*")
            if (r ~= nil and g ~= nil and t ~= nil and count ~= nil) then
                r = tonumber(r)+1
                g = tonumber(g)+1
                t = tonumber(t)+1
                tmpList = stringsplit(line, " ")
                table.remove(tmpList, 1)
                table.remove(tmpList, 1)
                table.remove(tmpList, 1)
                table.remove(tmpList, 1)
                time = tonumber(tmpList[1])
                events = tonumber(tmpList[2])
                table.remove(tmpList, 1)
                table.remove(tmpList, 1)
                
                table.insert(group_data[g][r]["Time"], t, time)
                table.insert(group_data[g][r]["Count"], t, count)
                for c=1, events do
                    if results[g][r][c] == nil then
                        results[g][r][c] = {}
                    end
                    if results[g][r][c][t] == nil then
                        results[g][r][c][t] = {}
                    end
                    local tmp = tonumber(tmpList[c])
                    results[g][r][c][t]["Value"] = tmp
                    results[g][r][c][t]["Counter"] = group_list[g]["Events"][c]["Counter"]
                end
            end
        end
    end
    return group_data, results
end

likwid.getMarkerResults = getMarkerResults


local function msr_available(flags)
    local ret = likwid_access("/dev/cpu/0/msr", flags)
    if ret == 0 then
        return true
    else
        local ret = likwid_access("/dev/msr0", flags)
        if ret == 0 then
            return true
        end
    end
    return false
end
likwid.msr_available = msr_available


local function addSimpleAsciiBox(container,lineIdx, colIdx, label)
    local box = {}
    if container[lineIdx] == nil then
        container[lineIdx] = {}
    end
    box["width"] = 1
    box["label"] = label
    table.insert(container[lineIdx], box)
end
likwid.addSimpleAsciiBox = addSimpleAsciiBox

local function addJoinedAsciiBox(container,lineIdx, startColIdx, endColIdx, label)
    local box = {}
    if container[lineIdx] == nil then
        container[lineIdx] = {}
    end
    box["width"] = endColIdx-startColIdx+1
    box["label"] = label
    table.insert(container[lineIdx], box)
end
likwid.addJoinedAsciiBox = addJoinedAsciiBox

local function printAsciiBox(container)
    local boxwidth = 0
    local numLines = #container
    local maxNumColumns = 0
    for i=1,numLines do
        if #container[i] > maxNumColumns then
            maxNumColumns = #container[i]
        end
        for j=1,#container[i] do
            if container[i][j]["label"]:len() > boxwidth then
                boxwidth = container[i][j]["label"]:len()
            end
        end
    end
    boxwidth = boxwidth + 2
    boxline = "+" .. string.rep("-",((maxNumColumns * (boxwidth+2)) + maxNumColumns+1)) .. "+"
    print(boxline)
    for i=1,numLines do
        innerboxline = "| "
        local numColumns = #container[i]
        for j=1,numColumns do
            innerboxline = innerboxline .. "+"
            if container[i][j]["width"] == 1 then
                innerboxline = innerboxline .. string.rep("-", boxwidth)
            else
                innerboxline = innerboxline .. string.rep("-", (container[i][j]["width"] * boxwidth + (container[i][j]["width"]-1)*3))
            end
            innerboxline = innerboxline .. "+ "
        end
        
        boxlabelline = "| "
        for j=1,numColumns do
            local offset = 0
            local width = 0
            local labellen = container[i][j]["label"]:len()
            local boxlen = container[i][j]["width"]
            if container[i][j]["width"] == 1 then
                width = (boxwidth - labellen)/2;
                offset = (boxwidth - labellen)%2;
            else
                width = (boxlen * boxwidth + ((boxlen-1)*3) - labellen)/2;
                offset = (boxlen * boxwidth + ((boxlen-1)*3) - labellen)%2;
            end
            boxlabelline = boxlabelline .. "|" .. string.rep(" ",(width+offset))
            boxlabelline = boxlabelline .. container[i][j]["label"]
            boxlabelline = boxlabelline ..  string.rep(" ",(width)) .. "| "
        end
        print(innerboxline .. "|")
        print(boxlabelline .. "|")
        print(innerboxline .. "|")
    end
    print(boxline)
end
likwid.printAsciiBox = printAsciiBox

-- Some helpers for output file substitutions
-- getpid already defined by Lua-C-Interface
local function gethostname()
    local f = io.popen("hostname -s","r")
    local hostname = f:read("*all"):gsub("^%s*(.-)%s*$", "%1")
    f:close()
    return hostname
end

likwid.gethostname = gethostname

local function getjid()
    local jid = os.getenv("PBS_JOBID")
    if jid == nil then
        jid = "X"
    end
    return jid
end

likwid.getjid = getjid

local function getMPIrank()
    local rank = os.getenv("PMI_RANK")
    if rank == nil then
        rank = os.getenv("OMPI_COMM_WORLD_RANK")
        if rank == nil then
            rank = "X"
        end
    end
    return rank
end

likwid.getMPIrank = getMPIrank

return likwid
