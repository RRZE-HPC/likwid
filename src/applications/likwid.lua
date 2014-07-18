local likwid = {}

groupfolder = "/home/rrze/unrz/unrz139/Work/likwid/trunk/groups"
architecture = "ivybridge"

local function getopt(args, ostr)
    local arg, place = nil, 0;
    return function ()
        if place == 0 then -- update scanning pointer
            place = 1
            if #args == 0 or args[1]:sub(1, 1) ~= '-' then place = 0; return nil end
            if #args[1] >= 2 then
                place = place + 1
                if args[1]:sub(2, 2) == '-' then -- found "--"
                    place = 0
                    table.remove(args, 1);
                    return nil;
                end
            end
        end
        local optopt = args[1]:sub(place, place);
        place = place + 1;
        local oli = ostr:find(optopt);
        if optopt == ':' or oli == nil then -- unknown option
            if optopt == '-' then return nil end
            if place > #args[1] then
                table.remove(args, 1);
                place = 0;
            end
            return '?';
        end
        oli = oli + 1;
        if ostr:sub(oli, oli) ~= ':' then -- do not need argument
            arg = true;
            if place > #args[1] then
                table.remove(args, 1);
                place = 0;
            end
        else -- need an argument
            if place <= #args[1] then -- no white space
                arg = args[1]:sub(place);
            else
                table.remove(args, 1);
                if #args == 0 then -- an option requiring argument is the last one
                    place = 0;
                    if ostr:sub(1, 1) == ':' then return ':' end
                    return '?';
                else arg = args[1] end
            end
            table.remove(args, 1);
            place = 0;
        end
        return optopt, arg;
    end
end

likwid.getopt = getopt

local function tablelength(T)
    local count = 0
    if T == nil then return count end
    for _ in pairs(T) do count = count + 1 end
    return count
end

likwid.tablelength = tablelength

local function tableprint(T)
    if T == nil then return end
    outstr = ""
    for i, item in pairs(T) do
        outstr = outstr .. "," .. item
    end
    print("["..outstr:sub(2,outstr:len()).."]")
end

likwid.tableprint = tableprint

function stringsplit(astr, sSeparator, nMax, bRegexp)
    assert(sSeparator ~= '')
    assert(nMax == nil or nMax >= 1)

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

local function cpustr_to_cpulist_scatter(cpustr)
    local cpulist = {}
    local domain_list = {}
    local s,e = cpustr:find(":")
    if s ~= nil then
        local domain = cpustr:sub(1,s-1)
        local expression = cpustr:sub(s+1,cpustr:len())
        local affinity = likwid_getAffinityInfo()
        local topo = likwid_getCpuTopology()

        for dom,content in pairs(affinity["domains"]) do
            s,e = dom:find(domain)
            if s ~= nil then 
                table.insert(domain_list, dom)
            end
        end

        local num_domains = tablelength(domain_list)
        local domain_idx = 1
        local threadID = 0
        for i=0,topo["numHWThreads"]-1 do
            table.insert(cpulist, affinity["domains"][domain_list[domain_idx]]["processorList"][threadID])
            domain_idx = domain_idx + 1
            if domain_idx == num_domains+1 then domain_idx = 1 end
            if (domain_idx == 1) then threadID = threadID + 1 end
        end
    else
        print("Cannot parse scatter expression")
        return {}
    end
    return cpulist
end

local function cpustr_to_cpulist_expression(cpustr)
    local cpulist = {}
    local affinity = likwid_getAffinityInfo()
    local s1,e1 = cpustr:find("E:")
    expression_list = stringsplit(cpustr:sub(e1+1,cpustr:len()),"@")
    for expr_idx,expr in pairs(expression_list) do
        subexpr_list = stringsplit(expr,":")
        if tablelength(subexpr_list) == 2 then
            domain = subexpr_list[1]
            s1,e1 = subexpr_list[2]:find("-")
            local s2,e2 = subexpr_list[2]:find(",")
            if affinity["domains"][domain] == nil then
                print("Domain "..domain.." not valid")
                return {}
            elseif s1 ~= nil or s2 ~= nil then
                print("Only single count required, no list like " .. subexpr_list[2])
                return {}
            end
            local count = tonumber(subexpr_list[2])
            for i=0,count-1 do
                table.insert(cpulist,affinity["domains"][domain]["processorList"][i])
            end
        elseif tablelength(subexpr_list) == 4 then
            domain = subexpr_list[1]
            count = tonumber(subexpr_list[2])
            chunk = tonumber(subexpr_list[3])
            stride = tonumber(subexpr_list[4])
            if affinity["domains"][domain] == nil then
                print("Domain "..domain.." not valid")
                return {}
            end
            index = 0
            for i=0,count-1 do
                for j=0,chunk-1 do
                    table.insert(cpulist,affinity["domains"][domain]["processorList"][index+j])
                end
                index = index + stride
                if (index > affinity["domains"][domain]["numberOfProcessors"]) then
                    index = 0
                end
            end
        else
            print("Cannot parse expression")
            return {}
        end
    end
    return cpulist
end

local function cpustr_to_cpulist_logical(cpustr)
    local cpulist = {}
    local affinity = likwid_getAffinityInfo()
    local domain = "N"
    local s1,e1 = cpustr:find(":")
    if s1 == nil then 
        e1 = -1
    else
        if cpustr:sub(1,s1-1) ~= "L" then
            domain = cpustr:sub(1,s1-1)
        end
    end
    print(domain)
    tableprint(affinity["domains"][domain]["processorList"])
    expression_list = stringsplit(cpustr:sub(e1+1,cpustr:len()),"@")
    for expr_idx,expr in pairs(expression_list) do
        s1,e1 = expr:find(",")
        local s2,e2 = expr:find("-")
        if s1 ~= nil then
            subexpr_list = stringsplit(expr,",")
            for subexpr_index, subexpr in pairs(subexpr_list) do
                s2,e2 = subexpr:find("-")
                if s2 ~= nil then
                    local tmp_list = stringsplit(subexpr,"-")
                    local start = tonumber(tmp_list[1])
                    local ende = tonumber(tmp_list[2])
                    if ende >= tablelength(affinity["domains"][domain]["processorList"]) then
                        print("Expression selects too many CPUs")
                        return {}
                    end
                    for i=start,ende do
                        table.insert(cpulist,affinity["domains"][domain]["processorList"][i])
                    end
                else
                    if tonumber(subexpr) < tablelength(affinity["domains"][domain]["processorList"]) then
                        table.insert(cpulist,affinity["domains"][domain]["processorList"][subexpr])
                    end
                end
            end
        elseif s2 ~= nil then
            subexpr_list = stringsplit(expr,"-")
            local start = tonumber(subexpr_list[1])
            local ende = tonumber(subexpr_list[2])
            if ende >= tablelength(affinity["domains"][domain]["processorList"]) then
                print("Expression selects too many CPUs")
                return {}
            end
            for i=start,ende do
                table.insert(cpulist,affinity["domains"][domain]["processorList"][i])
            end
        else
            table.insert(cpulist,affinity["domains"][domain]["processorList"][tonumber(expr)])
        end
    end
    return cpulist
end

local function cpustr_to_cpulist_physical(cpustr)
    local cpulist = {}
    local affinity = likwid_getAffinityInfo()
    local s1,e1 = cpustr:find(",")
    local s2,e2 = cpustr:find("-")
    if s1 ~= nil then
        local expression_list = stringsplit(cpustr,",")
        for i, expr in pairs(expression_list) do
            s2,e2 = expr:find("-")
            if s2 ~= nil then
                local subList = stringsplit(expr,"-")
                local start = tonumber(subList[1])
                local ende = tonumber(subList[2])
                if ende >= tablelength(affinity["domains"]["N"]["processorList"]) then
                    print("Expression selects too many CPUs")
                    return {}
                end
                for i=start,ende do
                    table.insert(cpulist, i)
                end
            else
                if tonumber(expr) < tablelength(affinity["domains"]["N"]["processorList"]) then
                    table.insert(cpulist, tonumber(expr))
                end
            end
        end
    elseif s2 ~= nil then
        subList = stringsplit(cpustr,"-")
        local start = tonumber(subList[1])
        local ende = tonumber(subList[2])
        if ende >= tablelength(affinity["domains"]["N"]["processorList"]) then
            print("Expression selects too many CPUs")
            return {}
        end
        for i=start,ende do
            table.insert(cpulist, i)
        end
    else
        table.insert(cpulist, tonumber(cpustr))
    end
    return cpulist
end

local function cpustr_to_cpulist(cpustr)
    local cpulist = {}
    local filled = false
    local s1,e1 = cpustr:find("N")
    if s1 == nil then s1,e = cpustr:find("S") end
    if s1 == nil then s1,e = cpustr:find("C") end
    if s1 == nil then s1,e = cpustr:find("M") end
    local s2,e2 = cpustr:find("L")
    if s1 ~= nil then
        s1,e1 = cpustr:find("scatter")
        if s1 ~= nil then
            cpulist = cpustr_to_cpulist_scatter(cpustr)
            filled = true
        end
        s1,e1 = cpustr:find("E")
        if s1 ~= nil then
            cpulist = cpustr_to_cpulist_expression(cpustr)
            filled = true
        end
        if filled == false then
            cpulist = cpustr_to_cpulist_logical(cpustr)
        end
    elseif s2 ~= nil then
        cpulist = cpustr_to_cpulist_logical(cpustr)
    else
        cpulist = cpustr_to_cpulist_physical(cpustr)
    end
    return tablelength(cpulist),cpulist
end

likwid.cpustr_to_cpulist = cpustr_to_cpulist
likwid.cpustr_to_cpulist_physical = cpustr_to_cpulist_physical

local function nodestr_to_nodelist(cpustr)
    local cpulist = {}
    local numainfo = likwid_getNumaInfo()
    local s1,e1 = cpustr:find(",")
    local s2,e2 = cpustr:find("-")
    if s1 ~= nil then
        local expression_list = stringsplit(cpustr,",")
        for i, expr in pairs(expression_list) do
            s2,e2 = expr:find("-")
            if s2 ~= nil then
                local subList = stringsplit(cpustr,"-")
                local start = tonumber(subList[1])
                local ende = tonumber(subList[2])
                if ende >= numainfo["numberOfNodes"] then
                    print("Expression selects too many nodes, host has ".. tablelength(numainfo)-1 .." nodes")
                    return {}
                end
                for i=start,ende do
                    table.insert(cpulist, i)
                end
            else
                if tonumber(expr) >= numainfo["numberOfNodes"] then
                    print("Expression selects too many nodes, host has ".. tablelength(numainfo)-1 .." nodes")
                    return {}
                end
            end
        end
    elseif s2 ~= nil then
        cpulist = stringsplit(cpustr,"-")
    else
        table.insert(cpulist, tonumber(cpustr))
    end
    return tablelength(cpulist),cpulist
end

likwid.nodestr_to_nodelist = nodestr_to_nodelist

local function get_groups()
    groups = {}
    local f = io.popen("ls " .. groupfolder .. "/" .. architecture)
    t = stringsplit(f:read("*a"),"\n")
    for i, a in pairs(t) do
        if a ~= "" then
            table.insert(groups,a:sub(0,a:len()-4))
        end
    end
    return #groups,groups
end

likwid.get_groups = get_groups

local function get_groupdata(group)
    groupdata = {}
    local group_exist = 0
    num_groups, groups = get_groups()
    for i, a in pairs(groups) do
        if (a == group) then group_exist = 1 end
    end
    if (group_exist == 0) then return end
    
    local f = assert(io.open(groupfolder .. "/" .. architecture .. "/" .. group .. ".txt", "r"))
    local t = f:read("*all")
    f:close()
    local parse_eventset = false
    local parse_metrics = false
    local parse_long = false
    groupdata["EventString"] = ""
    groupdata["Events"] = {}
    groupdata["Metrics"] = {}
    groupdata["LongDescription"] = ""
    nr_events = 0
    nr_metrics = 1
    for i, line in pairs(stringsplit(t,"\n")) do
        
        if (parse_eventset or parse_metrics or parse_long) and line:len() == 0 then
            parse_eventset = false
            parse_metrics = false
            parse_long = false
        end
        local s,e = line:find("SHORT")
        if s ~= nil then
            linelist = stringsplit(line, "%s+", nil, "%s+")
            table.remove(linelist, 1)
            groupdata["ShortDescription"] = table.concat(linelist, " ")  
        end
        s,e = line:find("EVENTSET")
        if s ~= nil then
            parse_eventset = true
            
        end
        s,e = line:find("METRICS")
        if s ~= nil then
            parse_metrics = true
        end
        s,e = line:find("LONG")
        if s ~= nil then
            parse_long = true
        end
        
        if parse_eventset and line:find("EVENTSET") == nil then
            linelist = stringsplit(line, "%s+", nil, "%s+")
            groupdata["EventString"] = groupdata["EventString"] .. "," .. linelist[2] .. ":" .. linelist[1]
            groupdata["Events"][nr_events] = {}
            groupdata["Events"][nr_events]["Event"] = linelist[2]
            groupdata["Events"][nr_events]["Counter"] = linelist[1]
            nr_events = nr_events + 1
        end
        
        if parse_metrics and line:find("METRICS") == nil then
            linelist = stringsplit(line, "%s+", nil, "%s+")
            formula = linelist[#linelist]
            table.remove(linelist)
            groupdata["Metrics"][nr_metrics] = {}
            groupdata["Metrics"][nr_metrics]["description"] = table.concat(linelist, " ")  
            groupdata["Metrics"][nr_metrics]["formula"] = formula
            nr_metrics = nr_metrics + 1
        end
        
        if parse_long and line:find("LONG") == nil then
            groupdata["LongDescription"] = groupdata["LongDescription"] .. "\n" .. line
        end
    end
    groupdata["LongDescription"] = groupdata["LongDescription"]:sub(2)
    groupdata["EventString"] = groupdata["EventString"]:sub(2)
    
    return groupdata
    
end

likwid.get_groupdata = get_groupdata

local function evaluate_groupmetrics(group, results)
    gdata = get_groupdata(group)
    metrics = gdata["Metrics"]
    output = {}
    for i=1,#metrics do
        formula = metrics[i]["formula"]
        for counter, result in pairs(results) do
            formula = string.gsub(formula, tostring(counter), tostring(result))
        end
        result = assert(loadstring("return (" .. formula .. ")")())
        if (result ~= nil) then
            output[metrics[i]["description"]] = result
        end
    end
    return output
end

likwid.evaluate_groupmetrics = evaluate_groupmetrics

local function parse_time(timestr)
    local s1,e1 = timestr:find("ms")
    local s2,e2 = timestr:find("us")
    if s1 ~= nil then
        duration = tonumber(timestr:sub(1,s1-1)) * 1.E03
    elseif s2 ~= nil then
        duration = tonumber(timestr:sub(1,s2-1))
    else
        s1,e1 = timestr:find("s")
        if s1 == nil then
            print("Cannot parse time for timeline mode, " .. timestr .. "not well formatted")
            os.exit(1)
        end
        duration = tonumber(timestr:sub(1,s1-1)) * 1.E06
    end
    return duration
end

likwid.parse_time = parse_time


return likwid
