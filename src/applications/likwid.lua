local likwid = {}

groupfolder = "/home/rrze/unrz/unrz139/Work/likwid/trunk/groups"

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
                    return 0,{}
                end
                for i=start,ende do
                    table.insert(cpulist, i)
                end
            else
                if tonumber(expr) >= numainfo["numberOfNodes"] then
                    print("Expression selects too many nodes, host has ".. tablelength(numainfo)-1 .." nodes")
                    return 0,{}
                end
            end
        end
    elseif s2 ~= nil then
        local subList = stringsplit(cpustr,"-")
        local start = tonumber(subList[1])
        local ende = tonumber(subList[2])
        if ende >= numainfo["numberOfNodes"] then
            print("Expression selects too many nodes, host has ".. tablelength(numainfo)-1 .." nodes")
            return 0,{}
        end
        for i=start,ende do
            table.insert(cpulist, i)
        end
    else
        if (tonumber(cpustr) < numainfo["numberOfNodes"]) then
            table.insert(cpulist, tonumber(cpustr))
        end
    end
    return tablelength(cpulist),cpulist
end

likwid.nodestr_to_nodelist = nodestr_to_nodelist

local function get_groups(architecture)
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

local function get_groupdata(architecture, group)
    groupdata = {}
    local group_exist = 0
    num_groups, groups = get_groups(architecture)
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
    groupdata["GroupString"] = group
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

local function new_groupdata(eventString)
    local gdata = {}
    local num_events = 0
    gdata["Events"] = {}
    gdata["GroupString"] = eventString
    local eventslist = likwid.stringsplit(eventString,",")
    for i,e in pairs(eventslist) do
        eventlist = likwid.stringsplit(e,":")
        gdata["Events"][num_events] = {}
        gdata["Events"][num_events]["Event"] = eventlist[1]
        gdata["Events"][num_events]["Counter"] = eventlist[2]
        num_events = num_events + 1
    end
    return gdata
end

likwid.new_groupdata = new_groupdata

local function evaluate_groupmetrics(group, results)
    local cpuinfo = likwid_getCpuInfo()
    local gdata = get_groupdata(cpuinfo["short_name"], group)
    local metrics = gdata["Metrics"]
    local output = {}
    for i=1,#metrics do
        local formula = metrics[i]["formula"]
        for counter, result in pairs(results) do
            formula = string.gsub(formula, tostring(counter), tostring(result))
        end
        local result = assert(loadstring("return (" .. formula .. ")")())
        if (result ~= nil) then
            output[i] = result
        end
    end
    return output
end

likwid.evaluate_groupmetrics = evaluate_groupmetrics

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

local function get_spaces(str, max_space)
    local length = str:len()
    local back = math.ceil((max_space-length)/2)
    local front = max_space - length - back
    return string.rep(" ", front),string.rep(" ", back)
end

local function min_max_avg(values)
    min = math.huge
    max = 0.0
    sum = 0.0
    count = 0
    for _, value in pairs(values) do
        if (value < min) then min = value end
        if (value > max) then max = value end
        sum = sum + value
        count = count + 1
    end
    return min, max, sum/count
end


local function print_output(groupID, groupdata, cpulist)
    
    local num_events = likwid_getNumberOfEvents(groupId);
    local num_threads = likwid_getNumberOfThreads(groupID);
    local results = {}
    local metric_input = {}
    local mins = {}
    local maxs = {}
    local avgs = {}
    local max_eventName = 0
    local max_result = 0
    local front_space = 0
    local back_space = 0
    metric_input["time"] = likwid_getRuntimeOfGroup(groupID)* 1.E-06
    metric_input["inverseClock"] = 1.0/likwid_getCpuClock();
    for i=0,num_events-1 do
        results[i] = {}
        if groupdata["Events"][i]["Event"]:len() > max_eventName then
            max_eventName = groupdata["Events"][i]["Event"]:len()
        end
        for j=0,num_threads-1 do
            results[i][j] = likwid_getResult(groupID, i, j)
            if tostring(results[i][j]):len() > max_result then
                max_result = tostring(results[i][j]):len()
            end
        end
    end
    print(string.rep("-",4+max_eventName+ (max_result+3)*#cpulist))
    local heading = "Event"
    local front = ""
    local back = ""
    front, back = get_spaces(heading, max_eventName)
    heading = "| " .. front .. heading .. back .. " | "
    for j=1,num_threads do
        local cpuid = "Core " .. tostring(cpulist[j])
        front, back = get_spaces(cpuid, max_result)
        heading = heading .. front .. cpuid .. back .. " | "
    end
    print(heading)
    print(string.rep("-",4+max_eventName+ (max_result+3)*#cpulist))
    for i=0,num_events-1 do
        front, back = get_spaces(groupdata["Events"][i]["Event"], max_eventName)
        local event_result = "| " .. front .. groupdata["Events"][i]["Event"] .. back .. " | "
        metric_input[groupdata["Events"][i]["Counter"]] = 0.0
        mins[groupdata["Events"][i]["Event"]],maxs[groupdata["Events"][i]["Event"]], avgs[groupdata["Events"][i]["Event"]] = min_max_avg(results[i])
        for j=0,num_threads-1 do
            front, back = get_spaces(tostring(results[i][j]), max_result)
            event_result = event_result .. front .. tostring(results[i][j]) .. back .. " | "
            metric_input[groupdata["Events"][i]["Counter"]] = metric_input[groupdata["Events"][i]["Counter"]] + results[i][j]
        end
        print(event_result)
    end
    print(string.rep("-",4+max_eventName+ (max_result+3)*#cpulist))
    
    print()
    print(string.rep("-",4+max_eventName+ (max_result+3)*3))
    heading = "Event"
    front, back = get_spaces(heading, max_eventName)
    heading = "| " .. front .. heading .. back .. " | "
    event_string = "MIN"
    front, back = get_spaces(event_string, max_result)
    heading = heading .. front .. "MIN" .. back .. " | "
    heading = heading .. front .. "MAX" .. back .. " | "
    heading = heading .. front .. "AVG" .. back .. " | "
    print(heading)
    print(string.rep("-",4+max_eventName+ (max_result+3)*3))
    for i=0,num_events-1 do
        event_string = groupdata["Events"][i]["Event"]
        front, back = get_spaces(event_string, max_eventName)
        event_string = "| " .. front .. event_string .. back .. " | "
        heading = string.format("%d", mins[groupdata["Events"][i]["Event"]])
        front, back = get_spaces(heading, max_result)
        event_string = event_string .. front .. heading .. back .. " | "
        heading = string.format("%d", maxs[groupdata["Events"][i]["Event"]])
        front, back = get_spaces(heading, max_result)
        event_string = event_string .. front .. heading .. back .. " | "
        heading = string.format("%d", avgs[groupdata["Events"][i]["Event"]])
        front, back = get_spaces(heading, max_result)
        event_string = event_string .. front .. heading .. back .. " | "
        print(event_string)
    end
    print(string.rep("-",4+max_eventName+ (max_result+3)*3))
    
    if groupdata["Metrics"] then
        max_eventName = 0
        max_result = 0
        metric_input = evaluate_groupmetrics(groupdata["GroupString"], metric_input)
        for i, res in pairs(metric_input) do
            local desc = groupdata["Metrics"][i]["description"]
            local print_res = string.format("%.5f",res)
            if desc:len() > max_eventName then
                max_eventName = desc:len()
            end
            if print_res:len() > max_result then
                max_result = print_res:len()
            end
        end
        print()
        
        heading = "Metric"
        front, back = get_spaces(heading, max_eventName)
        heading = "| " .. front .. heading .. back .. " | "
        event_string = "Result"
        front, back = get_spaces(event_string, max_result)
        heading = heading .. front .. event_string .. back .. " | "
        
        print(string.rep("-",4+max_eventName+ (max_result+3)))
        print(heading)
        print(string.rep("-",4+max_eventName+ (max_result+3)))
        for i, res in pairs(metric_input) do
            event_string = "| "
            local desc = groupdata["Metrics"][i]["description"]
            local print_res = string.format("%.5f",res)
            current_length = desc:len()
            front_space = math.ceil((max_eventName-current_length)/2)
            back_space = max_eventName - current_length - front_space
            event_string = event_string .. string.rep(" ",front_space) .. desc .. string.rep(" ",back_space) .. " | "
            current_length = print_res:len()
            front_space = math.ceil((max_result-current_length)/2)
            back_space = max_result - current_length - front_space
            event_string = event_string .. string.rep(" ",front_space) .. print_res .. string.rep(" ",back_space) .. " | "
            print(event_string)
        end
    end
    print(string.rep("-",4+max_eventName+ (max_result+3)))
end

likwid.print_output = print_output

function getResults()
    local results = {}
    local nr_groups = likwid_getNumberOfGroups()
    local nr_threads = likwid_getNumberOfThreads()
    for i=0,nr_groups-1 do
        results[i] = {}
        local nr_events = likwid_getNumberOfEvents(i)
        for j=0,nr_events-1 do
            results[i][j] = {}
            for k=0, nr_threads-1 do
                results[i][j][k] = likwid_getResult(i,j,k)
            end
        end
    end
    return results
end

likwid.getResults = getResults

return likwid
