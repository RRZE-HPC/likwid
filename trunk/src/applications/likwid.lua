local likwid = {}

groupfolder = "/home/rrze/unrz/unrz139/Work/likwid/trunk/groups"

likwid.dline = string.rep("=",24)
likwid.hline =  string.rep("-",80)
likwid.sline = string.rep("*",80)


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
    for counter,value in pairs(counters_to_values) do
        formula = string.gsub(formula, tostring(counter), tostring(value))
    end
    local result = assert(loadstring("return (" .. formula .. ")")())
    if (result == nil) then
        result = "Nan"
    end
    return result
end

likwid.calculate_metric = calculate_metric

local function printtable(tab)
    local nr_columns = tablelength(tab)
    if nr_columns == 0 then
        print("Table has no columns. Empty table?")
        return ""
    end
    local nr_lines = tablelength(tab[1])
    local min_lengths = {}
    local max_lengths = {}
    for i, col in pairs(tab) do
        if tablelength(col) ~= nr_lines then
            print("Not all columns have the same row count, nr_lines"..tostring(nr_lines)..", current "..tablelength(col))
            return ""
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

    for i=0,num_events-1 do
        results[i+1] = {}
        for j=0,num_threads-1 do
            results[i+1][j] = likwid_getResult(groupID, i, j)
        end
    end
    
    tab =  {}
    tab[1] = {"Event"}
    counterlist = {}
    for i=0,num_events-1 do
        table.insert(tab[1],groupdata["Events"][i]["Event"])
    end
    for j=0,num_threads-1 do
        tmpList = {"Core "..tostring(cpulist[j+1])}
        for i=1,num_events do
            local tmp = tostring(results[i][j])
            if tostring(results[i][j]):len() > 6 then
                tmp = string.format("%e", results[i][j])
            end
            table.insert(tmpList, tmp)
        end
        table.insert(tab, tmpList)
    end
    likwid.printtable(tab)
    local mins = {}
    local maxs = {}
    local sums = {}
    local avgs = {}
    mins[1] = "Min"
    maxs[1] = "Max"
    sums[1] = "Sum"
    avgs[1] = "Avg"
    for i=1,num_events do
        mins[i+1] = math.huge
        maxs[i+1] = 0
        sums[i+1] = 0
        for j=0, num_threads-1 do
            if results[i][j] < mins[i+1] then
                mins[i+1] = results[i][j]
            end
            if results[i][j] > maxs[i+1] then
                maxs[i+1] = results[i][j]
            end
            sums[i+1] = sums[i+1] + results[i][j]
        end
        avgs[i+1] = sums[i+1] / num_threads
        if tostring(avgs[i+1]):len() > 6 then
            avgs[i+1] = string.format("%e",avgs[i+1])
        end
        if tostring(mins[i+1]):len() > 6 then
            mins[i+1] = string.format("%e",mins[i+1])
        end
        if tostring(maxs[i+1]):len() > 6 then
            maxs[i+1] = string.format("%e",maxs[i+1])
        end
        if tostring(sums[i+1]):len() > 6 then
            sums[i+1] = string.format("%e",sums[i+1])
        end
    end
    
    for i=#tab,2,-1 do
        table.remove(tab,i)
    end
    table.insert(tab, sums)
    table.insert(tab, maxs)
    table.insert(tab, mins)
    table.insert(tab, avgs)
    likwid.printtable(tab)
    
    if groupdata["Metrics"] then
    
        counterlist["time"] = likwid_getRuntimeOfGroup(groupID)* 1.E-06
        counterlist["inverseClock"] = 1.0/likwid_getCpuClock();
        tab = {}
        tab[1] = {"Metric"}
        for m=1,#groupdata["Metrics"] do
            table.insert(tab[1],groupdata["Metrics"][m]["description"] )
        end
        for j=0,num_threads-1 do
            tmpList = {"Core "..tostring(cpulist[j+1])}
            for i=1,num_events do
                counterlist[groupdata["Events"][i-1]["Counter"]] = results[i][j]
            end
            for m=1,#groupdata["Metrics"] do
                local tmp = calculate_metric(groupdata["Metrics"][m]["formula"], counterlist)
                if tostring(tmp):len() > 6 then
                    tmp = string.format("%e",tmp)
                end
                table.insert(tmpList, tostring(tmp))
            end
            table.insert(tab,tmpList)
        end
        likwid.printtable(tab)
        
        mins = {}
        maxs = {}
        sums = {}
        avgs = {}
        
        mins[1] = "Min"
        maxs[1] = "Max"
        sums[1] = "Sum"
        avgs[1] = "Avg"
        nr_lines = #tab[1]
        for j=2,nr_lines do
            for i=1, num_threads do
                if mins[j] == nil then
                    mins[j] = math.huge
                end
                if maxs[j] == nil then
                    maxs[j] = 0
                end
                if sums[j] == nil then
                    sums[j] = 0
                end
                
                local tmp = tonumber(tab[i+1][j])
                if tmp < mins[j] then
                    mins[j] = tmp
                end
                if tmp > maxs[j] then
                    maxs[j] = tmp
                end
                sums[j] = sums[j] + tmp
            end
            avgs[j] = sums[j] / num_threads
            if tostring(avgs[j]):len() > 6 then
                avgs[j] = string.format("%e",avgs[j])
            end
            if tostring(mins[j]):len() > 6 then
                mins[j] = string.format("%e",mins[j])
            end
            if tostring(maxs[j]):len() > 6 then
                maxs[j] = string.format("%e",maxs[j])
            end
            if tostring(sums[j]):len() > 6 then
                sums[j] = string.format("%e",sums[j])
            end
        end

        tab = {}
        tab[1] = {"Metric"}
        for m=1,#groupdata["Metrics"] do
            table.insert(tab[1],groupdata["Metrics"][m]["description"].." STAT" )
        end
        table.insert(tab, sums)
        table.insert(tab, maxs)
        table.insert(tab, mins)
        table.insert(tab, avgs)

        likwid.printtable(tab)
    end
end

likwid.print_output = print_output

local function printMarkerOutput(groups, results, groupData, cpulist)
    local nr_groups = #groups
    local nr_regions = #groups[1]
    local nr_threads = likwid.tablelength(groups[1][1]["Time"])
    for g=1,nr_groups do
        local gdata = nil
        if nr_groups ~= 1 then
            gdata = groupData[g]
        else
            gdata = groupData
        end
        local nr_events = likwid.tablelength(gdata["Events"])
        for r=1,nr_regions do
            print(likwid.dline)
            str = "Group "..tostring(g)..": Region "..groups[g][r]["Name"]
            print(str)
            print(likwid.dline)
            
            tab = {}
            tab[1] = {"Region Info","RDTSC Runtime [s]","call count"}
            for thread=1, nr_threads do
            --for thread, value in pairs(groups[g][r]["Time"]) do
                local tmpList = {}
                table.insert(tmpList, "Core "..tostring(cpulist[thread]))
                table.insert(tmpList, string.format("%.6f", groups[g][r]["Time"][thread]))
                table.insert(tmpList, tostring(groups[g][r]["Count"][thread]))
                table.insert(tab, tmpList)
            end
            likwid.printtable(tab)
            
            tab = {}
            tab[1] = {"Event"}
            for e=0,nr_events-1 do
                table.insert(tab[1],gdata["Events"][e]["Event"])
            end
            for t=0,nr_threads-1 do
                local tmpList = {}
                table.insert(tmpList, "Core "..tostring(cpulist[t+1]))
                for e=1,nr_events do
                    local tmp = results[r][g][e][t]["Value"]
                    if tmp == nil then
                        tmp = 0
                    end
                    table.insert(tmpList, string.format("%e",tmp))
                end
                table.insert(tab, tmpList)
            end
            likwid.printtable(tab)


            if likwid.tablelength(gdata["Metrics"]) > 0 then
                tab = {}
                tmpList = {"Metric"}
                for m=1,#gdata["Metrics"] do
                    table.insert(tmpList, gdata["Metrics"][m]["description"])
                end
                table.insert(tab, tmpList)
                for t=0,nr_threads-1 do
                    counterlist = {}
                    for e=1,nr_events do
                        counterlist[results[r][g][e][t]["Counter"]] = results[r][g][e][t]["Value"]
                    end
                    counterlist["inverseClock"] = 1.0/likwid_getCpuClock();
                    counterlist["time"] = groups[g][r]["Time"][t+1]
                    tmpList = {}
                    table.insert(tmpList, "Core "..tostring(t))
                    for m=1,#gdata["Metrics"] do
                        local tmp = likwid.calculate_metric(gdata["Metrics"][m]["formula"],counterlist)
                        if tmp == nil or tostring(tmp) == "-nan" then
                            tmp = 0
                        elseif tostring(tmp):len() > 6 then
                            tmp = string.format("%e",tmp)
                        end
                        table.insert(tmpList, tostring(tmp))
                    end
                    table.insert(tab,tmpList)
                end
                likwid.printtable(tab)
                
                mins = {}
                maxs = {}
                sums = {}
                avgs = {}
                
                for col=2,nr_threads+1 do
                    for row=2, #gdata["Metrics"]+1 do
                        if mins[row-1] == nil then
                            mins[row-1] = math.huge
                        end
                        if maxs[row-1] == nil then
                            maxs[row-1] = 0
                        end
                        if sums[row-1] == nil then
                            sums[row-1] = 0
                        end
                        tmp = tonumber(tab[col][row])
                        if tmp ~= nil then
                            if tmp < mins[row-1] then
                                mins[row-1] = tmp
                            end
                            if tmp > maxs[row-1] then
                                maxs[row-1] = tmp
                            end
                            sums[row-1] = sums[row-1] + tmp
                        else
                            mins[row-1] = 0
                            maxs[row-1] = 0
                            sums[row-1] = 0
                        end
                    end
                end
                for i=1,#sums do
                    avgs[i] = sums[i]/nr_threads
                    if tostring(avgs[i]):len() > 6 then
                        avgs[i] = string.format("%e",avgs[i])
                    end
                    if tostring(mins[i]):len() > 6 then
                        mins[i] = string.format("%e",mins[i])
                    end
                    if tostring(maxs[i]):len() > 6 then
                        maxs[i] = string.format("%e",maxs[i])
                    end
                    if tostring(sums[i]):len() > 6 then
                        sums[i] = string.format("%e",sums[i])
                    end
                end
                
                tab = {}
                tmpList = {"Metric"}
                for m=1,#gdata["Metrics"] do
                    table.insert(tmpList, gdata["Metrics"][m]["description"].." STAT")
                end
                table.insert(tab, tmpList)
                tmpList = {"Sum"}
                for m=1,#sums do
                    table.insert(tmpList, sums[m])
                end
                table.insert(tab, tmpList)
                tmpList = {"Min"}
                for m=1,#mins do
                    table.insert(tmpList, mins[m])
                end
                table.insert(tab, tmpList)
                tmpList = {"Max"}
                for m=1,#maxs do
                    table.insert(tmpList, maxs[m])
                end
                table.insert(tab, tmpList)
                tmpList = {"Avg"}
                for m=1,#avgs do
                    table.insert(tmpList, avgs[m])
                end
                table.insert(tab, tmpList)
                likwid.printtable(tab)
            end
            
            print()
        end
    end
end


likwid.print_markerOutput = printMarkerOutput

function getResults()
    local results = {}
    local nr_groups = likwid_getNumberOfGroups()
    local nr_threads = likwid_getNumberOfThreads()
    for i=1,nr_groups do
        results[i] = {}
        local nr_events = likwid_getNumberOfEvents(i-1)
        for j=1,nr_events do
            results[i][j] = {}
            for k=1, nr_threads do
                results[i][j][k] = likwid_getResult(i-1,j-1,k-1)
            end
        end
    end
    return results
end

likwid.getResults = getResults

function getMarkerResults(filename)
    local NUM_PMC = 85
    local cpuinfo = likwid_getCpuInfo()
    local ctr_and_events = likwid_getEventsAndCounters()
    local group_data = {}
    local results = {}
    local f = assert(io.open(filename, "r"))
    local lines = stringsplit(f:read("*all"),"\n")
    f:close()
    -- Read first line with general counts
    local tmpList = stringsplit(lines[1]," ")
    local nr_threads = tmpList[1]
    local nr_regions = tmpList[2]
    local nr_groups = tmpList[3]
    local regions_per_group = nr_regions/nr_groups
    table.remove(lines,1)
    
    -- Read Region IDs and names from following lines
    for r=1, regions_per_group do
        results[r] = {}
        for g=1, nr_groups do
            if group_data[g] == nil then
                group_data[g] = {}
            end
            if group_data[g][r] == nil then
                group_data[g][r] = {}
            end
            results[r][g] = {}
            tmpList = stringsplit(lines[1],":")
            tmpList = stringsplit(tmpList[2], "-")
            table.remove(tmpList, #tmpList)
            group_data[g][r]["Name"] = table.concat(tmpList,"-")
            group_data[g][r]["Time"] = {}
            group_data[g][r]["Count"] = {}
            table.remove(lines,1)
        end
    end

    for l=1, nr_regions*nr_threads do
        tmpList = stringsplit(lines[l], " ")
        r = (tonumber(tmpList[1]) % regions_per_group) + 1
        g = math.floor((tonumber(tmpList[1]) / regions_per_group) + 1)
        table.remove(tmpList, 1 )
        t = tonumber(tmpList[1])
        table.remove(tmpList, 1 )
        group_data[g][r]["Count"][t+1] = tonumber(tmpList[1])
        table.remove(tmpList, 1 )
        time = tonumber(tmpList[1])
        group_data[g][r]["Time"][t+1] = time
        table.remove(tmpList, 1 )
        for c=1,#ctr_and_events["Counters"] do
            if results[r][g][c] == nil then
                results[r][g][c] = {}
            end
            if results[r][g][c][t] == nil then
                results[r][g][c][t] = {}
            end
            local tmp = tonumber(tmpList[c])
            results[r][g][c][t]["Value"] = tmp
            results[r][g][c][t]["Counter"] = ctr_and_events["Counters"][c]
        end
    end
    
    --[[for r,_ in pairs(results) do
        for g,_ in pairs(results[r]) do
            for c,_ in pairs(results[r][g]) do
                for t,_ in pairs(results[r][g][c]) do
                    print(r,g,c,t,results[r][g][c][t]["Counter"],results[r][g][c][t]["Value"])
                end
            end
        end
    end]]
    
    return group_data, results
end

likwid.getMarkerResults = getMarkerResults

function createBitMask(gdata)
    local ctr_and_events = likwid_getEventsAndCounters()
    local bitmask_low = 0
    local bitmask_high = 0
    for i, tab in pairs(gdata["Events"]) do
        for j, ctr in pairs(tab) do
            if j == "Counter" then
                for k, c in pairs(ctr_and_events["Counters"]) do
                    if c == ctr then
                        if k-1 < 64 then
                            bitmask_low = bitmask_low + math.pow(2,k-1)
                        else
                            bitmask_high = bitmask_high + math.pow(2,k-1-63)
                        end
                    end
                end
            end
        end
    end
    return "0x" .. string.format("%x",bitmask_low) .. " 0x" .. string.format("%x",bitmask_high)
end

likwid.createBitMask = createBitMask

return likwid
