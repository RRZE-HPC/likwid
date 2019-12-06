#!../ext/lua/lua

---- Configuration ----
local default_opts = "EVENT_OPTION_NONE_MASK"
local opt_split_key = "|"

local event_match = "^EVENT_([A-Z0-9_]+)[ ]+(0x[A-F0-9]+)[ ]+([A-Z0-9|]+)"
local umask_match_long = "^UMASK_([A-Z0-9_]*)[ ]+(0x[A-F0-9]+)[ ]+(0x[A-F0-9]+)[ ]*(0x[A-F0-9]+)"
local umask_match_short = "^UMASK_([A-Z0-9_]*)[ ]+(0x[A-F0-9]+)"
local defopts_match = "^DEFAULT_OPTIONS_([A-Z0-9_]+)[ ]+([xA-Z0-9_=,]+)"
local opts_match ="^OPTIONS_([A-Z0-9_]+)[ ]+([A-Z0-9_|]+)"

local defoption_match="([A-Z0-9_]+)=(0x[A-F0-9]+)[,]*"
local option_match="([A-Z0-9_]+)[|]*"

local header = [[
/* DONT TOUCH: GENERATED FILE! */


#define NUM_ARCH_EVENTS_%s %d


static PerfmonEvent  %s_arch_events[NUM_ARCH_EVENTS_%s] = {
]]
-----------------------

if #arg ~= 2 then
    print("ERROR: Usage: lua gen_events.lua perfmon_<ARCH>_events.txt perfmon_<ARCH>_events.h\n")
    os.exit(1)
end

local inputfile = arg[1]
local outputfile = arg[2]
local arch = nil

if not inputfile:match(".*[/]*perfmon_([A-Za-z0-9]+)_events.txt") then
    print("ERROR: The input filename must follow the scheme perfmon_<ARCH>_events.txt\n")
    os.exit(1)
else
    arch = inputfile:match("perfmon_([A-Za-z0-9]+)_events.txt")
end

f = io.open(inputfile, "r")
local input = {}
if f then
    f:close()
else
    print(string.format("ERROR: The input path %s not readable", inputfile))
    os.exit(1)
end


function new_event()
    e = {}
    e.name = nil
    e.eventId = nil
    e.limit = nil
    e.umask = nil
    e.cfg = nil
    e.cmask = nil
    e.opts = default_opts
    e.defopts = nil
    return e
end

function copy_event(event)
    e = new_event()
    e.name = event.name
    e.eventId = event.eventId
    e.limit = event.limit
    return e
end

function format_event(event)
    s = string.format("%q, %q, %s,%s,%s,%s,", event.name,
                                                event.limit,
                                                event.eventId,
                                                event.umask,
                                                event.cfg or "0",
                                                event.cmask or "0")
    local tmp = {}
    if event.defopts then
        for optkey, optval in event.defopts:gmatch(defoption_match) do
            if event.opts == default_opts then
                event.opts = optkey.."_MASK"
            else
                event.opts = event.opts.."|"..optkey.."_MASK"
            end
            table.insert(tmp, string.format("{%s, %s}", optkey, optval))
        end
        event.defopts = "{"..table.concat(tmp, ", ") .. "}"
    else
        event.defopts = "{}"
    end
    s = s .. string.format("%d,%s,%s", #tmp, event.opts, event.defopts)
    return "{" .. s .. "},"
end

function valid_event(event)
    if event.name and event.umask and event.limit and event.eventId then
        return true
    end
    return false
end

event = new_event()

local output = {}
for line in io.lines(inputfile) do
    if line:match(event_match) then
        eventname, eventId, limit = line:match(event_match)
        if eventname then event.name = eventname end
        if eventId then event.eventId = eventId end
        if limit then event.limit = limit end
    elseif line:match(umask_match_long) then
        umask_key, umask, cfg, cmask = line:match(umask_match_long)
        if umask_key:match(string.format("%s[A-Z0-9_]*", event.name)) then
            local key = event.name
            event.name = umask_key
            if umask then event.umask = umask end
            if cfg then event.cfg = cfg end
            if cmask then event.cmask = cmask end
            if valid_event(event) then
                table.insert(output, event)
                event = copy_event(event)
                event.name = key
            end
        end
    elseif line:match(umask_match_short) then
        umask_key, umask = line:match(umask_match_short)
        if umask_key:match(string.format("%s[A-Z0-9_]*", event.name)) then
            local key = event.name
            event.name = umask_key
            if umask then event.umask = umask end
            if valid_event(event) then
                table.insert(output, event)
                event = copy_event(event)
                event.name = key
            end
        end
    elseif line:match(defopts_match) then
        defopt_key, defopt_value = line:match(defopts_match)
        if defopt_key:match(string.format("%s[A-Z0-9_]*", event.name)) then
            event.defopts = defopt_value
        end
    elseif line:match(opts_match) then
        opt_key, opt_value = line:match(opts_match)
        if opt_key:match(string.format("%s[A-Z0-9_]*", event.name)) then
            event.opts = opt_value
        end
    end
end
if valid_event(event) then
    table.insert(output, event)
end

f = io.open(outputfile, "w")
if f then
    f:write(string.format(header, arch:upper(), #output, arch, arch:upper()))
    for _, e in pairs(output) do
        s = format_event(e)
        f:write("  "..s.."\n")
    end
    f:write("};\n")
else
    print(string.format("ERROR: The output path %s not writable", outputfile))
    os.exit(1)
end
