/*! \page lua_Info Information about LIKWID's Lua API
<H1>How to include Lua API into own Lua applications</H1>
<CODE>
package.path = package.path .. ';${PATH_TO_LIKWID.LUA}/?.lua'<BR>
local likwid = require("likwid")<BR>
</CODE>
<P></P>
Now all function and variables can be called with<BR>
<CODE>likwid.<I>functionname()</I></CODE><BR>
or<BR>
<CODE>likwid.<I>variable</I></CODE>

<H1>Global variables defined by LIKWID's Lua API</H1>
<TABLE>
<TR>
  <TH>Variablename</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>\a groupfolder</TD>
  <TD>Path to the folder containing the definitions of the performance groups</TD>
</TR>
<TR>
  <TD>\a version</TD>
  <TD>Version of LIKWID</TD>
</TR>
<TR>
  <TD>\a release</TD>
  <TD>Release number of LIKWID</TD>
</TR>
<TR>
  <TD>\a pinlibpath</TD>
  <TD>Path to the pinning library. Is added automatically to $LD_PRELOAD by \ref likwid-pin and \ref likwid-perfctr</TD>
</TR>
<TR>
  <TD>\a hline</TD>
  <TD>Horizontal line with 80 '-' characters</TD>
</TR>
<TR>
  <TD>\a sline</TD>
  <TD>Horizontal line with 80 '*' characters</TD>
</TR>
<TR>
  <TD>\a dline</TD>
  <TD>Horizontal line with 80 '=' characters</TD>
</TR>
</TABLE>
*/

/*! \page lua_Config Config file module
<H1>Data type definition for Lua config file module in the Lua API</H1>
\anchor lua_config
<H2>Config file read</H2>
<P>This structure is returned by \ref getConfiguration function<BR>The config file can be creted with \ref likwid-genTopoCfg executable</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a topologyFile</TD>
  <TD>Path to the config file containing topology information</TD>
</TR>
<TR>
  <TD>\a daemonPath</TD>
  <TD>Path to the access daemon</TD>
</TR>
<TR>
  <TD>\a daemonMode</TD>
  <TD>Access mode for LIKWID (0 = direct access, 1 = access daemon)</TD>
</TR>
<TR>
  <TD>\a maxNumThreads</TD>
  <TD>Maximal amount of hardware threads in the system</TD>
</TR>
<TR>
  <TD>\a maxNumNodes</TD>
  <TD>Maximal amount of NUMA nodes in the system</TD>
</TR>
<TR>
  <TD>\a maxHashTableSize</TD>
  <TD>Maximal size for the internally used hash table</TD>
</TR>
</TABLE>

<H1>Function definitions for Lua config file module in the Lua API</H1>
\anchor getConfiguration
<H2>getConfiguration()</H2>
<P>Read the configuration file and return a list of config options</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>List of configuration options, see \ref lua_config</TD>
</TR>
</TABLE>

\anchor setVerbosity
<H2>setVerbosity()</H2>
<P>Define and/or change the verbosity level of LIKWID</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a verbosity</TD>
      <TD>0 = only errors<BR>1 = infos<BR>2 = detail<BR>3 = developer<BR>Other flags are rejected.</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor putConfiguration
<H2>putConfiguration()</H2>
<P>Frees the C-structures that were created by \ref getConfiguration function.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

*/

/*! \page lua_Access Access client module
<H1>Data type definition for Lua access client module in the Lua API</H1>
<H1>Function definitions for Lua access client module in the Lua API</H1>
\anchor setAccessMode
<H2>setAccessMode()</H2>
<P>Define and/or change the access mode to the MSR and PCI registers</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a accessFlag</TD>
      <TD>0 = direct access<BR>1 = access daemon<BR>Other flags are rejected.</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Always 0</TD>
</TR>
</TABLE>

*/

/*! \page lua_CPUTopology CPU information module
<H1>Data type definition for CPU information module in the Lua API</H1>
\anchor lua_cpuinfo
<H2>Cpu Info</H2>
<P>This structure is returned by \ref getCpuInfo function<BR>It is similar to the C struct CpuInfo</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a family</TD>
  <TD>Family ID of CPU</TD>
</TR>
<TR>
  <TD>\a model</TD>
  <TD>Model ID of CPU</TD>
</TR>
<TR>
  <TD>\a stepping</TD>
  <TD>Revision of CPU</TD>
</TR>
<TR>
  <TD>\a clock</TD>
  <TD>Base clock frequency</TD>
</TR>
<TR>
  <TD>\a turbo</TD>
  <TD>Flag if the system supports the Turbo mode</TD>
</TR>
<TR>
  <TD>\a name</TD>
  <TD>Name of the microarchitecture</TD>
</TR>
<TR>
  <TD>\a osname</TD>
  <TD>Name of the CPU as given by manufacturer</TD>
</TR>
<TR>
  <TD>\a short_name</TD>
  <TD>Short name of microarchitecture</TD>
</TR>
<TR>
  <TD>\a features</TD>
  <TD>String with all interesting CPU feature flags as a space separated list</TD>
</TR>
<TR>
  <TD>\a featureFlags</TD>
  <TD>Bitmask with all interesting CPU feature flags<BR>Bit positions can be retrieved from the FeatureBit enum</TD>
</TR>
<TR>
  <TD>\a isIntel</TD>
  <TD>Flag to check if the system is using Intel CPUs</TD>
</TR>
<TR>
  <TD>\a perf_version</TD>
  <TD>Version of architectural performance monitoring capabilities</TD>
</TR>
<TR>
  <TD>\a perf_num_ctr</TD>
  <TD>Amount of core-local general-purpose counters</TD>
</TR>
<TR>
  <TD>\a perf_num_fixed_ctr</TD>
  <TD>Amount of core-local fixed-purpose counters</TD>
</TR>
<TR>
  <TD>\a perf_width_ctr</TD>
  <TD>Register width of core-local counters</TD>
</TR>
</TABLE>


\anchor lua_cputopo
<H2>Cpu Topology</H2>
<P>This structure is returned by \ref getCpuTopology function<BR>The nested list structure is similar to the C struct CpuTopology.</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a numHWThreads</TD>
  <TD>Total amount of hardware threads in the system</TD>
</TR>
<TR>
  <TD>\a activeHWThreads</TD>
  <TD>Amount of active hardware threads in the system</TD>
</TR>
<TR>
  <TD>\a numSockets</TD>
  <TD>Number of CPU sockets in the system</TD>
</TR>
<TR>
  <TD>\a numCoresPerSocket</TD>
  <TD>Number of physical cores of each socket in the system</TD>
</TR>
<TR>
  <TD>\a numThreadsPerCore</TD>
  <TD>Number of hardware threads of each core in the system</TD>
</TR>
<TR>
  <TD>\a numCacheLevels</TD>
  <TD>Amount of cache levels in the system</TD>
</TR>
<TR>
  <TD>\a threadPool<BR>(List with<BR>\a numHWThreads entries)</TD>
    <TD>
    <TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>\a threadId</TD>
      <TD>Thread ID</TD>
    </TR>
    <TR>
      <TD>\a coreId</TD>
      <TD>ID of physical CPU core</TD>
    </TR>
    <TR>
      <TD>\a apicId</TD>
      <TD>ID of the interrupt line for the hardware thread as defined by ACPI</TD>
    </TR>
    <TR>
      <TD>\a packageId</TD>
      <TD>ID of CPU socket for the current thread</TD>
    </TR>
    <TR>
      <TD>\a inCpuSet</TD>
      <TD>Defines whether the thread is available in current cpuset</TD>
    </TR>
    </TABLE>
    </TD>
</TR>
<TR>
  <TD>\a cacheLevels<BR>(List with<BR>\a numCacheLevels entries)</TD>
    <TD>
    <TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>\a level</TD>
      <TD>Level of cache</TD>
    </TR>
    <TR>
      <TD>\a associativity</TD>
      <TD>Associativity in cache level</TD>
    </TR>
    <TR>
      <TD>\a sets</TD>
      <TD>Sets in cache level</TD>
    </TR>
    <TR>
      <TD>\a lineSize</TD>
      <TD>Size of a cache line in cache level</TD>
    </TR>
    <TR>
      <TD>\a size</TD>
      <TD>Size in bytes of cache level</TD>
    </TR>
    <TR>
      <TD>\a threads</TD>
      <TD>Amount of threads sharing the cache</TD>
    </TR>
    <TR>
      <TD>\a inclusive</TD>
      <TD>Inclusiveness of cache</TD>
    </TR>
    <TR>
      <TD>\a type</TD>
      <TD>
        <TABLE>
        <TR>
          <TH>Typename</TH>
          <TH>comment</TH>
        </TR>
        <TR>
          <TD>DATACACHE</TD>
          <TD>Cache manages only data</TD>
        </TR>
        <TR>
          <TD>INSTRUCTIONCACHE</TD>
          <TD>Cache manages only instructions</TD>
        </TR>
        <TR>
          <TD>UNIFIEDCACHE</TD>
          <TD>Cache manages data and instructions</TD>
        </TR>
        <TR>
          <TD>ITLB</TD>
          <TD>Translation Lookaside Buffer for instruction page addresses</TD>
        </TR>
        <TR>
          <TD>DTLB</TD>
          <TD>Translation Lookaside Buffer for data page addresses</TD>
        </TR>
        <TR>
          <TD>NOCACHE</TD>
          <TD>Type cannot be determined</TD>
        </TR>
        </TABLE>
      </TD>
    </TR>
    </TABLE>
    </TD>
</TR>
<TR>
  <TD>\a topologyTree</TD>
  <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>\a ID</TD>
      <TD>ID of socket</TD>
    </TR>
    <TR>
      <TD>\a Childs</TD>
      <TD><TABLE>
        <TR>
            <TH>Membername</TH>
            <TH>Comment</TH>
        </TR>
        <TR>
            <TD>\a ID</TD>
            <TD>ID of CPU core</TD>
        </TR>
        <TR>
            <TD>\a Childs</TD>
            <TD>List of thread IDs for the current CPU core</TD>
        </TR>
      </TABLE></TD>
  </TABLE></TD>
</TR>
</TABLE>


<H1>Function definitions for Lua CPU information module in the Lua API</H1>
\anchor getCpuInfo
<H2>getCpuInfo()</H2>
<P>Get basic information about the CPUs in the system</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Cpu Info \ref lua_cpuinfo</TD>
</TR>
</TABLE>

\anchor getCpuTopology
<H2>getCpuTopology()</H2>
<P>Get the topology information about the CPUs in the system</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Return</TD>
  <TD>Cpu Topology \ref lua_cputopo</TD>
</TR>
</TABLE>

<H2>putTopology()</H2>
<P>Frees C struct CpuInfo and CpuTopology. You can still use the lua_cpuinfo and lua_cputopo data structures<BR>If you call \ref getCpuInfo or \ref getCpuTopology functions again after calling this function, the topology information will be read again.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Return</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor cpustr_to_cpulist
<H2>cpustr_to_cpulist()</H2>
<P>Resolve the given CPU expression string to a list of CPUs as available in the system</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuexpression</TD>
      <TD>CPU expression string. Look at \ref likwid-pin for possible formats</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Return</TD>
  <TD><TABLE>
    <TR>
      <TD>\a nrCPUs</TD>
      <TD>Number of CPUs in the \a cpulist</TD>
    </TR>
    <TR>
      <TD>\a cpulist</TD>
      <TD>List containing the CPU IDs after resolution of the cpu expression</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>


*/


/*! \page lua_NumaInfo NUMA memory topology module

<H1>Data type definition for Lua NUMA topology module in the Lua API</H1>
\anchor lua_numainfo
<H2>NUMA Info</H2>
<P>This structure is returned by \ref getNumaInfo function<BR>It is similar to the C struct NumaTopology</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a numberOfNodes</TD>
  <TD>Amount of NUMA nodes in the system</TD>
</TR>
<TR>
  <TD>\a nodes</TD>
    <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>id</TD>
      <TD>ID of NUMA node</TD>
    </TR>
    <TR>
      <TD>totalMemory</TD>
      <TD>Total amount of memory in the NUMA domain</TD>
    </TR>
    <TR>
      <TD>freeMemory</TD>
      <TD>Free amount of memory in the NUMA domain</TD>
    </TR>
    <TR>
      <TD>numberOfProcessors</TD>
      <TD>Amount of CPUs in the NUMA domain</TD>
    </TR>
    <TR>
      <TD>numberOfDistances</TD>
      <TD>Amount of distances to local and remote NUMA nodes</TD>
    </TR>
    <TR>
      <TD>processors</TD>
      <TD>List of CPU IDs in the NUMA domain</TD>
    </TR>
    <TR>
      <TD>distances</TD>
      <TD>Two dimensional list of distances to NUMA nodes in the system</TD>
    </TR>
    </TABLE></TD>
</TR>
</TABLE>

<H1>Function definitions for Lua NUMA topology module in the Lua API</H1>
\anchor getNumaInfo
<H2>getNumaInfo()</H2>
<P>Get information about the NUMA domains in the system</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>NUMA Info \ref lua_numainfo</TD>
</TR>
</TABLE>


<H2>putNumaInfo()</H2>
<P>Frees C struct NumaTopology. You can still use the lua_numainfo data structure<BR>If you call \ref getNumaInfo function again after calling this function, the NUMA topology information will be read again.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Return</TD>
  <TD>None</TD>
</TR>
</TABLE>

<H2>setMemInterleaved()</H2>
<P>Set the 'Interleaved' memory policy to allocate data only on given CPUs</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a nrThreads</TD>
      <TD>Amount of threads in the \a threads2Cpus list</TD>
    </TR>
    <TR>
      <TD>\a threads2Cpus</TD>
      <TD>List of thread to CPU relations</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Return</TD>
  <TD>None</TD>
</TR>
</TABLE>

<H2>nodestr_to_nodelist()</H2>
<P>Resolve the given node expression in NUMA affinity domain</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a node expression</TD>
      <TD>List of CPUs in NUMA node</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Return</TD>
  <TD><TABLE>
    <TR>
      <TD>\a nrThreads</TD>
      <TD>Amount of threads in the \a threads2Cpus list</TD>
    </TR>
    <TR>
      <TD>\a threads2Cpus</TD>
      <TD>List of thread to CPU relations</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

<H2>sockstr_to_socklist()</H2>
<P>Resolve the given socket expression in socket affinity domain</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a socket expression</TD>
      <TD>List of CPUs in socket affinity domain</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Return</TD>
  <TD><TABLE>
    <TR>
      <TD>\a nrThreads</TD>
      <TD>Amount of threads in the \a threads2Cpus list</TD>
    </TR>
    <TR>
      <TD>\a threads2Cpus</TD>
      <TD>List of thread to CPU relations</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

*/

/*! \page lua_AffinityInfo Thread affinity module

<H1>Data type definition for Lua thread affinity module in the Lua API</H1>
\anchor lua_affinityinfo
<H2>Affinity Info</H2>
<P>This structure is returned by \ref getAffinityInfo function<BR>It is similar to the C struct AffinityDomains</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a numberOfAffinityDomains</TD>
  <TD>Total amount of affinity domains in the system</TD>
</TR>
<TR>
  <TD>\a numberOfSocketDomains</TD>
  <TD>Amount of affinity domains for CPU sockets in the system</TD>
</TR>
<TR>
  <TD>\a numberOfNumaDomains</TD>
  <TD>Amount of affinity domains for NUMA domains in the system</TD>
</TR>
<TR>
  <TD>\a numberOfCacheDomains</TD>
  <TD>Amount of affinity domains for LLC domains in the system</TD>
</TR>
<TR>
  <TD>\a numberOfProcessorsPerSocket</TD>
  <TD>Amount of hardware threads for each CPU socket in the system</TD>
</TR>
<TR>
  <TD>\a numberOfCoresPerCache</TD>
  <TD>Amount of physical CPU cores for each LLC in the system</TD>
</TR>
<TR>
  <TD>\a numberOfProcessorsPerCache</TD>
  <TD>Amount of hardware threads for each LLC in the system</TD>
</TR>
<TR>
  <TD>\a domains</TD>
    <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>tag</TD>
      <TD>Tag identifiying the affinity domain</TD>
    </TR>
    <TR>
      <TD>numberOfCores</TD>
      <TD>Amount of physical CPU cores in the affinity domain</TD>
    </TR>
    <TR>
      <TD>numberOfProcessors</TD>
      <TD>Amount of hardware threads in the affinity domain</TD>
    </TR>
    <TR>
      <TD>processorList</TD>
      <TD>List with hardware thread IDs that are in the affinity domain</TD>
    </TR>
    </TABLE></TD>
</TR>
</TABLE>
<H1>Function definitions for Lua thread affinity module in the Lua API</H1>
\anchor getAffinityInfo
<H2>getAffinityInfo()</H2>
<P>Get information about the affinity domains in the system</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>NUMA Info \ref lua_affinityinfo</TD>
</TR>
</TABLE>
<H2>putAffinityInfo()</H2>
<P>Frees C struct AffinityDomains. You can still use the lua_affinityinfo data structure<BR>If you call \ref getAffinityInfo function again after calling this function, the thread affinity information will be read again.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>
\anchor pinProcess
<H2>pinProcess()</H2>
<P>Pins the current pocess to the given CPU ID</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>CPU to pin the process on</TD>
    </TR>
    <TR>
      <TD>\a silent</TD>
      <TD>Verbosity of pinning method</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>
*/


/*! \page lua_Perfmon Performance monitoring module
<H1>Data type definition for Lua performance monitoring module in the Lua API</H1>
\anchor lua_counterinfo
<H2>Event and Counter Info</H2>
<P>This structure is returned by \ref getEventsAndCounters function</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a Counters</TD>
  <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>Name</TD>
      <TD>Counter name as used by LIKWID</TD>
    </TR>
    <TR>
      <TD>Index</TD>
      <TD>Index of counter definition in internal list of counters</TD>
    </TR>
    <TR>
      <TD>Type</TD>
      <TD>ID number of counter type, use TypeName to get a human-readable name</TD>
    </TR>
    <TR>
      <TD>TypeName</TD>
      <TD>Name of counter type</TD>
    </TR>
    <TR>
      <TD>Options</TD>
      <TD>String with the options available for the counter</TD>
    </TR>
    </TABLE></TD>
</TR>
<TR>
  <TD>\a Events</TD>
  <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>Name</TD>
      <TD>Event name as used by LIKWID</TD>
    </TR>
    <TR>
      <TD>ID</TD>
      <TD>Event ID as defined by CPU vendor</TD>
    </TR>
    <TR>
      <TD>Umask</TD>
      <TD>Umask further restricting the event defined by ID</TD>
    </TR>
    <TR>
      <TD>Limit</TD>
      <TD>String containing the name(s) of registers the event can be programmed on</TD>
    </TR>
    <TR>
      <TD>Options</TD>
      <TD>String with the options available for the event</TD>
    </TR>
    </TABLE></TD>
</TR>
</TABLE>

\anchor lua_groupdata
<H2>Info about a performance group</H2>
<P>This structure is returned by \ref get_groupdata function</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>EventString</TD>
  <TD>Event set used for the performance group. Well formatted for \ref addEventSet function</TD>
</TR>
<TR>
  <TD>GroupString</TD>
  <TD>Name of the performance group</TD>
</TR>
<TR>
  <TD>LongDescription</TD>
  <TD>Description of the group. The 'LONG' section in the performance group file</TD>
</TR>
<TR>
  <TD>\a Events</TD>
  <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>Event ID</TD>
      <TD><TABLE>
      <TR>
        <TD>\a Event</TD>
        <TD>Name of event</TD>
      </TR>
      <TR>
        <TD>\a Counter</TD>
        <TD>LIKWID's name of the counter register</TD>
      </TR>
      </TABLE></TD>
    </TR>
    </TABLE></TD>
</TR>
<TR>
  <TD>\a Metrics</TD>
  <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>Metric ID</TD>
      <TD><TABLE>
      <TR>
        <TD>\a description</TD>
        <TD>Descriptive information of the metric</TD>
      </TR>
      <TR>
        <TD>\a formula</TD>
        <TD>Formula for calculating the metrics value</TD>
      </TR>
      </TABLE></TD>
    </TR>
    </TABLE></TD>
</TR>
</TABLE>


\anchor lua_pcidevinfo
<H2>Info about online PCI devices used for performance monitoring</H2>
<P>This structure is returned by \ref getOnlineDevices function</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a Name (used by LIKWID)</TD>
  <TD><TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>Name</TD>
      <TD>Name of PCI device</TD>
    </TR>
    <TR>
      <TD>Path</TD>
      <TD>Path to PCI device</TD>
    </TR>
    <TR>
      <TD>Type</TD>
      <TD>Human-readable name of the PCI device type</TD>
    </TR>
    <TR>
      <TD>TypeDescription</TD>
      <TD>Description about the PCI device</TD>
    </TR>
    </TABLE></TD>
</TR>
</TABLE>

<H1>Function definitions for Lua performance monitoring module in the Lua API</H1>
\anchor init
<H2>init()</H2>
<P>Initializes the Perfmon module of LIKWID, like opening the MSR files and check the PCI devices<BR>If in access daemon mode, a single daemon instance is started to forward measurements on all given CPUs</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a nrThreads</TD>
      <TD>Number of CPUs that should be measured</TD>
    </TR>
    <TR>
      <TD>\a thread2Cpus</TD>
      <TD>List with length \a nrThreads containing the relation between thread number and measured CPU</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Error code, 0 for success</TD>
</TR>
</TABLE>

\anchor addEventSet
<H2>addEventSet()</H2>
<P>Creates the internal management structures for the given event set. Checks the registers and if needed PCI device access<BR>The \ref init function as to be called previously</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a eventSet</TD>
      <TD>String composed of all events in the event set. Format is Event1:Counter1(:Option11:Options12:...),Event2:Counter2...</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>The group ID of the added event set</TD>
</TR>
</TABLE>


\anchor setupCounters
<H2>setupCounters()</H2>
<P>Setup the config registers to measure the events defined by group</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a groupID</TD>
      <TD>ID of group returned by \ref addEventSet function.</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Error code, 0 for success</TD>
</TR>
</TABLE>

\anchor startCounters
<H2>startCounters()</H2>
<P>Starts the perfmon group previously set up with \ref setupCounters function.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Error code, 0 for success</TD>
</TR>
</TABLE>

\anchor stopCounters
<H2>stopCounters()</H2>
<P>Stops the perfmon group and reads the counters into the internal result section. Use the \ref getResult or \ref getResults functions to get the results.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Error code, 0 for success</TD>
</TR>
</TABLE>

\anchor readCounters
<H2>readCounters()</H2>
<P>Reads the perfmon group into the internal result section. Use the \ref getResult or \ref getResults functions to get the results.<BR>The counters will be stopped shortly and started after reading to exclude the LIKWID code from measurements.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Error code, 0 for success</TD>
</TR>
</TABLE>

\anchor switchGroup
<H2>switchGroup()</H2>
<P>Switches the currently active group in the perfmon module. If the given group ID does not exist, it fallbacks to group ID 1.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a newgroup</TD>
      <TD>Switch active group to \a newgroup</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Error code, 0 for success</TD>
</TR>
</TABLE>

\anchor finalize
<H2>finalize()</H2>
<P>Destroy internal structures and clean all used registers</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Always 0</TD>
</TR>
</TABLE>

\anchor getResult
<H2>getResult()</H2>
<P>Get result for a group, event, thread combination. All options must be given</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a groupID</TD>
      <TD>Return result from group defined by \a groupID</TD>
    </TR>
    <TR>
      <TD>\a eventID</TD>
      <TD>Return result for event with \a eventID. Position in string given to \ref addEventSet function</TD>
    </TR>
    <TR>
      <TD>\a threadID</TD>
      <TD>Return result for thread with \a threadID as defined by the \a thread2Cpus input parameter for \ref init function</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Result</TD>
</TR>
</TABLE>

\anchor getResults
<H2>getResults()</H2>
<P>Get all results for all group, event, thread combinations</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Three-dimensional list with results. First dim. is groups, second dim. is events and third dim. are the threads</TD>
</TR>
</TABLE>

\anchor getMarkerResults
<H2>getMarkerResults()</H2>
<P>Get the results for an output file written by \ref MarkerAPI</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a filename</TD>
      <TD>Filename written by \ref MarkerAPI</TD>
    </TR>
    <TR>
      <TD>\a group_list</TD>
      <TD>List of defined groups</TD>
    </TR>
    <TR>
      <TD>\a num_cpus</TD>
      <TD>Amount of defined CPUs. Is used just used for checking if the \ref MarkerAPI run is valid. If LIKWID_MARKER_THREADINIT is not called properly the tests will fail</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Four-dimensional list with results. First dim. is groups, second dim. is management regions, and third dim. are the events and fourth dim. are the threads</TD>
</TR>
</TABLE>

\anchor getEventsAndCounters
<H2>getEventsAndCounters()</H2>
<P>Get a list containing all event and counter definitions</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Event and counter info like \ref lua_counterinfo</TD>
</TR>
</TABLE>

\anchor getOnlineDevices
<H2>getOnlineDevices()</H2>
<P>Get a list containing all online PCI devices</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>PCI device info like \ref lua_pcidevinfo</TD>
</TR>
</TABLE>

\anchor getNumberOfGroups
<H2>getNumberOfGroups()</H2>
<P>Returns the number of event sets (groups) added to the perfmon module</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Amount of configured groups</TD>
</TR>
</TABLE>

\anchor getIdOfActiveGroup
<H2>getIdOfActiveGroup()</H2>
<P>Returns the ID of the currently active group</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>ID of active group</TD>
</TR>
</TABLE>

\anchor getRuntimeOfGroup
<H2>getRuntimeOfGroup()</H2>
<P>Returns the measurement time of the given groupID</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a groupID</TD>
      <TD>Return the measurement time for group defined by \a groupID</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Measurement time of group</TD>
</TR>
</TABLE>

\anchor getNumberOfEvents
<H2>getNumberOfEvents()</H2>
<P>Returns the amount of events for the given groupID</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a groupID</TD>
      <TD>Return the measurement time for group defined by \a groupID</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Amount of events in group</TD>
</TR>
</TABLE>

\anchor getNumberOfThreads
<H2>getNumberOfThreads()</H2>
<P>Returns the number of threads as given to \ref init function</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Amount of measurement threads</TD>
</TR>
</TABLE>

\anchor get_groups
<H2>get_groups()</H2>
<P>Returns a list of all performance groups in \a groupfolder</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a architecture</TD>
      <TD>Short name of architecture. Can be found in CPU info \ref lua_cpuinfo as \a short_name</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a numerOfGroups</TD>
      <TD>Amount of groups in \a groupfolder for given \a architecture</TD>
    </TR>
    <TR>
      <TD>\a groups</TD>
      <TD>List with the names of all performance groups in \a groupfolder for given \a architecture</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

\anchor get_groupdata
<H2>get_groupdata()</H2>
<P>Read in the performance group \a group</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a group</TD>
      <TD>Get group data for \a group </TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a groupdata</TD>
      <TD>Structure with all group information found for the performance group \a group, see \ref lua_groupdata</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

*/

/*! \page lua_PowerInfo Power and Energy monitoring module
<H1>Data type definition for Lua power and energy monitoring module in the Lua API</H1>
\anchor lua_powerinfo
<H2>Power Information</H2>
<P>This structure is returned by \ref getPowerInfo function<BR>The nested list structure is almost similar to the C struct CpuTopology.</P>
<TABLE>
<TR>
  <TH>Membername</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>\a hasRAPL</TD>
  <TD>If set, the system supports power readings through the RAPL interface</TD>
</TR>
<TR>
  <TD>\a baseFrequency</TD>
  <TD>Nominal clock frequency of the system</TD>
</TR>
<TR>
  <TD>\a minFrequency</TD>
  <TD>Minimal supported clock frequency of the system</TD>
</TR>
<TR>
  <TD>\a powerUnit</TD>
  <TD>Multiplier for power readings</TD>
</TR>
<TR>
  <TD>\a timeUnit</TD>
  <TD>Multiplier for time readings from RAPL</TD>
</TR>
<TR>
  <TD>\a turbo</TD>
    <TD>
    <TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>\a numSteps</TD>
      <TD>Amount of turbo mode steps</TD>
    </TR>
    <TR>
      <TD>\a steps</TD>
      <TD>List containing the turbo mode steps</TD>
    </TR>
    </TABLE></TD>
</TR>
<TR>
  <TD>\a domains</TD>
    <TD>
    <TABLE>
    <TR>
      <TH>Membername</TH>
      <TH>Comment</TH>
    </TR>
    <TR>
      <TD>\a RAPL domain</TD>
      <TD>
        <TABLE>
        <TR>
          <TH>Typename</TH>
          <TH>comment</TH>
        </TR>
        <TR>
          <TD>ID</TD>
          <TD>Type of domain (PKG, PP0, PP1, DRAM)</TD>
        </TR>
        <TR>
          <TD>energyUnit</TD>
          <TD>Multiplier for energy readings for RAPL domain</TD>
        </TR>
        <TR>
          <TD>supportStatus</TD>
          <TD>RAPL domain has a status register to read energy values</TD>
        </TR>
        <TR>
          <TD>supportPerf</TD>
          <TD>RAPL domain has a perf register</TD>
        </TR>
        <TR>
          <TD>supportPolicy</TD>
          <TD>RAPL domain has a policy register to define a global energy policy</TD>
        </TR>
        <TR>
          <TD>supportLimit</TD>
          <TD>RAPL domain has a policy register to define a limit for the energy consumption</TD>
        </TR>
        <TR>
          <TD>supportInfo</TD>
          <TD>RAPL domain has a policy register to define a limit for the energy consumption</TD>
        </TR>
        <TR>
          <TD>tdp</TD>
          <TD>Thermal Design Power<BR>Only if supportInfo is set</TD>
        </TR>
        <TR>
          <TD>minPower</TD>
          <TD>Minimal power consumption for the RAPL domain<BR>Only if supportInfo is set</TD>
        </TR>
        <TR>
          <TD>maxPower</TD>
          <TD>Maximal power consumption for the RAPL domain<BR>Only if supportInfo is set</TD>
        </TR>
        <TR>
          <TD>maxTimeWindow</TD>
          <TD>Maximal duration between updates of the RAPL status registers<BR>Only if supportInfo is set</TD>
        </TR>
        </TABLE>
        </TD>
    </TR>
    </TABLE>
    </TD>
</TR>
</TABLE>
<H1>Function definitions for Lua power and energy monitoring module in the Lua API</H1>
\anchor getPowerInfo
<H2>getPowerInfo()</H2>
<P>Get information about the RAPL interface in the system</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Power Info \ref lua_powerinfo</TD>
</TR>
</TABLE>
\anchor putPowerInfo
<H2>putPowerInfo()</H2>
<P>Frees C struct PowerInfo. You can still use the lua_powerinfo data structure<BR>If you call \ref getPowerInfo function again after calling this function, the power information struct will be filled again.</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor startPower
<H2>startPower()</H2>
<P>Start measuring given RAPL domain on given CPU</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>Start the power measurement on CPU \a cpuID</TD>
    </TR>
    <TR>
      <TD>\a domainID</TD>
      <TD>Start the power measurement for domain domainID<BR>Possible values: 0=PKG, 1=PP0, 2=PP1, 3=DRAM</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Power value at start</TD>
</TR>
</TABLE>

\anchor stopPower
<H2>stopPower()</H2>
<P>Stop measuring given RAPL domain on given CPU</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>Stop the power measurement on CPU \a cpuID</TD>
    </TR>
    <TR>
      <TD>\a domainID</TD>
      <TD>Stop the power measurement for domain domainID<BR>Possible values: 0=PKG, 1=PP0, 2=PP1, 3=DRAM</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Power value at stop</TD>
</TR>
</TABLE>


\anchor printEnergy
<H2>printEnergy()</H2>
<P></P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a before</TD>
      <TD>Result from \ref startPower function</TD>
    </TR>
    <TR>
      <TD>\a after</TD>
      <TD>Result from \ref stopPower function</TD>
    </TR>
    <TR>
      <TD>\a domainID</TD>
      <TD>Print the power result for domain domainID<BR>Possible values: 0=PKG, 1=PP0, 2=PP1, 3=DRAM</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Power value at stop</TD>
</TR>
</TABLE>

\anchor limitGet
<H2>limitGet() (EXPERIMENTAL)</H2>
<P>Get the current limit in the limit register of domain. The limit is defined as maximal power consumption in a time window</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>Get limit for CPU \a cpuID</TD>
    </TR>
    <TR>
      <TD>\a domainID</TD>
      <TD>Get limit for domain domainID<BR>Possible values: 0=PKG, 1=PP0, 2=PP1, 3=DRAM</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a power</TD>
      <TD>Power limit value</TD>
    </TR>
    <TR>
      <TD>\a time</TD>
      <TD>Duration of time window</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>


\anchor limitSet
<H2>limitSet() (EXPERIMENTAL)</H2>
<P></P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>Set limit for CPU \a cpuID</TD>
    </TR>
    <TR>
      <TD>\a domainID</TD>
      <TD>Set limit for domain domainID<BR>Possible values: 0=PKG, 1=PP0, 2=PP1, 3=DRAM</TD>
    </TR>
    <TR>
      <TD>\a power</TD>
      <TD>Set power value to \a power</TD>
    </TR>
    <TR>
      <TD>\a time</TD>
      <TD>Set time window value to \a time</TD>
    </TR>
    <TR>
      <TD>\a clamp</TD>
      <TD>Should the limit be clamped or can it sometimes exceed the power limit if in total the limit is satisfied</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Error code, 0 for success</TD>
</TR>
</TABLE>

\anchor limitState
<H2>limitState() (EXPERIMENTAL)</H2>
<P>Get the state of the limit</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>Get the state on CPU \a cpuID</TD>
    </TR>
    <TR>
      <TD>\a domainID</TD>
      <TD>Get the state for domain domainID<BR>Possible values: 0=PKG, 1=PP0, 2=PP1, 3=DRAM</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>State, 0 for off, 1 for on</TD>
</TR>
</TABLE>
*/

/*! \page lua_ThermalInfo Thermal monitoring module
<H1>Data type definition for Lua thermal monitoring module in the Lua API</H1>
<H1>Function definitions for Lua thermal monitoring module in the Lua API</H1>
\anchor initTemp
<H2>initTemp()</H2>
<P>Initialize the thermal measurements on given CPU</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>Initialize thermal readings on CPU \a cpuID</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor initTemp
<H2>readTemp()</H2>
<P>Measure the temperature on given CPU</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a cpuID</TD>
      <TD>Read the temperature on CPU \a cpuID</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Temperature</TD>
</TR>
</TABLE>
*/

/*! \page lua_Timer Time measurement module
<H1>Data type definition for Lua time measurement module in the Lua API</H1>
<H1>Function definitions for Lua time measurement module in the Lua API</H1>
\anchor getCpuClock
<H2>getCpuClock()</H2>
<P>Returns the nominal clock speed</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Clock speed in Hz</TD>
</TR>
</TABLE>

\anchor startClock
<H2>startClock()</H2>
<P>Start the TSC clock</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Current timestamp</TD>
</TR>
</TABLE>

\anchor stopClock
<H2>stopClock()</H2>
<P>Stop the TSC clock</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Current timestamp</TD>
</TR>
</TABLE>

\anchor getClockCycles
<H2>getClockCycles()</H2>
<P>Return the amount of cycles between start and stop timestamps</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a start</TD>
      <TD>Start timestamp</TD>
    </TR>
    <TR>
      <TD>\a stop</TD>
      <TD>Stop timestamp</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Amount of cycles between start and stop</TD>
</TR>
</TABLE>

\anchor getClock
<H2>getClock()</H2>
<P>Return the time in seconds between start and stop timestamps</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a start</TD>
      <TD>Start timestamp</TD>
    </TR>
    <TR>
      <TD>\a stop</TD>
      <TD>Stop timestamp</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Time in seconds between start and stop</TD>
</TR>
</TABLE>

\anchor sleep
<H2>sleep()</H2>
<P>Sleep for specified amount of seconds</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a seconds</TD>
      <TD>Sleep for seconds</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Remaining time to sleep. >0 if sleep is interrupted</TD>
</TR>
</TABLE>

\anchor usleep
<H2>usleep()</H2>
<P>Sleep for at least the specified amount of microseconds</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a usecs</TD>
      <TD>Sleep for microseconds. \a usec must be in range 1 to 999999</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>Status of usleep. If interrupted the status is != 0</TD>
</TR>
</TABLE>

*/

/*! \page lua_MemSweep Memory sweeping module
<H1>Data type definition for Lua memory sweeping module in the Lua API</H1>
<H1>Function definitions for Lua memory sweeping module in the Lua API</H1>
\anchor memSweep
<H2>memSweep()</H2>
<P>Sweep the memory and LLC for given threads</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a nrThreads</TD>
      <TD>Amount of threads in the \a threads2Cpus list</TD>
    </TR>
    <TR>
      <TD>\a Cpus</TD>
      <TD>List with thread to CPU relations</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor memSweepDomain
<H2>memSweepDomain()</H2>
<P>Sweep the memory and LLC for a given NUMA domain</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a domainID</TD>
      <TD>Sweep the memory and LLC at the NUMA domain specified by \a domainID</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>
*/

/*! \page lua_Misc Miscellaneous functions module
<H1>Data type definition for Lua miscellaneous functions module in the Lua API</H1>
<H1>Function definitions for Lua miscellaneous functions module in the Lua API</H1>
\anchor startProgram
<H2>startProgram()</H2>
<P>Start an executable in a new thread</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a Exec</TD>
      <TD>String containing the executable and its options</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>PID of newly created thread</TD>
</TR>
</TABLE>

\anchor checkProgram
<H2>checkProgram()</H2>
<P>Check if the executable is running</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>True/False</TD>
</TR>
</TABLE>

\anchor killProgram
<H2>killProgram()</H2>
<P>Kill the executable with SIGTERM</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a PID</TD>
      <TD>PID to send the SIGTERM signal</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>


\anchor setenv
<H2>setenv()</H2>
<P>Set environment variable. Lua only provides getenv()</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a Name</TD>
      <TD>Name of environment variable</TD>
    </TR>
    <TR>
      <TD>\a Value</TD>
      <TD>Value for the environment variable</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor getpid
<H2>getpid()</H2>
<P>Get the PID of the current process</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>PID number</TD>
</TR>
</TABLE>

\anchor access
<H2>access()</H2>
<P>Check the file existance for a given filepath</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a Filepath</TD>
      <TD>Name of Filepath to check for existance</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>PID number</TD>
</TR>
</TABLE>

\anchor msr_available
<H2>msr_available()</H2>
<P>Check whether the msr files are available. Basically checks whether the msr kernel module is loaded properly</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>True/False</TD>
</TR>
</TABLE>

\anchor gethostname
<H2>gethostname()</H2>
<P>Returns the hostname of the system in short format</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a Hostname</TD>
      <TD>Hostname in short format</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

\anchor getjid
<H2>getjid()</H2>
<P>Returns the job ID if running in a batch environment. Basically reads the <CODE>PBS_JOBID</CODE> environment variable</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a Job ID</TD>
      <TD>Job ID or 'X' if not in batch environment</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

\anchor getMPIrank
<H2>getMPIrank()</H2>
<P>Returns the MPI rank of the current process. Basically read the <CODE>PMI_RANK</CODE> and <CODE>OMPI_COMM_WORLD_RANK</CODE> environment variables</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD>None</TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a MPI Rank</TD>
      <TD>MPI rank or 'X' if not in MPI environment</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>
*/


/*! \page lua_InputOutput Input and output functions module
<H1>Data type definition for Lua output functions module in the Lua API</H1>
<H1>Function definitions for Lua output functions module in the Lua API</H1>
\anchor getopt
<H2>getopt()</H2>
<P>Read commandline parameters and split them to the given options. The version LIKWID uses was originally taken from the web but extended to talk short '-o' and long options "--option". It returns an iterator for the commandline options.<BR>Basic usage:<BR></P>
<CODE>
for opt,arg in likwid.getopt(arg, {"n:","h"}) do<BR>
&nbsp;&nbsp;&nbsp;&nbsp;if (type(arg) == "string") then<BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;local s,e = arg:find("-")<BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if s == 1 then<BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print(string.format("ERROR: Argmument %s to option -%s starts with invalid character -.", arg, opt))<BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print("ERROR: Did you forget an argument to an option?")<BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;os.exit(1)<BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end<BR>
&nbsp;&nbsp;&nbsp;&nbsp;end<BR>
&nbsp;&nbsp;&nbsp;&nbsp;--parse options<BR>
end<BR>
</CODE><BR>
The option 'n' takes an argument, specified by the ':'. If found the option argument for option 'h' is true. The type check for the argument is recommended to get errors with an argument awaiting option where the argument is missing.
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a commandline</TD>
      <TD>Normally, Lua saves the commandline parameters in variable 'arg'</TD>
    </TR>
    <TR>
      <TD>\a optionlist</TD>
      <TD>List of options that should be recognized. Options with ':' as last character need an argument</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a option</TD>
      <TD>Option string found on the commandline without leading '-'</TD>
    </TR>
    <TR>
      <TD>\a argument</TD>
      <TD>Argument to the \a option. If \a option does not require an argument, true or false is returned in \a argument</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

\anchor parse_time
<H2>parse_time()</H2>
<P>Parses time interval describing strings like 2s, 100ms or 250us</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a timestr</TD>
      <TD>String describing a time interval</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD><TABLE>
    <TR>
      <TD>\a duration</TD>
      <TD>Time string \a timestr resolved to usecs</TD>
    </TR>
  </TABLE></TD>
</TR>
</TABLE>

\anchor printtable
<H2>printtable()</H2>
<P>Prints the given two dimensional table as fancy ASCII table. For CSV output use \ref printcsv</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a table</TD>
      <TD>Two dimensional list with table entries. First dim. are columns and second dim. the lines</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor printcsv
<H2>printcsv()</H2>
<P>Prints the given two dimensional table in CSV format. For ASCII table output see \ref printtable</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a table</TD>
      <TD>Two dimensional list with table entries. First dim. are columns and second dim. the lines</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor stringsplit
<H2>stringsplit()</H2>
<P>Splits the given string at separating character</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a str</TD>
      <TD>String to split</TD>
    </TR>
    <TR>
      <TD>\a sSeparator</TD>
      <TD>String with separating character</TD>
    </TR>
    <TR>
      <TD>\a nMax</TD>
      <TD>Split string maximally \a nMax times (optional)</TD>
    </TR>
    <TR>
      <TD>\a bRegexp</TD>
      <TD>Lua RegEx string for separation (optional)</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>List of \a str splitted at \a sSeparator or \a bRegexp</TD>
</TR>
</TABLE>

\anchor printOutput
<H2>printOutput()</H2>
<P>Prints results</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a groups</TD>
      <TD>List of groups for printing</TD>
    </TR>
    <TR>
      <TD>\a results</TD>
      <TD>List of results as returned by \ref getResults function</TD>
    </TR>
    <TR>
      <TD>\a groupData</TD>
      <TD>List of group data structures</TD>
    </TR>
    <TR>
      <TD>\a cpulist</TD>
      <TD>List of thread ID to CPU ID relations</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor print_markerOutput
<H2>print_markerOutput()</H2>
<P>Prints results of a Marker API run. This is different to \ref printOutput because we have to resolve the measurement regions</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a groups</TD>
      <TD>List of groups for printing</TD>
    </TR>
    <TR>
      <TD>\a results</TD>
      <TD>List of results as returned by \ref getMarkerResults function</TD>
    </TR>
    <TR>
      <TD>\a groupData</TD>
      <TD>List of group data structures</TD>
    </TR>
    <TR>
      <TD>\a cpulist</TD>
      <TD>List of thread ID to CPU ID relations</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>


\anchor addSimpleAsciiBox
<H2>addSimpleAsciiBox()</H2>
<P>Add a simple ASCII box with given label to box container. This function is only used by \ref likwid-topology</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a container</TD>
      <TD>Box container containing all boxes</TD>
    </TR>
    <TR>
      <TD>\a lineIdx</TD>
      <TD>Add box at line index \a lineIdx</TD>
    </TR>
    <TR>
      <TD>\a colIdx</TD>
      <TD>Add box at column index \a colIdx</TD>
    </TR>
    <TR>
      <TD>\a label</TD>
      <TD>Content of the box</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor addJoinedAsciiBox
<H2>addJoinedAsciiBox()</H2>
<P>Add a joined ASCII box with given label to box container. Joined boxes can span the space of multiple simple boxes. This function is only used by \ref likwid-topology</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a container</TD>
      <TD>Box container containing all boxes</TD>
    </TR>
    <TR>
      <TD>\a lineIdx</TD>
      <TD>Add box at line index \a lineIdx</TD>
    </TR>
    <TR>
      <TD>\a startColIdx</TD>
      <TD>Start joined box at column index \a startColIdx</TD>
    </TR>
    <TR>
      <TD>\a endColIdx</TD>
      <TD>End joined box at column index \a endColIdx</TD>
    </TR>
    <TR>
      <TD>\a label</TD>
      <TD>Content of the box</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>

\anchor printAsciiBox
<H2>printAsciiBox()</H2>
<P>Print the box container previously filled with \ref addSimpleAsciiBox and \ref addJoinedAsciiBox. This function is only used by \ref likwid-topology</P>
<TABLE>
<TR>
  <TH>Direction</TH>
  <TH>Data type(s)</TH>
</TR>
<TR>
  <TD>Input Parameter</TD>
  <TD><TABLE>
    <TR>
      <TD>\a container</TD>
      <TD>Box container containing all boxes</TD>
    </TR>
  </TABLE></TD>
</TR>
<TR>
  <TD>Returns</TD>
  <TD>None</TD>
</TR>
</TABLE>
*/
