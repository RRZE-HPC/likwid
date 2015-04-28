/*! \page likwid-pin <CODE>likwid-pin</CODE>

<H1>Information</H1>
<CODE>likwid-pin</CODE> is a command line application to pin a sequential or multithreaded application to dedicated processors. It can be used as replacement for taskset.
Opposite to taskset no affinity mask but single processors are specified. For multithreaded applications based on the <A HREF="https://computing.llnl.gov/tutorials/pthreads/"><CODE>pthreads</CODE></A> library the <CODE>pthread_create</CODE> library call is overloaded through <CODE>LD_PRELOAD</CODE> and each created thread is pinned to a dedicated processor as specified in the pinning list. Per default every generated thread is pinned to the core in the order of calls to <CODE>pthread_create</CODE>. It is possible to skip single threads.<BR>
<BR>
For OpenMP implementations, GCC and ICC compilers are explicitly supported. Clang's OpenMP backend should also work as it is built on top of Intel's OpenMP runtime library. Others may also work.<BR>
<BR>
<CODE>likwid-pin</CODE> sets the environment variable <CODE>OMP_NUM_THREADS</CODE> for you if not already present. It will set as many threads as present in the pin expression.  Be aware that with <A HREF="https://computing.llnl.gov/tutorials/pthreads/"><CODE>pthreads</CODE></A> the parent thread is always pinned. If you create for example 4 threads with <CODE>pthread_create</CODE> and do not use the parent process as worker you still have to provide <CODE>num_threads + 1</CODE> processor ids.<BR>
<BR>
<CODE>likwid-pin</CODE> supports different numberings for pinning. Per default physical numbering of the cores is used. This is the numbering also \ref likwid-topology reports. But also logical numbering inside the node or the sockets can be used. For details look at \ref CPU_expressions. <!--If using with a N (e.g. -c N:0-6) the cores are logical numbered over the whole node. Physical cores come first. If a system e.g. has 8 cores with 16 SMT threads with -c N:0-7 you get all physical cores.  If you specify -c N:0-15 you get all physical cores and all SMT threads. With S you can specify logical numberings inside sockets, again physical cores come first. You can mix different domains with a @. <CODE>-c S0:0-3\@S2:2-3</CODE> you pin thread 0-3 to logical cores 0-3 on socket 0 and threads 4-6 on logical cores 2-3 on socket 2.--><BR>

For applications where first touch policy on NUMA systems cannot be employed <CODE>likwid-pin</CODE> can be used to turn on interleave memory placement. This can significantly speed up the performance of memory bound multi threaded codes. All NUMA nodes the user pinned threads to are used for interleaving.

<H1>Options</H1>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>-h, --help</TD>
  <TD>Print help message</TD>
</TR>
<TR>
  <TD>-v, --version</TD>
  <TD>Print version information</TD>
</TR>
<TR>
  <TD>-V, --verbose &lt;level&gt;</TD>
  <TD>Verbose output during execution for debugging. Possible values for &lt;level&gt;:
  <TABLE>
    <TR>
      <TD>0</TD>
      <TD>Output only errors</TD>
    </TR>
    <TR>
      <TD>1</TD>
      <TD>Output some information</TD>
    </TR>
    <TR>
      <TD>2</TD>
      <TD>Output detailed information</TD>
    </TR>
    <TR>
      <TD>3</TD>
      <TD>Output developer information</TD>
    </TR>
  </TABLE>
  </TD>
</TR>
<TR>
  <TD>-c &lt;arg&gt;</TD>
  <TD>Define the CPUs that the application should be pinned on. LIKWID provides an intuitive and feature-rich syntax for CPU expressions.<BR>See section \ref CPU_expressions for details.</TD>
</TR>
<TR>
  <TD>-S, --sweep</TD>
  <TD>Sweep memory and clean LLC of NUMA domains used by the given CPU expression</TD>
</TR>
<TR>
  <TD>-i</TD>
  <TD>Activate interleaved memory policy for NUMA domains used by the given CPU expression</TD>
</TR>
<TR>
  <TD>-p</TD>
  <TD>Print the thread affinity domains. If -c is set on the commandline, the affinity domains filled only with the given CPUs are printed.</TD>
</TR>
<TR>
  <TD>-q, --quiet</TD>
  <TD>Don't print infos of the pinning process</TD>
</TR>
<TR>
  <TD>-s, --skip &lt;arg&gt;</TD>
  <TD>'arg' must be a bitmask in hex. Threads with the ID equal to a set bit in bitmask will be skipped during pinning<BR>Example: 0x1 = Thread 0 is skipped.</TD>
</TR>
<TR>
  <TD>-d</TD>
  <TD>Set the delimiter for the output of -p. Default is ','</TD>
</TR>
</TABLE>

\anchor thread_affinity_domains
<H1>Affinity Domains</H1>
While gathering the system topology, LIKWID groups the CPUs into so-called thread affinity domains. A thread affinity domain is a group of CPU IDs that are related to some kind of central entity of the system. The most common domain is the node domain (<CODE>N</CODE>) that contains all CPUs available in the system. Other domains group the CPUs according to socket, LLC or NUMA node relation. <CODE>likwid-pin</CODE> prints out all available affinity domains with the commandline option <CODE>-p</CODE>.The following list introduces all affinity domains with the used domain names:
<TABLE>
<TR>
  <TH>Domain name</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD><CODE>N</CODE></TD>
  <TD>Includes all CPUs in the system</TD>
</TR>
<TR>
  <TD><CODE>S&lt;number&gt;</CODE></TD>
  <TD>Includes all CPUs that reside on CPU socket x</TD>
</TR>
<TR>
  <TD><CODE>C&lt;number&gt;</CODE></TD>
  <TD>Includes all CPUs that share the same LLC with ID <CODE>&lt;number&gt;</CODE>.<BR>This domain often contains the same CPUs as the <CODE>S&lt;number&gt;</CODE> domain because many CPU socket have a LLC shared by all CPUs of the socket</TD>
</TR>
<TR>
  <TD><CODE>M&lt;number&gt;</CODE></TD>
  <TD>Includes all CPUs that are attached to the same NUMA memory domain</TD>
</TR>
</TABLE>

\anchor CPU_expressions
<H1>CPU expressions</H1>
One outstanding feature of LIKWID are the CPU expressions which are resolved to the CPUs in the actual system. There are multiple formats that can be chosen where each offers a convenient way to select the desired CPUs for execution or measurement. The CPU expressions are used for <CODE>likwid-pin</CODE> as well as \ref likwid-perfctr. This section introduces the 4 formats and gives examples.

<H3>Physical numbering:</H3>
The first and probably most natural way of defining a list of CPUs is the usage of the physical numbering, similar to the numbering of the operating system and the IDs printed by \ref likwid-topology. The desired CPU IDs can be set as comma-separated list, as range or a combination of both.
<UL>
<LI><CODE>-c 1</CODE><BR>
Run only on CPU with ID 1
</LI>
<LI><CODE>-c 1,4</CODE><BR>
Run on CPUs with ID 1 and 4
</LI>
<LI><CODE>-c 1-3</CODE><BR>
Run on CPUs ranging from ID 1 to ID 3, hence CPUs 1,2,3
</LI>
<LI><CODE>-c 0,1-3</CODE><BR>
Run on CPU with ID 0 and the CPU range starting from ID 1 to ID3, hence 0,1,2,3
</LI>
</UL>
<H3>Logical numbering:</H3>
Besides the enumeration of physical CPU IDs, LIKWID supports the logical numbering inside of an affinity domain. For logical selection, the indicies inside of the desired affinity domain has to be given on the commandline. The logical numbering can be selected by prefixing the cpu expression with <CODE>L:</CODE>. The format is <CODE>L:&lt;indices&gt;</CODE> assuming affinity domain <CODE>N</CODE> or <CODE>L:&lt;affinity domain&gt;:&lt;indices&gt;</CODE>. Moreover, it is automatically activated if working inside of a CPU set (e.g. cgroups). For the examples we assume that the node affinity domain looks like this: <CODE>0,4,1,5,2,6,3,7</CODE>:
<UL>
<LI><CODE>-c L:1</CODE><BR>
Run only on CPU 0, the first entry in the affinity domain <CODE>N</CODE>
</LI>
<LI><CODE>-c L:1,4</CODE><BR>
Run on CPUs 0 and 5, the first and fifth entry in the affinity domain <CODE>N</CODE>
</LI>
<LI><CODE>-c L:1-3</CODE><BR>
Run on CPUs ranging from index 1 to index 3 in the affinity domain <CODE>N</CODE>, hence CPUs 0,4,1
</LI>
<LI><CODE>-c L:N:1,4-6</CODE><BR>
Run on CPUs with index 1 and the range of indices from 4 to 6 in given affinity domain <CODE>N</CODE>, hence CPUs 0,5,2,6
</LI>
</UL>
<B>Note</B>: List indicies in Lua start with 1!
<H3>Numbering by expression:</H3>
The most powerful format is probably the expression format. The format combines the input values for a selection function in a convenient way. In order to activate the expression format, the CPU string must be prefixed with <CODE>E:</CODE>. The basic format is <CODE>E:&lt;affinity domain&gt;:&lt;numberOfThreads&gt;</CODE> which selects simply the given <CODE>&lt;numberOfThreads&gt;</CODE> in the supplied <CODE>&lt;affinity domain&gt;</CODE>. The extended format is <CODE>E:&lt;affinity domain&gt;:&lt;numberOfThreads&gt;:&lt;chunksize&gt;:&lt;stride&gt;</CODE> and it selects the given <CODE>&lt;numberOfThreads&gt;</CODE> in the supplied <CODE>&lt;affinity domain&gt;</CODE> but takes <CODE>&lt;chunksize&gt;</CODE> threads in row with a distance of <CODE>&lt;stride&gt;</CODE>. For the examples we assume that the node affinity domain looks like this: <CODE>0,4,1,5,2,6,3,7</CODE>:
<UL>
<LI><CODE>-c E:N:1</CODE><BR>
Selects the first entry in the node affinity domain, thus CPU 0
</LI>
<LI><CODE>-c E:N:2</CODE><BR>
Selects the first two entries in the node affinity domain, thus CPUs 0 and 4
</LI>
<LI><CODE>-c E:N:2:1:2</CODE><BR>
Selects 1 CPU in a row and skips 2 entries twice, thus we get CPUs 0 and 1
</LI>
<LI><CODE>-c E:N:4:2:4</CODE><BR>
Selects in total 4 CPUs, 2 in a row with a stride of 4, thus CPUs 0,4,2,6
</LI>
</UL>
<H3>Scatter expression:</H3>
The scatter expression distributes the threads evenly over the desired affinity domains. In contrast to the previous selection methods, the scatter expression schedules threads over multiple affinity domains. Although you can also select <CODE>N</CODE> as scatter domain, the intended domains are <CODE>S</CODE>, <CODE>C</CODE> and <CODE>M</CODE>. The scattering selects physical cores first. For the examples we assume that the socket affinity domain looks like this: <CODE>S0 = 0,4,1,5</CODE> and <CODE>S1 = 2,6,3,7</CODE>, hence 8 hardware threads on a system with 2 SMT threads per CPU core.
<UL>
<LI><CODE>-c S:scatter</CODE><BR>
The resulting CPU list is 0,2,1,3,4,6,5,7
</LI>
<LI><CODE>-c M:scatter</CODE><BR>
Scatter the threads evenly over all NUMA memory domains. A kind of interleaved thread policy.
</LI>
</UL>
*/
