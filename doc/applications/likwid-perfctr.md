/*! \page likwid-perfctr <CODE>likwid-perfctr</CODE>

<H1>Information</H1>
<CODE>likwid-perfctr</CODE> is a lightweight command line application to configure and read out hardware performance monitoring data
on supported x86 processors. It can measure either as wrapper without changing the measured application
or with \ref Marker_API functions inside the code, which will turn on and off the counters. Moreover, there are the timeline and stethoscope mode.
There are preconfigured performance groups with useful event sets and derived metrics. Additonally, arbitrary events can be measured with
custom event sets. The \ref Marker_API can measure mulitple named regions and the results are accumulated over multiple region calls.
<P>
<B>Note</B> that <CODE>likwid-perfctr</CODE> measures all events on the specified CPUs and not only the context of the executable. On a highly loaded system it will be hard to determine which part of the given application caused the counter increment. Moreover, it is necessary to ensure that processes and threads are pinned to dedicated resources. You can either pin the application yourself or use the builtin pin functionality.

<H1>Options</H1>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>-h, --help</TD>
  <TD>Print help message.</TD>
</TR>
<TR>
  <TD>-v, --version</TD>
  <TD>Print version information.</TD>
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
  <TD>-i, --info</TD>
  <TD>Print \a CPUID information about processor and about Intel Performance Monitoring features.</TD>
</TR>
<TR>
  <TD>-g, --group &lt;arg&gt;</TD>
  <TD>Specify which event string or performance group should be measured.</TD>
</TR>
<TR>
  <TD>-c &lt;arg&gt;</TD>
  <TD>Defines the CPUs that should be measured<BR>See \ref CPU_expressions on the \ref likwid-pin page for information about the syntax.</TD>
</TR>
<TR>
  <TD>-C &lt;arg&gt;</TD>
  <TD>Defines the CPUs that should be measured and pin the executable to the CPUs<BR>See \ref CPU_expressions on the \ref likwid-pin page for information about the syntax.</TD>
</TR>
<TR>
  <TD>-H</TD>
  <TD>Print information about a performance group given with -g, --group option.</TD>
</TR>
<TR>
  <TD>-m</TD>
  <TD>Run in marker API mode</TD>
</TR>
<TR>
  <TD>-a</TD>
  <TD>Print available performance groups for current processor.</TD>
</TR>
<TR>
  <TD>-e</TD>
  <TD>Print available counters and performance events and suitable options of current processor.</TD>
</TR>
<TR>
  <TD>-E &lt;pattern&gt;</TD>
  <TD>Print available performance events matching &lt;pattern&gt; and print the usable counters for the found events.<BR>The matching is done with *&lt;pattern&gt;*, so all events matching the substring are returned.</TD>
</TR>
<TR>
  <TD>-o, --output &lt;file&gt;</TD>
  <TD>Store all ouput to file instead of stdout. LIKWID enables the reformatting of output files according to their suffix.<BR>You can place additional output formatters in folder <CODE>&lt;PREFIX&gt;/share/likwid</CODE>. LIKWID ships with the two filter scripts <CODE>xml</CODE> and <CODE>csv</CODE>.<BR>Moreover, there are substitutions possible in the output filename. <CODE>\%h</CODE> is replaced by the host name, <CODE>\%p</CODE> by the PID, <CODE>\%j</CODE> by the job ID of batch systems and <CODE>\%r</CODE> by the MPI rank.</TD>
</TR>
<TR>
  <TD>-S &lt;time&gt;</TD>
  <TD>Specify the time between starting and stopping of counters. Can be used to monitor applications. Option does not require an executable<BR>Examples for &lt;time&gt; are 1s, 250ms, 500us.</TD>
</TR>
<TR>
  <TD>-t &lt;time&gt;</TD>
  <TD>Activates the timeline mode that reads the counters in the given frequency &lt;time&gt; during the whole run of the executable<BR>Examples for &lt;time&gt; are 1s, 250ms, 500us.</TD>
</TR>
<TR>
  <TD>-O</TD>
  <TD>Print output in CSV format (conform to <A HREF="https://tools.ietf.org/html/rfc4180">RFC 4180</A>).</TD>
</TR>
<TR>
  <TD>-s, --skip &lt;arg&gt;</TD>
  <TD>'arg' must be a bitmask in hex. Threads with the ID equal to a set bit in bitmask will be skipped during pinning<BR>Example: 0x1 = Thread 0 is skipped.</TD>
</TR>
</TABLE>

<H1>Examples</H1>
<UL>
<LI><CODE>likwid-perfctr -C 0-2 -g TLB ./a.out</CODE><BR>
Pin the executable <CODE>./a.out</CODE> to CPUs 0,1,2 and measure on the specified CPUs the performance group <CODE>TLB</CODE>. If not set, the environment variable <CODE>OMP_NUM_THREADS</CODE> is set to 3.
</LI>
<LI><CODE>likwid-perfctr  -C 0-4  -g INSTRUCTIONS_RETIRED_SSE:PMC0,CPU_CLOCKS_UNHALTED:PMC3 ./a.out</CODE><BR>
Pin the executable <CODE>./a.out</CODE> to CPUs 0,1,2,3,4 and measure on the specified CPUs the event set <CODE>INSTRUCTIONS_RETIRED_SSE:PMC0,CPU_CLOCKS_UNHALTED:PMC3</CODE>.<BR>The event set consists of two event definitions:
    <UL>
    <LI><CODE>INSTRUCTIONS_RETIRED_SSE:PMC0</CODE> measures event <CODE>INSTRUCTIONS_RETIRED_SSE</CODE> using counter register named <CODE>PMC0</CODE></LI>
    <LI><CODE>CPU_CLOCKS_UNHALTED:PMC3</CODE> measures event <CODE>CPU_CLOCKS_UNHALTED</CODE> using counter register named <CODE>PMC3</CODE>. This event can be used to calculate the run time of the application.</LI>
    </UL>
</LI>

<LI><CODE>likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,UNC_L3_LINES_IN_ANY:UPMC0 ./a.out</CODE><BR>
Run and pin executable <CODE>./a.out</CODE> on CPU 0 with a custom event set containing three events.<BR>The event set consists of three event definitions:
    <UL>
    <LI><CODE>INSTR_RETIRED_ANY:FIXC0</CODE> measures event <CODE>INSTR_RETIRED_ANY</CODE> using Intel's fixed-purpose counter register named <CODE>FIXC0</CODE>.</LI>
    <LI><CODE>CPU_CLK_UNHALTED_CORE:FIXC1</CODE> measures event <CODE>CPU_CLOCKS_UNHALTED</CODE> using Intel's fixed-purpose counter register named <CODE>FIXC1</CODE>. This event can be used to calculate the run time of the application.</LI>
    <LI><CODE>UNC_L3_LINES_IN_ANY:UPMC0</CODE> measures event <CODE>UNC_L3_LINES_IN_ANY</CODE> using Uncore counter register named <CODE>UPMC0</CODE>. Uncore counters are socket-specific, hence LIKWID reads the counter registers only on one CPU per socket.</LI>
    </UL>
</LI>

<LI><CODE>likwid-perfctr -m -C 0-4  -g INSTRUCTIONS_RETIRED_SSE:PMC0,CPU_CLOCKS_UNHALTED:PMC3 ./a.out</CODE><BR>
Run and pin the executable to CPUs 0,1,2,3,4 and activate the Marker API. The code in <CODE>a.out</CODE> is assumed to be instrumented with LIKWID's Marker API. Only the marked code regions are measured.
    <UL>
    <LI><CODE>INSTRUCTIONS_RETIRED_SSE:PMC0</CODE> measures event <CODE>INSTRUCTIONS_RETIRED_SSE</CODE> using counter register named <CODE>PMC0</CODE>.</LI>
    <LI><CODE>CPU_CLOCKS_UNHALTED:PMC3</CODE> measures event <CODE>CPU_CLOCKS_UNHALTED</CODE> using counter register named <CODE>PMC3</CODE>. This event can be used to calculate the run time of the application.</LI>
    </UL>
The Marker API for C/C++ offers 6 functions to measure named regions. You can use instrumented code with and without LIKWID. In order to activate the Marker API, <CODE>-DLIKWID_PERFMON</CODE> needs to be added to the compiler call. The following listing describes each function shortly (complete list see \ref Marker_API):
    <UL>
    <LI><CODE>LIKWID_MARKER_INIT</CODE>: Initialize LIKWID globally. Must be called in serial region and only once.</LI>
    <LI><CODE>LIKWID_MARKER_THREADINIT</CODE>: Initialize LIKWID for each thread. Must be called in parallel region and executed by every thread.</LI>
    <LI><CODE>LIKWID_MARKER_START('compute')</CODE>: Start a code region and associate it with the name 'compute'. The names are freely selectable and are used for grouping and outputting regions.</LI>
    <LI><CODE>LIKWID_MARKER_STOP('compute')</CODE>: Stop the code region associated with the name 'compute'.</LI>
    <LI><CODE>LIKWID_MARKER_SWITCH</CODE>: Switches to the next performance group or event set in a round-robin fashion. Can be used to measure the same region with multiple events. If called inside a code region, the results for all groups will be faulty. Be aware that each programming of the config registers causes overhead.</LI>
    <LI><CODE>LIKWID_MARKER_CLOSE</CODE>: Finalize LIKWID globally. Should be called in the end of your application. This writes out all region results to a file that is picked up by <CODE>likwid-perfctr</CODE> for evaluation.</LI>
    </UL>
</LI>

<LI><CODE>likwid-perfctr -c 0-3  -g FLOPS_DP -t 300ms ./a.out 2> out.txt</CODE><BR>
Runs the executable <CODE>a.out</CODE> and measures the performance group <CODE>FLOPS_DP</CODE> on CPUs 0,1,2,3 every 300 ms. Since <CODE>-c</CODE> is used, the application is not pinned to the CPUs and <CODE>OMP_NUM_THREADS</CODE> is not set. The performance group <CODE>FLOPS_DP</CODE> is not available on every architecture, use <CODE>likwid-perfctr -a</CODE> for a complete list. Please note, that <CODE>likwid-perfctr</CODE> writes the measurements to stderr while the application's output and LIKWID's final results are printed to stdout.<BR>
The syntax of the timeline mode output lines is:<BR>
<CODE>&lt;groupID&gt; &lt;numberOfEvents&gt; &lt;numberOfThreads&gt; &lt;Timestamp&gt; &lt;Event1_Thread1&gt; &lt;Event1_Thread2&gt; ... &lt;EventN_ThreadN&gt;</CODE><BR>
You can also use the tool \ref likwid-perfscope to print the measured values live with <CODE>gnuplot</CODE>.
</LI>

<LI><CODE>likwid-perfctr -c 0-3  -g FLOPS_DP -S 2s</CODE><BR>
Measures the performance group <CODE>FLOPS_DP</CODE> on CPUs 0,1,2,3 for 2 seconds. This option can be used to measure application from external or to perform low-level system monitoring.
</LI>

<LI><CODE>likwid-perfctr -c S0:1\@S1:1  -g LLC_LOOKUPS_DATA_READ:CBOX0C0:STATE=0x9 -S 2s</CODE><BR>
Measures the event <CODE> LLC_LOOKUPS_DATA_READ</CODE> on the first CPU of socket 0 and the first CPU on socket 1 for 2 seconds using the counter 0 in CBOX 0 (LLC cache coherency engine). The counting is filtered to only lookups in the 'invalid' and 'modified' state. Look at the microarchitecture Uncore documentation for possible bitmasks. Which option is available for which counter class can be found in section \ref Architectures.
</LI>
</UL>

\anchor performance_groups
<H1>Performance groups</H1>
One of the outstanding features of LIKWID are the performance groups. Each microarchitecture has its own set of events and related counters and finding the suitable events in the documentation is tedious. Moreover, the raw results of the events are often not meaningful, they need to be combined with other events like run time or clock speed. LIKWID addresses those problems by providing performance groups that specify a set of events and counter combinations as well as a set of derived metrics. Starting with LIKWID 4, the performance group definitions are not compiled in anymore, they are read on the fly when they are selected on the commandline. This enables users to define their own performance groups without recompiling and reinstalling LIKWID.<BR>
<B>Please note that performance groups is a feature of the Lua API and not available for the C/C++ API.</B>
<H3>Directory structure</H3>
While installation of LIKWID, the performance groups are copied to the path <CODE>${INSTALL_PREFIX}/share/likwid</CODE>. In this folder there is one subfolder per microarchitecture that contains all performance groups for that microarchitecture. The folder names are not freely selectable, they are defined in <CODE>src/topology.c</CODE>. For every microarchitecture at the time of release, there is already a folder that can be extended with your own performance groups. You can change the path to the performance group directory structure by settings the variable <CODE>likwid.groupfolder</CODE> in your Lua application, the default is <CODE>${INSTALL_PREFIX}/share/likwid</CODE>.
<H3>Syntax of performance group files</H3>
<CODE>SHORT &lt;string&gt;</CODE> // Short description of the performance group<BR>
<BR>
<CODE>EVENTSET</CODE> // Starts the event set definition<BR>
<CODE>&lt;counter&gt;(:&lt;options&gt;) &lt;event&gt;</CODE> // Each line defines one event/counter combination with optional options.<BR>
<CODE>FIXC0 INSTR_RETIRED_ANY</CODE> // Example<BR>
<BR>
<CODE>METRICS</CODE> // Starts the derived metric definitions<BR>
<CODE>&lt;metricname&gt; &lt;formula&gt;</CODE> // Each line defines one derived metric. <CODE>&lt;metricname&gt;</CODE> can contain spaces, <CODE>&lt;formula&gt;</CODE> must be free of spaces. The counter names (with options) and the variables <CODE>time</CODE> and <CODE>inverseClock</CODE> can be used as variables in <CODE>&lt;formula&gt;</CODE>.
<CODE>CPI  FIXC1/FIXC0</CODE> // Example<BR>
<BR>
<CODE>LONG</CODE> // Starts the detailed description of the performance group<BR>
<CODE>&lt;TEXT&gt;</CODE> // <CODE>&lt;TEXT&gt;</CODE> is displayed with <CODE>-H</CODE> commandline option

\anchor Marker_API
<H1>Marker API</H1>
The Marker API enables measurement of user-defined code regions in order to get deeper insight what is happening at a specific point in the application. The Marker API itself has 6 commands. In order to activate the Marker API, the code must be compiled with <CODE>-DLIKWID_PERFMON</CODE>. If the code is compiled without this define, the Marker API functions perform no operation and cause no overhead. You can also run code compiled with the define without measurements but a message will be printed.<BR>
<H2>C/C++ Code</H2>
<H3>Original code</H3>
<CODE>
\#include &lt;stdlib.h&gt;<BR>
\#include &lt;stdio.h&gt;<BR>
\#include &lt;omp.h&gt;<BR>
<BR>
int main(int argc, char* argv[])<BR>
{<BR>
&nbsp;&nbsp;int i=0;<BR>
&nbsp;&nbsp;double sum = 0;<BR>
\#pragma omp parallel for reduction(+:sum)<BR>
&nbsp;&nbsp;for(i=0;i&lt;100000;i++)<BR>
&nbsp;&nbsp;{<BR>
&nbsp;&nbsp;&nbsp;&nbsp;sum += 1.0/(omp_get_thread_num()+1);<BR>
&nbsp;&nbsp;}<BR>
&nbsp;&nbsp;printf("Sum is %f\n", sum);<BR>
&nbsp;&nbsp;return 0;<BR>
}<BR>
</CODE>
<H3>Instrumented code</H3>
<CODE>
\#include &lt;stdlib.h&gt;<BR>
\#include &lt;stdio.h&gt;<BR>
\#include &lt;omp.h&gt;<BR>
\#include &lt;likwid.h&gt;<BR>
<BR>
int main(int argc, char* argv[])<BR>
{<BR>
&nbsp;&nbsp;int i=0;<BR>
&nbsp;&nbsp;double sum = 0;<BR>
&nbsp;&nbsp;LIKWID_MARKER_INIT;<BR>
\#pragma omp parallel<BR>
{<BR>
&nbsp;&nbsp;LIKWID_MARKER_THREADINIT;<BR>
}<BR>
\#pragma omp parallel<BR>
{<BR>
&nbsp;&nbsp;LIKWID_MARKER_START("sum");<BR>
\#pragma omp for reduction(+:sum)<BR>
&nbsp;&nbsp;for(i=0;i&lt;100000;i++)<BR>
&nbsp;&nbsp;{<BR>
&nbsp;&nbsp;&nbsp;&nbsp;sum += 1.0/(omp_get_thread_num()+1);<BR>
&nbsp;&nbsp;}<BR>
&nbsp;&nbsp;LIKWID_MARKER_STOP("sum");<BR>
}<BR>
&nbsp;&nbsp;printf("Sum is %f\n", sum);<BR>
&nbsp;&nbsp;LIKWID_MARKER_CLOSE;<BR>
&nbsp;&nbsp;return 0;<BR>
}<BR>
</CODE>
<H3>Running code</H3>
With the help of <CODE>likwid-perfctr</CODE> the counters are configured to the selected events. The counters are also started and stopped by <CODE>likwid-perfctr</CODE>, the Marker API only reads the counters to minimize the overhead of the instrumented application. Only if you use <CODE>LIKWID_MARKER_SWITCH</CODE> the Marker API itself configures a new event set to the registers. Basically, <CODE>likwid-perfctr</CODE> exports the whole configuration needed by the Marker API through environment variables that are evaluated during <CODE>LIKWID_MARKER_INIT</CODE>. In the end, <CODE>likwid-perfctr</CODE> picks up the file with the results of the Marker API run and prints out the performance results.<BR>
In order to build your instrumented application:<BR>
<CODE>$CC -openmp -L&lt;PATH_TO_LIKWID_LIBRARY&gt; -I&lt;PATH_TO_LIKWID_INCLUDES&gt; &lt;SRC_CODE&gt; -o &lt;EXECUTABLE&gt; -llikwid</CODE><BR>
With standard installation, the paths are <CODE>&lt;PATH_TO_LIKWID_LIBRARY&gt;=/usr/local/lib</CODE> and <CODE>&lt;PATH_TO_LIKWID_INCLUDES&gt;=/usr/local/include</CODE><BR>
Example Marker API call:<BR>
<CODE>likwid-perfctr -C 0-4 -g L3 <B>-m</B> ./a.out</CODE>
<BR>
<BR>

<H2>Fortran Code</H2>
Besides the Marker API for C/C++ programms, LIKWID offers to build a Fortran module to access the Marker API functions from Fortran. Only the Marker API calls are exported, not the whole API. In <CODE>config.mk</CODE> the variable <CODE>FORTRAN_INTERFACE</CODE> must be set to true. LIKWID's default is to use the Intel Fortran compiler to build the interface but it can be modified to use GCC's Fortran compiler in <CODE>make/include_&lt;COMPILER&gt;</CODE>.


*/
