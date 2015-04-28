/*! \page likwid-mpirun <CODE>likwid-mpirun</CODE>

<H1>Information</H1>
<CODE>likwid-mpirun</CODE>
A tool to start and monitor MPI applications with LIKWID. It can be used as supplement of the MPI implementations' startup programm like <CODE>mpirun</CODE> or <CODE>mpiexec</CODE> with some enhancements for pinning of OpenMP thread in hybrid jobs. Moreover, <CODE>likwid-mpirun</CODE> can insert calls to \ref likwid-perfctr to measure hardware performance counters for each MPI process and its threads, including Marker API. Since the <A HREF="http://modules.sourceforge.net/">modules</A> system is widely used on clustered systems, <CODE>likwid-mpirun</CODE> checks the modules system to get MPI and OpenMP types. The hostfile must be in the format of the loaded MPI implementation.

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
  <TD>-d, --debug</TD>
  <TD>Print debug information</TD>
</TR>
<TR>
  <TD>-n, -np, --n, --np &lt;arg&gt;</TD>
  <TD>Specify the number of processes for MPI</TD>
</TR>
<TR>
  <TD>--nperdomain &lt;domain&gt;:&lt;arg&gt;</TD>
  <TD>Schedule &lt;arg&gt; MPI processes for each affinity domain starting with &lt;domain&gt;, e.g S:2 translates in two MPI processes per socket.<BR><CODE>likwid-mpirun</CODE> assumes that all participating hosts have the same topology.</TD>
</TR>
<TR>
  <TD>--hostfile &lt;file&gt;</TD>
  <TD>Specify the file that should be used as hostfile.<BR>If not set, <CODE>likwid-mpirun</CODE> checks the <CODE>PBS_NODEFILE</CODE> environment variable</TD>
</TR>
<TR>
  <TD>--pin &lt;expr&gt;</TD>
  <TD>For hybrid pinning specify the thread pinning expression for each MPI process.<BR>The format is similar to \ref CPU_expressions separated by '_' for multiple processes.<BR>If -np is not set, the number of MPI processes is calculated using the pinning expressions.</TD>
</TR>
<TR>
  <TD>-s, --skip &lt;arg&gt;</TD>
  <TD>'arg' must be a bitmask in hex. Threads with the ID equal to a set bit in bitmask will be skipped during pinning<BR>Example: 0x1 = Thread 0 is skipped.</TD>
</TR>
<TR>
  <TD>--mpi &lt;mpitype&gt;</TD>
  <TD>Specify the type of the MPI implementation.<BR><CODE>likwid-mpirun</CODE> tries to read the MPI implementation from the <A HREF="http://modules.sourceforge.net/">modules</A> system.<BR>If not recognized automatically, possible values are <B>intelmpi</B>, <B>openmpi</B> and <B>mvapich2</B>.</TD>
</TR>
<TR>
  <TD>--omp &lt;omptype&gt;</TD>
  <TD>Specify the type of OpenMP implementation.<BR><CODE>likwid-mpirun</CODE> tries to read the OpenMP implementation using <I>ldd</I> and the <A HREF="http://modules.sourceforge.net/">modules</A> system.<BR>If not recognized automatically, possible values are <B>intel</B> and <B>gnu</B></TD>
</TR>
<TR>
  <TD>-g, --group &lt;eventset&gt;</TD>
  <TD>Use \ref likwid-perfctr to measure performance data for the MPI processes and OpenMP threads.<BR>&lt;eventset&gt; can be either a performance group or a custom event string.<BR>For details see \ref performance_groups.</TD>
</TR>
<TR>
  <TD>-m, --marker</TD>
  <TD>Activate the \ref Marker_API for the measurements with \ref likwid-perfctr.</TD>
</TR>
<TR>
  <TD>-O</TD>
  <TD>Print results in CSV format (conform to <A HREF="https://tools.ietf.org/html/rfc4180">RFC 4180</A>)</TD>
</TR>
</TABLE>

<H1>Examples</H1>
<UL>
<LI><CODE>likwid-mpirun -np 32 ./a.out</CODE><BR>
Runs <CODE>./a.out</CODE> with 32 MPI processes distributed over the hosts in <CODE>PBS_NODEFILE</CODE>
</LI>
<LI><CODE>likwid-mpirun -nperdomain S:1 ./a.out</CODE><BR>
Runs <CODE>./a.out</CODE> using one MPI process per socket over the hosts in <CODE>PBS_NODEFILE</CODE>.<BR>The total amount of processes is calculated by &lt;numberOfSocketDomains&gt; * &lt;processCountPerDomain&gt; * &lt;hostsInHostfile&gt;
</LI>
<LI><CODE>likwid-mpirun --hostfile host.list -pin S0:2_S1:2 ./a.out</CODE><BR>
Runs <CODE>./a.out</CODE> using two MPI processes per host in <CODE>host.list</CODE>.<BR>The first MPI process on each host and its 2 threads are pinned to the first two CPUs on socket <CODE>S0</CODE>,<BR>the second MPI process on each host and its 2 threads are pinned to the first two CPUs on socket <CODE>S1</CODE>
</LI>
<LI><CODE>likwid-mpirun -nperdomain S:2 -g MEM ./a.out</CODE><BR>
Runs <CODE>./a.out</CODE> with 2 MPI processes per socket on each host in <CODE>PBS_NODEFILE</CODE> and measure the <CODE>MEM</CODE> performance group<BR>
Only one process per socket measures the Uncore/RAPL counters, the other one(s) only core-local counters.
</LI>
</UL>
*/
