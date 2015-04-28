/*! \page likwid-bench <CODE>likwid-bench</CODE>

<H1>Information</H1>
<CODE>likwid-bench</CODE> is a benchmark suite for low-level (assembly) benchmarks to measure bandwidths and instruction throughput for specific instruction code on x86 systems. The currently included benchmark codes include common data access patterns like load and store but also calculations like vector triad and sum.
<CODE>likwid-bench</CODE> includes architecture specific benchmarks for x86, x86_64 and x86 for Intel Xeon Phi coprocessors. The performance values can either be calculated by <CODE>likwid-bench</CODE> or measured using hardware performance counters by using \ref likwid-perfctr as a wrapper to <CODE>likwid-bench</CODE>. This requires to build <CODE>likwid-bench</CODE> with instrumentation enabled in config.mk (<CODE>INSTRUMENT_BENCH</CODE>).


<H1>Options</H1>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>-h</TD>
  <TD>Print help message</TD>
</TR>
<TR>
  <TD>-a</TD>
  <TD>List all available benchmarks</TD>
</TR>
<TR>
  <TD>-p</TD>
  <TD>List all available thread affinity domains</TD>
</TR>
<TR>
  <TD>-d &lt;delim&gt;</TD>
  <TD>Use &lt;delim&gt; instead of ',' for the output of -p</TD>
</TR>
<TR>
  <TD>-l &lt;test&gt;</TD>
  <TD>List characteristics of &lt;test&gt; like number of streams, data used per loop iteration, ...</TD>
</TR>
<TR>
  <TD>-t &lt;test&gt;</TD>
  <TD>Perform assembly benchmark &lt;test&gt;</TD>
</TR>
<TR>
  <TD>-s &lt;min_time&gt;</TD>
  <TD>Minimal time in seconds to run the benchmark.<BR>Using this time, the iteration count is determined automatically to provide reliable results. Default is 1. If the determined iteration count is below 10, it is normalized to 10.</TD>
</TR>
<TR>
  <TD>-w &lt;workgroup&gt;</TD>
  <TD>Set a workgroup for the benchmark. A workgroup can have different formats:<BR>
  <TABLE>
    <TR>
      <TH>Format</TH>
      <TH>Description</TH>
    </TR>
    <TR>
      <TD>&lt;affinity_domain&gt;:&lt;size&gt;</TD>
      <TD>Allocate in total &lt;size&gt; in affinity domain &lt;affinity_domain&gt;.<BR><CODE>likwid-bench</CODE> starts as many threads as available in affinity domain &lt;affinity_domain&gt;</TD>
    </TR>
    <TR>
      <TD>&lt;affinity_domain&gt;:&lt;size&gt;:&lt;num_threads&gt;</TD>
      <TD>Allocate in total &lt;size&gt; in affinity domain &lt;affinity_domain&gt;.<BR><CODE>likwid-bench</CODE> starts &lt;num_threads&gt; in affinity domain &lt;affinity_domain&gt;</TD>
    </TR>
    <TR>
      <TD>&lt;affinity_domain&gt;:&lt;size&gt;:&lt;num_threads&gt;:&lt;chunk_size&gt;:&lt;stride&gt;</TD>
      <TD>Allocate in total &lt;size&gt; in affinity domain &lt;affinity_domain&gt;.<BR><CODE>likwid-bench</CODE> starts &lt;num_threads&gt; in affinity domain &lt;affinity_domain&gt; with &lt;chunk_size&gt; selected in row and a distance of &lt;stride&gt;.<BR>See \ref CPU_expressions on the \ref likwid-pin page for further information.</TD>
    </TR>
    <TR>
      <TD>&lt;above_formats&gt;-&lt;streamID&gt;:&lt;stream_domain&gt;</TD>
      <TD>In combination with every above mentioned format, the test streams (arrays, vectors) can be place in different affinity domains than the threads.<BR>This can be achieved by adding a stream placement option -&lt;streamID&gt;:&lt;stream_domain&gt; for all streams of the test to the workgroup definition.<BR>The stream with &lt;streamID&gt; is placed in affinity domain &lt;stream_domain&gt;.<BR>The amount of streams of a test can be determined with the -l &lt;test&gt; commandline option.</TD>
    </TR>
  </TD>
  </TABLE>
</TR>
</TABLE>


<H1>Examples</H1>
<UL>
<LI><CODE>likwid-bench -t copy -w S0:100kB</CODE><BR>
Run test <CODE>copy</CODE> using all threads in affinity domain <CODE>S0</CODE>. The input and output stream of the <CODE>copy</CODE> benchmark sum up to <CODE>100kB</CODE> placed in affinity domain <CODE>S0</CODE>. The iteration count is calculated automatically.
</LI>
<LI><CODE>likwid-bench -t triad -i 100 -w S0:1GB:2:1:2</CODE><BR>
Run test <CODE>triad</CODE> using <CODE>2</CODE> threads in affinity domain <CODE>S0</CODE>. Assuming <CODE>S0 = 0,4,1,5</CODE> the threads are pinned to CPUs 0 and 1, hence skipping of one thread during selection. The streams of the <CODE>triad</CODE> benchmark sum up to <CODE>1GB</CODE> placed in affinity domain <CODE>S0</CODE>. The number of iteration is explicitly set to <CODE>100</CODE>
</LI>
<LI><CODE>likwid-bench -t update -w S0:100kB -w S1:100kB</CODE><BR>
Run test <CODE>update</CODE> using all threads in affinity domain <CODE>S0</CODE> and <CODE>S1</CODE>. The threads scheduled on <CODE>S0</CODE> use stream that sum up to <CODE>100kB</CODE>. Similar to <CODE>S1</CODE> the threads are placed there working only on their socket-local streams. The results of both workgroups are combined.
</LI>
<LI><CODE>likwid-perfctr -c E:S0:4 -g MEM -m likwid-bench -t update -w S0:100kB:4</CODE><BR>
Run test <CODE>update</CODE> using <CODE>4</CODE> threads in affinity domain <CODE>S0</CODE>. The input and output stream of the <CODE>copy</CODE> benchmark sum up to <CODE>100kB</CODE> placed in affinity domain <CODE>S0</CODE>. The benchmark execution is measured using the \ref Marker_API. It measures the <CODE>MEM</CODE> performance group on the first four CPUs of the <CODE>S0</CODE> affinity domain. For further information about hardware performance counters see \ref likwid-perfctr<BR><B>Note:</B> Currently it is not possible to pin the threads already at <CODE>likwid-perfctr</CODE>. The pinning is done by <CODE>likwid-bench</CODE>
</LI>
<LI><CODE>likwid-bench -t copy -w S0:1GB:2:1:2-0:S1,1:S1</CODE><BR>
Run test <CODE>copy</CODE> using <CODE>2</CODE> threads in affinity domain <CODE>S0</CODE> skipping one thread during selection. The two streams used in the <CODE>copy</CODE> benchmark have the IDs 0 and 1 and a summed up size of <CODE>1GB</CODE>. Both streams are placed in affinity domain <CODE>S1</CODE>.
</LI>
</UL>



*/
