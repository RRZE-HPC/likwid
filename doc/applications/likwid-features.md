/*! \page likwid-features <CODE>likwid-features</CODE>

<H1>Information</H1>
<CODE>likwid-features</CODE> is a command line application to print and change some features of the CPU. Most of the features cannot be changed during runtime, but the prefetchers <CODE>HW_PREFETCHER</CODE>, <CODE>CL_PREFETCHER</CODE>, <CODE>DCU_PREFETCHER</CODE>, <CODE>IP_PREFETCHER</CODE> can be altered.<BR>

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
  <TD>-a, --all</TD>
  <TD>List all features</TD>
</TR>
<TR>
  <TD>-c, --cpus &lt;arg&gt;</TD>
  <TD>Define the CPUs that should be modified. For information about the syntax see \ref CPU_expressions on the \ref likwid-pin page</TD>
</TR>
<TR>
  <TD>-l, --list</TD>
  <TD>List all features with current state for the CPUs defined at -c/--cpus</TD>
</TR>
<TR>
  <TD>-e, --enable &lt;list&gt;</TD>
  <TD>Comma-separated list of features that should be enabled</TD>
</TR>
<TR>
  <TD>-d, --disable &lt;list&gt;</TD>
  <TD>Comma-separated list of features that should be disabled</TD>
</TR>
</TABLE>

<B><CODE>likwid-features</CODE> will be deprecated in 5.4</B>

*/
