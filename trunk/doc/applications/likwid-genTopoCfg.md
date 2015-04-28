/*! \page likwid-genTopoCfg <CODE>likwid-genTopoCfg</CODE>

<H1>Information</H1>
<CODE>likwid-genTopoCfg</CODE> is a command line application that stores the system's CPU and NUMA topology to
file. LIKWID applications use this file to read in the topology fast instead of re-gathering all values. The path to the topology configuration can be set in the global LIKWID configuration file, see \ref likwid.cfg.

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
  <TD>-o &lt;file&gt;</TD>
  <TD>Use &lt;file&gt; instead of the default output /etc/likwid-topo.cfg./TD>
</TR>
</TABLE>


*/

