/*! \page likwid-memsweeper <CODE>likwid-memsweeper</CODE>

<H1>Information</H1>
<CODE>likwid-memsweeper</CODE> is a command line application to shrink the file buffer cache by filling the NUMA domain with random pages. Moreover, the tool invalidates all cachelines in the LLC.


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
  <TD>-c &lt;list&gt;</TD>
  <TD>Sweeps the memory and LLC cache for NUMA domains listed in &lt;list&gt;.</TD>
</TR>
</TABLE>

<H1>Examples</H1>
<UL>
<LI><CODE>likwid-memsweeper -c 0,1</CODE><BR>
Cleans the memory and LLC on NUMA nodes identified by the node IDs 0 and 1.
</LI>
</UL>

*/
