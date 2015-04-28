/*! \page likwid.cfg <CODE>likwid.cfg</CODE>
<H1>Information</H1>
<CODE>likwid.cfg</CODE> is the global configuration file for LIKWID but it is optional. The configuration is normally defined at compile time. It allows to set the path to the access mode for the MSR/PCI access daemon and some other basic options.<BR>
LIKWID searches for the configuration file at different paths like <CODE>/usr/local/etc/likwid.cfg</CODE>.<BR>
<B>Note: It was introduced with version 4 and is not fully integrated in the LIKWID code.</B>

<H1>Config file options</H1>
<H1>Config file</H1>
The global configuration file has the following options:
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>topology_file = &lt;path&gt;</TD>
  <TD>Path to the toplogy file created with \ref likwid-genTopoCfg</TD>
</TR>
<TR>
  <TD>access_mode = &lt;daemon|direct&gt;</TD>
  <TD>Set access mode. The direct mode can only used by users with root priviledges. The daemon uses \ref likwid-accessD.</TD>
</TR>
<TR>
  <TD>daemon_path = &lt;path&gt;</TD>
  <TD>Path to the access daemon.</TD>
</TR>
<TR>
  <TD>max_threads = &lt;arg&gt;</TD>
  <TD>Adjust maximally supported threads/CPUs. <B>Note:</B> not use by now, fixed at compile time.</TD>
</TR>
<TR>
  <TD>max_nodes = &lt;arg&gt;</TD>
  <TD>Adjust maximally supported NUMA nodes. <B>Note:</B> not use by now, fixed at compile time.</TD>
</TR>
</TABLE>


*/
