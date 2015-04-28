/*! \page likwid-powermeter <CODE>likwid-powermeter</CODE>

<H1>Information</H1>
likwid-powermeter is a command line application to get the energy comsumption on Intel RAPL capable processors. Currently
all Intel CPUs starting with Intel SandyBridge are supported. It also prints information about TDP and Turbo Mode steps supported.
The Turbo Mode information works on all Turbo mode enabled Intel processors. The tool can be either used in stethoscope mode for a specified duration or as a wrapper to your application measuring your complete run. RAPL works on a per package (socket) base.
Please note that the RAPL counters are also accessible as normal events withing \ref likwid-perfctr.

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
  <TD>Specify sockets to measure</TD>
</TR>
<TR>
  <TD>-M &lt;0|1&gt;</TD>
  <TD>Set access mode to access MSRs. 0=direct, 1=accessDaemon</TD>
</TR>
<TR>
  <TD>-s &lt;time&gt;</TD>
  <TD>Set measure duration in us, ms or s. (default 2s)</TD>
</TR>
<TR>
  <TD>-i, --info</TD>
  <TD>Print information from <CODE>MSR_*_POWER_INFO</CODE> register and Turbo mode</TD>
</TR>
<TR>
  <TD>-t</TD>
  <TD>Print current temperatures of all CPU cores</TD>
</TR>
<TR>
  <TD>-f</TD>
  <TD>Print current temperatures of all CPU cores in Fahrenheit</TD>
</TR>
<TR>
  <TD>-p</TD>
  <TD>Print dynamic clocking and CPI values, uses \ref likwid-perfctr</TD>
</TR>
</TABLE>
*/
