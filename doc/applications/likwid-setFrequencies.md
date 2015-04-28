/*! \page likwid-setFrequencies <CODE>likwid-setFrequencies</CODE>

<H1>Information</H1>
<CODE>likwid-setFrequencies</CODE> is a command line application to set the clock frequency of CPU cores. Since only priviledged users are allowed to change the frequency of CPU cores, the application works in combination with a daemon
\ref likwid-setFreq . The daemon needs the suid permission bit to be set in order to manipulate the sysfs entries. With <CODE>likwid-setFrequencies</CODE> the clock of all cores inside a the cpu_list or affinity domain can be set to a specific frequency or governor at once.

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
  <TD>-l</TD>
  <TD>Print all configurable frequencies.</TD>
</TR>
<TR>
  <TD>-p</TD>
  <TD>Print the current frequencies for all CPU cores.</TD>
</TR>
<TR>
  <TD>-m</TD>
  <TD>Print all configurable governors./TD>
</TR>
<TR>
  <TD>-c &lt;arg&gt;</TD>
  <TD>Define the CPUs that should be modified. For information about the syntax see \ref CPU_expressions on the \ref likwid-pin page.</TD>
</TR>
<TR>
  <TD>-f, --freq &lt;arg&gt;</TD>
  <TD>Specify the frequency for the selected CPUs.</TD>
</TR>
<TR>
  <TD>-g &lt;arg&gt;</TD>
  <TD>Specify the governor for the selected CPUs.</TD>
</TR>
</TABLE>

<H1>Notice</H1>
Shortly before releasing the first version of LIKWID 4, the CPU frequency module and its behavior have changed compared to the previous <B>cpufreq</B> module. It is not possible anymore to set the CPU clock to a fixed frequency, you can only define a performance level called P-State. Inside that level, the CPU can vary its clock frequency. <CODE>likwid-setFrequencies</CODE> and its daemon \ref likwid-setFreq do not have support for the new kernel module <B>intel_pstate</B>. Therefore, the loaded driver is checked in the beginning.

*/
