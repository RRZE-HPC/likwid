/*! \page phi Intel&reg; Xeon Phi

<P>To use LIKWID you have to turn of power management on the MIC. LIKWID relies on
RDTSC being used for wallclock time. On the MIC this is only given if power
management is turned off. This can be configured in
<CODE>/etc/sysconfig/mic/default.conf</CODE>.<BR>

At the end of this file the power management is configured. The following
configuration worked:<BR>
<CODE>PowerManagement "cpufreq_off;corec6_off;pc3_off;pc6_off"</CODE>
</P>

<H1>Available performance monitors for the Intel&reg; Xeon Phi microarchitecture</H1>
<UL>
<LI>\ref PHI_PMC "General-purpose counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor PHI_PMC
<H2>General-purpose counters</H2>
<P>The Intel&reg; Xeon Phi microarchitecture provides 2 general-purpose counters consisting of a config and a counter register.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PMC0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC1</TD>
  <TD>*</TD>
</TR>
</TABLE>
<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>kernel</TD>
  <TD>N</TD>
  <TD>Set bit 17 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>anythread</TD>
  <TD>N</TD>
  <TD>Set bit 21 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
</TABLE>


*/
