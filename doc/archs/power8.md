\page power8 IBM&reg; POWER8

<H1>Available performance monitors for the IBM&reg; POWER8 microarchitecture</H1>
<UL>
<LI>\ref PWR8_PMC "General-purpose counters"</LI>
</UL>



\anchor PWR8_PMC
<H2>General-purpose counters</H2>
<P>The IBM&reg; POWER8 microarchitecture provides 4 general-purpose counters consisting of a config and a counter register. Additionally, there are two registers marked as general-purpose but they are fixed to specific events (like the fixed-purpose counters of Intel&reg; systems). Be aware that IBM names the counters PMC1-PMC6 while LIKWID uses PMC0-PMC5. Events like PM_PMCx_OVERFLOW are adapted accordingly.</P>
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
<TR>
  <TD>PMC2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC4</TD>
  <TD>PM_RUN_INST_CMPL</TD>
</TR>
<TR>
  <TD>PMC5</TD>
  <TD>PM_RUN_CYC</TD>
</TR>
</TABLE>
<H3>Available Options</H3>
<P>No options<P>

