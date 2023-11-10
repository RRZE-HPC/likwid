/*! \page applem1 Apple&reg; M1

<P>This page is valid for Apple&reg; M1.</P>

<H1>Available performance monitors for the Apple&reg; M1 microarchitecture</H1>
<UL>
<LI>\ref APPLE_M1_PMC "General-purpose counters"</LI>
</UL>

\anchor APPLE_M1_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Apple&reg; M microarchitecture provides 10 general-purpose counters consisting of a config and a counter register. LIKWID differentiates internally between Icestorm and Firestorm cores. Only a few events are currently known</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PMC0</TD>
  <TD>CPU_CYCLES</TD>
</TR>
<TR>
  <TD>PMC1</TD>
  <TD>INST_RETIRED</TD>
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
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC6</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC7</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC8</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC9</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the general-purpose counters of Apple&reg; M.</P>
