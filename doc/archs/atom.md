/*! \page atom Intel&reg; Atom

<P>The Intel&reg; Atom performance monitoring counters are equal to the ones of the Intel&reg; Core 2 microarchitecture but the event set is different.</P>
<H1>Available performance monitors for the Intel&reg; Atom microarchitecture</H1>
<UL>
<LI>\ref ATOM_FIXED "Fixed-purpose counters"</LI>
<LI>\ref ATOM_PMC "General-purpose counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor ATOM_FIXED
<H2>Fixed-purpose counters</H2>
<P>The Core2/Atom microarchitecture is the first architecture offering a set of fixed-purpose counters. Each counter can measure only one specific event. They are core-local, hence each hardware thread has its own set of fixed counters.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>FIXC0</TD>
  <TD>INSTR_RETIRED_ANY</TD>
</TR>
<TR>
  <TD>FIXC1</TD>
  <TD>CPU_CLK_UNHALTED_CORE</TD>
</TR>
<TR>
  <TD>FIXC2</TD>
  <TD>CPU_CLK_UNHALTED_REF</TD>
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
  <TD>kernel</TD>
  <TD>N</TD>
  <TD>Set bit (index*4) in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor ATOM_PMC
<H2>General-purpose counters</H2>
<P>The Intel&reg; Atom microarchitecture provides 2 general-purpose counters consiting of a config and a counter register. They are core-local, hence each hardware thread has its own set of general-purpose counters.</P>
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
