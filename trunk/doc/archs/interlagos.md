/*! \page interlagos AMD&reg; Interlagos

<H1>Available performance monitors for the AMD&reg; Interlagos microarchitecture</H1>
<UL>
<LI>\ref ILG_PMC "General-purpose counters"</LI>
<LI>\ref ILG_UPMC "Northbridge general-purpose counters"</LI>
</UL>


\anchor ILG_PMC
<H2>General-purpose counters</H2>
<P>The AMD&reg; Interlagos microarchitecture provides 6 general-purpose counters consiting of a config and a counter register. They are core-local, hence each hardware thread has its own set of general-purpose counters.</P>
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
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC5</TD>
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
  <TD>The value for threshold can range between 0x0 and 0x1F</TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor ILG_UPMC
<H2>Northbridge general-purpose counters</H2>
<P>The AMD&reg; Interlagos microarchitecture provides 4 general-purpose counters for the Northbridge consiting of a config and a counter register.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>UPMC0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPMC1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPMC2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPMC3</TD>
  <TD>*</TD>
</TR>
</TABLE>


*/
