/*! \page zen AMD&reg; Zen

<P>This page is valid for AMD&reg; Zen.</P>

<H1>Available performance monitors for the AMD&reg; Zen microarchitecture</H1>
<UL>
<LI>\ref ZEN_FIXED "Fixed-purpose counters"</LI>
<LI>\ref ZEN_PMC "General-purpose counters"</LI>
<LI>\ref ZEN_CBOX "L3 cache counters"</LI>
<LI>\ref ZEN_POWER "Power measurement counters"</LI>
<!-- <LI>\ref ZEN_UPMC "Northbridge counters"</LI> -->
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor ZEN_FIXED
<H2>Fixed-purpose counters</H2>
<P>The AMD&reg; Zen architecture provides some free-running registers with minimal control that offer some specific events</P>
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
  <TD>ACTUAL_CPU_CLOCK or APERF</TD>
</TR>
<TR>
  <TD>FIXC2</TD>
  <TD>MAX_CPU_CLOCK or MPERF</TD>
</TR>
</TABLE>
<P>Note: It is not recommended to use the fixed counters in metrics as they sometimes do not provide accurate results. Instead of INSTR_RETIRED_ANY please use RETIRED_INSTRUCTIONs for the general-purpose counters</P>

\anchor ZEN_PMC
<H2>General-purpose counters</H2>
<P>Commonly the AMD&reg; Zen microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
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



<H1>Counters available for one hardware thread per socket</H1>
\anchor ZEN_POWER
<H2>Power counter</H2>
<P>The AMD&reg; Zen microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PWR0</TD>
  <TD>PWR_CORE_ENERGY</TD>
</TR>
<TR>
  <TD>PWR1</TD>
  <TD>PWR_PKG_ENERGY</TD>
</TR>
</TABLE>
<P>Note: AMD Epyc CPUs partly provide zeros or wrong values for the PWR_CORE_ENERGY event</P>


\anchor ZEN_CBOX
<H2>L3 cache counters</H2>
<P>The AMD&reg; Zen microarchitecture provides measurements for the last level cache segments.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CPMC0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC5</TD>
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
  <TD>tid</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bit 56-63 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>4 bit hex value</TD>
  <TD>Set bits 48-51 in config register</TD>
  <TD></TD>
</TR>
</TABLE>
*/
