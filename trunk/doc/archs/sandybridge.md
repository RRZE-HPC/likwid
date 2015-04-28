/*! \page sandybridge Intel&reg; SandyBridge

<H1>Available performance monitors for the Intel&reg; IvyBridge microarchitecture</H1>
<UL>
<LI>\ref SNB_FIXED "Fixed-purpose counters"</LI>
<LI>\ref SNB_PMC "General-purpose counters"</LI>
<LI>\ref SNB_THERMAL "Thermal counters"</LI>
<LI>\ref SNB_POWER "Power measurement counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor SNB_FIXED
<H2>Fixed-purpose counters</H2>
<P>Since the Core2 microarchitecture, Intel&reg; provides a set of fixed-purpose counters. Each can measure only one specific event. They are core-local, hence each hardware thread has its own set of fixed counters.</P>
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
  <TD>anythread</TD>
  <TD>N</TD>
  <TD>Set bit 2+(index*4) in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>kernel</TD>
  <TD>N</TD>
  <TD>Set bit (index*4) in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor SNB_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; SandyBridge microarchitecture provides 4 general-purpose counters consiting of a config and a counter register. They are core-local, hence each hardware thread has its own set of general-purpose counters.</P>
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

<H3>Special handling for events</H3>
<P>The Intel&reg; SandyBridge microarchitecture provides measureing of offcore events in PMC counters. Therefore the stream of offcore events must be filtered using the OFFCORE_RESPONSE registers. The Intel&reg; SandyBridge microarchitecture has two of those registers. Although the PMC counters are core-local, the offcore filtering can only be done by one hardware thread attached to a shared L2 cache. LIKWID defines some events that perform the filtering according to the event name but also own filtering can be applied with the OFFCORE_RESPONSE_0_OPTIONS and OFFCORE_RESPONSE_1_OPTIONS events. Only for those events two more counter options are available:</P>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>match0</TD>
  <TD>16 bit hex value</TD>
  <TD>Input value masked with 0x8FFF and written to bits 0-15 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel&reg;&reg;.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg;&reg;&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A>.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>22 bit hex value</TD>
  <TD>Input value masked with 0x3F807F and written to bits 16-37 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel&reg;&reg;.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg;&reg;&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A>.</TD>
</TR>
</TABLE>

\anchor SNB_THERMAL
<H2>Thermal counter</H2>
<P>The Intel&reg; SandyBridge microarchitecture provides one register for the current core temperature.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>TMP0</TD>
  <TD>TEMP_CORE</TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor SNB_POWER
<H2>Power counter</H2>
<P>The Intel&reg; SandyBridge microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PWR0</TD>
  <TD>PWR_PKG_ENERGY</TD>
</TR>
<TR>
  <TD>PWR1</TD>
  <TD>PWR_PP0_ENERGY</TD>
</TR>
<TR>
  <TD>PWR2*</TD>
  <TD>PWR_PP1_ENERGY</TD>
</TR>
<TR>
  <TD>PWR3</TD>
  <TD>PWR_DRAM_ENERGY</TD>
</TR>
</TABLE>
*/
