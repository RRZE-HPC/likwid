\page zen3 AMD&reg; Zen3 (Ryzen, Epyc)

<H1>Available performance monitors for the AMD&reg; Zen3 microarchitecture</H1>
<UL>
<LI>\ref ZEN3_PMC "General-purpose counters"</LI>
<LI>\ref ZEN3_POWER_CORE "CPU core energy counters"</LI>
<LI>\ref ZEN3_CPMC "L3 cache general-purpose counters"</LI>
<LI>\ref ZEN3_POWER_SOCKET "Socket energy counters"</LI>
<LI> \ref ZEN3_DATA_FABRIC "Data Fabric counters"</LI>
</UL>


\anchor ZEN3_PMC
<H2>General-purpose counters</H2>
<P>The AMD&reg; Zen3 microarchitecture provides 6 general-purpose counters consisting of a config and a counter register.</P>
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
  <TD>7 bit hex value</TD>
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

<H1>Counters available for one hardware thread per CPU core</H1>
\anchor ZEN3_POWER_CORE
<H2>Power counters</H2>
<P>The AMD&reg; Zen3 microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PWR0</TD>
  <TD>RAPL_CORE_ENERGY</TD>
</TR>
</TABLE>
<P>There are more energy counters but only one for each socket (\ref ZEN3_POWER_SOCKET)</P>


<H1>Counters available for one hardware thread per shared L3 cache</H1>
\anchor ZEN3_CPMC
<H2>L3 general-purpose counters</H2>
<P>The AMD&reg; Zen3 microarchitecture provides 6 general-purpose counters for measuring L3 cache events. They consist of a config and a counter register.</P>
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
  <TD>Set bits 56-63 in config register</TD>
  <TD>Define which CPU thread should be counted. Bits: 0 = Core0-Thread0, 1 = Core0-Thread1, 2 = Core1-Thread0, 3 = Core1-Thread1, ... Default are all threads</TD>
</TR>
<TR>
  <TD>cid</TD>
  <TD>3 bit hex value</TD>
  <TD>Set bits 42-45 in config register</TD>
  <TD>Selects which core should be counted. If not specified, the all-cores flag (bit 47) is set</TD>
</TR>
<TR>
  <TD>slice</TD>
  <TD>4 bit hex value</TD>
  <TD>Set bits 48-51 in config register</TD>
  <TD>Selects which L3 slice should be counted. If not specified, the all-slices flag (bit 46) is set</TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor ZEN3_POWER_SOCKET
<H2>Power counters</H2>
<P>The AMD&reg; Zen3 microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PWR1</TD>
  <TD>RAPL_PKG_ENERGY</TD>
</TR>
</TABLE>
<P>There are more energy counters for each CPU core (\ref ZEN3_POWER_CORE)</P>

\anchor ZEN3_DATA_FABRIC
<H2>Data Fabric counters</H2>
<P>The AMD&reg; Zen3 microarchitecture provides additional Uncore counters for the so-called Data Fabric.</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>DFC0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC3</TD>
  <TD>*</TD>
</TR>
</TABLE>
