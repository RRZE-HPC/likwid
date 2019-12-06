/*! \page zen AMD&reg; Zen (Ryzen, Epyc)

<H1>Available performance monitors for the AMD&reg; Zen microarchitecture</H1>
<UL>
<LI>\ref ZEN_PMC "General-purpose counters"</LI>
<LI>\ref ZEN_POWER_CORE "CPU core energy counters"</LI>
<LI>\ref ZEN_CPMC "L3 cache general-purpose counters"</LI>
<LI>\ref ZEN_POWER_SOCKET "Socket energy counters"</LI>
<LI> \ref ZEN_DATA_FABRIC "Data Fabric counters"</LI>
</UL>


\anchor ZEN_PMC
<H2>General-purpose counters</H2>
<P>The AMD&reg; Zen microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
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
\anchor ZEN_POWER_CORE
<H2>Power counters</H2>
<P>The AMD&reg; Zen microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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
<P>There are more energy counters but only one for each socket (\ref ZEN_POWER_SOCKET)</P>


<H1>Counters available for one hardware thread per shared L3 cache</H1>
\anchor ZEN_CPMC
<H2>L3 general-purpose counters</H2>
<P>The AMD&reg; Zen microarchitecture provides 6 general-purpose counters for measuring L3 cache events. They consist of a config and a counter register.</P>

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
  <TD>match0</TD>
  <TD>4 bit hex value</TD>
  <TD>Set bits 48-51 in config register</TD>
  <TD>Controls which L3 slice are counting. Bits: 0 = L3 slice 0, 1 = L3 slice 1, 2 = L3 slice 2, 3 = L3 slice 3. Default are all slices</TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor ZEN_POWER_SOCKET
<H2>Power counters</H2>
<P>The AMD&reg; Zen microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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
<P>There are more energy counters for each CPU core (\ref ZEN_POWER_CORE)</P>

\anchor ZEN_DATA_FABRIC
<H2>Data Fabric counters</H2>
<P>The AMD&reg; Zen microarchitecture provides additional Uncore counters for the so-called Data Fabric. The register configuration was published but no event configurations. The events provided for the AMD&reg; Zen microarchitecture are backported from the documentation of the AMD&reg; Zen2 microarchitecture. The counters are NUMA node specific.</P>

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
