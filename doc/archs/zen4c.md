/*! \page zen4 AMD&reg; Zen4c (Ryzen, Epyc)

<H1>Available performance monitors for the AMD&reg; Zen4 microarchitecture</H1>
<UL>
<LI>\ref ZEN4C_FIXED "Fixed-purpose counters"</LI>
<LI>\ref ZEN4C_PMC "General-purpose counters"</LI>
<LI>\ref ZEN4C_POWER_CORE "CPU core energy counters"</LI>
<LI>\ref ZEN4C_CPMC "L3 cache general-purpose counters"</LI>
<LI>\ref ZEN4C_POWER_SOCKET "Socket energy counters"</LI>
<LI> \ref ZEN4C_DATA_FABRIC "Data Fabric counters"</LI>
</UL>

\anchor ZEN4C_FIXED
<H2>Fixed-purpose counters</H2>
<P>The AMD&reg; Zen4 microarchitecture provides three fixed-purpose counters for
retired instructions, actual CPU core clock (MPerf: This register increments in 
proportion to the actual number of core clocks cycles while the core is in C0) and
maximum CPU core clock (APerf: Incremented by hardware at the
P0 frequency while the core is in C0).</P>
<H3>Counter and events</H3>

<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>FIXC0</TD>
  <TD>INST_RETIRED_ANY (removed due to bad counts)</TD>
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


\anchor ZEN4C_PMC
<H2>General-purpose counters</H2>
<P>The AMD&reg; Zen4 microarchitecture provides 6 general-purpose counters consisting of a config and a counter register.</P>
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
  <TD>The value for threshold can range between 0x0 and 0x7F</TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per CPU core</H1>
\anchor ZEN4C_POWER_CORE
<H2>Power counters</H2>
<P>The AMD&reg; Zen4 microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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
<P>There are more energy counters but only one for each L3 segment (aka CCD) (\ref ZEN4C_POWER_L3)</P>


<H1>Counters available for one hardware thread per shared L3 cache</H1>
\anchor ZEN4C_CPMC
<H2>L3 general-purpose counters</H2>
<P>The AMD&reg; Zen4 microarchitecture provides 6 general-purpose counters for measuring L3 cache events. They consist of a config and a counter register.</P>
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

<H1>Counters available for one hardware thread per L3 segment (aka CCD)</H1>
\anchor ZEN4C_POWER_L3
<H2>Power counters</H2>
<P>The AMD&reg; Zen4 microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PWR1</TD>
  <TD>RAPL_DRAM_ENERGY</TD>
</TR>
</TABLE>
<P>There are more energy counters for each CPU core (\ref ZEN4C_POWER_CORE)</P>

\anchor ZEN4C_DATA_FABRIC
<H2>Data Fabric counters</H2>
<P>The AMD&reg; Zen4 microarchitecture provides additional Uncore counters for the so-called Data Fabric.</P>

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
<TR>
  <TD>DFC4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC6</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC7</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC8</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC9</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC10</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC11</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC12</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC13</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC14</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>DFC15</TD>
  <TD>*</TD>
</TR>
</TABLE>
