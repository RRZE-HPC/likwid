\page graniterapids Intel&reg; GraniteRapids

<P>This page is valid for GraniteRapids.</P>

<H1>Available performance monitors for the Intel&reg; GraniteRapids microarchitecture</H1>
<UL>
<LI>\ref GNR_FIXED "Fixed-purpose counters"</LI>
<LI>\ref GNR_METRICS "Performance metric counters"</LI>
<LI>\ref GNR_PMC "General-purpose counters"</LI>
<LI>\ref GNR_THERMAL "Thermal counters"</LI>
<LI>\ref GNR_VOLTAGE "Core voltage counters"</LI>
<LI>\ref GNR_POWER "Power measurement counters"</LI>
<LI>\ref GNR_UBOX "Uncore global counters"</LI>
<LI>\ref GNR_CBOX "Last level cache counters"</LI>
<LI>\ref GNR_MBOX "Memory channel counters"</LI>
<LI>\ref GNR_WBOX "Power control unit counters"</LI>
<LI>\ref GNR_QBOX "UPI interface counters"</LI>
<LI>\ref GNR_RBOX "Mesh-to-UPI counters (B2UPI)"</LI>
<LI>\ref GNR_IBOX "IIO box counters"</LI>
<LI>\ref GNR_BBOX "Mesh2Memory counters (B2CMI)"</LI>
<LI>\ref GNR_PBOX "Mesh2PCIe counters (B2CXL)"</LI>
<LI>\ref GNR_IRP "IIO ring port counters (IRP)"</LI>
<LI>\ref GNR_MDF "Embedded multi-die interconnect bridge (MDF)"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor GNR_FIXED
<H2>Fixed-purpose counters</H2>
<P>Since the Core2 microarchitecture, Intel&reg; provides a set of fixed-purpose counters. Each can measure only one specific event. The Intel&reg; GraniteRapids architecture adds a fourth fixed-purpose counter for the event TOPDOWN_SLOTS.</P>
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
<TR>
  <TD>FIXC3</TD>
  <TD>TOPDOWN_SLOTS</TD>
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

\anchor GNR_METRICS
<H2>Performance metric counters</H2>
<P>With the Intel&reg; Granite Rapids microarchitecture a new class of core-local counters was introduced, the so-called perf-metrics. The reflect the first level of the <A HREF="https://software.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html">Top-down Microarchitecture Analysis</A> tree. The events return the fraction of slots used by the event.</P>
<H3>Counter and events</H3>

<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>TMA0</TD>
  <TD>RETIRING</TD>
</TR>
<TR>
  <TD>TMA1</TD>
  <TD>BAD_SPECULATION</TD>
</TR>
<TR>
  <TD>TMA2</TD>
  <TD>FRONTEND_BOUND</TD>
</TR>
<TR>
  <TD>TMA3</TD>
  <TD>BACKEND_BOUND</TD>
</TR>
</TABLE> 

\anchor GNR_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; GraniteRapids microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PMC0</TD>
  <TD></TD>
</TR>
<TR>
  <TD>PMC1</TD>
  <TD></TD>
</TR>
<TR>
  <TD>PMC2</TD>
  <TD></TD>
</TR>
<TR>
  <TD>PMC3</TD>
  <TD></TD>
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
  <TD>The anythread option is deprecated but still supported. Please check the official Intel&reg; documentation how to use the option properly.</TD>
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
<TR>
  <TD>in_transaction</TD>
  <TD>N</TD>
  <TD>Set bit 32 in config register</TD>
  <TD>Only available if Intel&reg; Transactional Synchronization Extensions are available</TD>
</TR>
<TR>
  <TD>in_transaction_aborted</TD>
  <TD>N</TD>
  <TD>Set bit 33 in config register</TD>
  <TD>Only counter PMC2 and only if Intel&reg; Transactional Synchronization Extensions are available</TD>
</TR>
</TABLE>

<P>If HyperThreading is disabled, each core can use 8 general-purpose counters named PMC&lt;4-7&gt;.


\anchor GNR_THERMAL
<H2>Thermal counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides one register for the current core temperature.</P>
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

\anchor GNR_VOLTAGE
<H2>Core voltage counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides one register for the current core voltage.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>VTG0</TD>
  <TD>VOLTAGE_CORE</TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor GNR_POWER
<H2>Power counter</H2>
<P>The Intel&reg; Granite Rapids microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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
  <TD>PWR2</TD>
  <TD>PWR_PP1_ENERGY</TD>
</TR>
<TR>
  <TD>PWR3</TD>
  <TD>PWR_DRAM_ENERGY</TD>
</TR>
<TR>
  <TD>PWR4</TD>
  <TD>PWR_PLATFORM_ENERGY</TD>
</TR>
</TABLE>

\anchor GNR_UBOX
<H2>Uncore global counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements for the global uncore.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>UBOX0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UBOX1</TD>
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

\anchor GNR_CBOX
<H2>Last level cache counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements for the last level cache segments.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0-125&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-125&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-125&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-125&gt;C3</TD>
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
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<!--
<TR>
  <TD>match0</TD>
  <TD>28 bit hex value</TD>
  <TD>Set bits 32-57 in config register named UmaskExt</TD>
  <TD>See the special event handling section for more information.</TD>
</TR>
-->
</TABLE>

<!--<H3>Special event handling</H3>
<P>Event LLC_LOOKUP uses the umask field for the state specification. Further filters are in a field called UmaskExt. These filters can be addressed with the MASK0 option, thus LLC_LOOKUP_I:CBOX0C0:MATCH0=0x8 would count cache lines in I (=invalid) state filtered by RFOs (MATCH0=0x8). Most LLC_LOOKUP events use umask/state 0xFF for all states. If you want to use multiple states, use the STATE option like STATE=0xE for all SF states.<BR>

The events TOR_INSERTS and TOR_OCCUPANCY also use the UmaskExt field but with different width and meaning:<P>
<TABLE>
<TR>
  <TH>Bit offset</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>0</TD>
  <TD>Data Reads- local or remote. includes prefetches</TD>
</TR>
<TR>
  <TD>1</TD>
  <TD>All write transactions to the LLC - including writebacks to LLC and uncacheable write transactions
Does not include evict cleans or invalidates</TD>
</TR>
<TR>
  <TD>2</TD>
  <TD>Flush or Invalidates</TD>
</TR>
<TR>
  <TD>3</TD>
  <TD>RFOs - local or remote. includes prefetches</TD>
</TR>
<TR>
  <TD>4</TD>
  <TD>Code Reads- local or remote. includes prefetches</TD>
</TR>
<TR>
  <TD>5</TD>
  <TD>Any local or remote transaction to the LLC. Includes prefetches</TD>
</TR>
<TR>
  <TD>6</TD>
  <TD>Any local transaction to LLC, including prefetches from Core</TD>
</TR>
<TR>
  <TD>7</TD>
  <TD>Any local prefetch to LLC from an LLC</TD>
</TR>
<TR>
  <TD>8</TD>
  <TD>Any local prefetch to LLC from Core</TD>
</TR>
<TR>
  <TD>9</TD>
  <TD>Snoop transactions to the LLC from a remote agent</TD>
</TR>
<TR>
  <TD>10</TD>
  <TD>Non-snoop transactions to the LLC from a remote agent</TD>
</TR>
<TR>
  <TD>11</TD>
  <TD>Transactions to locally homed addresses</TD>
</TR>
<TR>
  <TD>12</TD>
  <TD>Transactions to remotely homed addresses</TD>
</TR>
</TABLE>

<P>The events WRITE_NO_CREDITS and READ_NO_CREDITS use the UmaskExt as real extension of the default umask field. Each of the bits corresponds to a memory controller (0-13). The first eight are covered by the bits in umask. The other 6 bits are in the UmaskExt field addressable with the MATCH0 option.</P>

<P>The event LLC_VICTIMS uses the MATCH0 option to differentiate between 'local only' and 'remote only' victims. If nothing is set, 'all' are counted. There are only two settings: MATCH0=0x20 for 'local only' and MATCH0=0x80 for 'remote only'.</P>
-->

\anchor GNR_MBOX
<H2>Memory channel counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements of the memory channels of each integrated Memory Controllers (iMC) in the Uncore.<BR>
The integrated Memory Controllers performance counters are exposed to the operating system through MMIO interfaces.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX&lt;0-11&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-11&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-11&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-11&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor GNR_WBOX
<H2>Power control unit counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements of the power control unit (PCU) in the Uncore.
<BR>
The PCU performance counters are exposed to the operating system through the MSR interface. The name WBOX originates from the Nehalem EX Uncore monitoring where those functional units are called WBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PCU&lt;0-4&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCU&lt;0-4&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCU&lt;0-4&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCU&lt;0-4&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
<!--
<TR>
  <TD>occupancy_edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 31 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>occupancy_invert</TD>
  <TD>N</TD>
  <TD>Set bit 30 in config register</TD>
  <TD></TD>
</TR>
-->
</TABLE>


\anchor GNR_QBOX
<H2>UPI interface counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements of the Ultra Path Interconnect Link layer (UPI LL) in the Uncore.</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>UPI&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPI&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPI&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPI&lt;0-5&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor GNR_RBOX
<H2>Mesh-to-UPI counters (B2UPI)</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements of the Mesh-to-UPI (B2UPI) interface in the Uncore.<BR>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>RBOX&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0-5&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD></TD>
</TR>
</TABLE>


\anchor GNR_IBOX
<H2>IIO box counters</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements of the IIO box in the Uncore.<BR>
The IIO box counters are exposed to the operating system through the MSR interface.
</P>


<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IBOX&lt;0-15&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-15&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-15&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-15&gt;C3</TD>
  <TD>*</TD>
</TR>

</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD></TD>
</TR>
</TABLE>


\anchor GNR_BBOX
<H2>Mesh2Memory counters (B2CMI)</H2>
<P>The Intel&reg; GraniteRapids microarchitecture provides measurements of the mesh B2CMI which connects the cores with the Uncore memory controllers devices.<br>
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>M2M&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0-5&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor GNR_PBOX
<H2>Mesh2CXL counters (B2CXL)</H2>
<P>The Intel&reg; Granite Rapids microarchitecture provides measurements of the mesh to the IIO devices.
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PBOX&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PBOX&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PBOX&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PBOX&lt;0-5&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor GNR_IRP
<H2>IIO ring ports counters (IRP)</H2>
<P>The Intel&reg; Granite Rapids microarchitecture provides measurements of the IIO ring ports.
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IRP&lt;0-15&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IRP&lt;0-15&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IRP&lt;0-15&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IRP&lt;0-15&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
</TABLE>


\anchor GNR_MDF
<H2>Embedded Multi-die Interconnect Bridge (MDF)</H2>
<P>The Intel&reg; Granite Rapids microarchitecture provides measurements of the embedded multi-die interconnect bridge.
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MDF&lt;0-79&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MDF&lt;0-79&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MDF&lt;0-79&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MDF&lt;0-79&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Operation</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

