\page sapphirerapids Intel&reg; SapphireRapids

<P>This page is valid for SapphireRapids.</P>

<H1>Available performance monitors for the Intel&reg; SapphireRapids microarchitecture</H1>
<UL>
<LI>\ref SPR_FIXED "Fixed-purpose counters"</LI>
<LI>\ref SPR_METRICS "Performance metric counters"</LI>
<LI>\ref SPR_PMC "General-purpose counters"</LI>
<LI>\ref SPR_THERMAL "Thermal counters"</LI>
<LI>\ref SPR_VOLTAGE "Core voltage counters"</LI>
<LI>\ref SPR_POWER "Power measurement counters"</LI>
<LI>\ref SPR_UBOX "Uncore global counters"</LI>
<LI>\ref SPR_CBOX "Last level cache counters"</LI>
<LI>\ref SPR_MBOX "Memory channel counters"</LI>
<LI>\ref SPR_WBOX "Power control unit counters"</LI>
<LI>\ref SPR_QBOX "UPI interface counters"</LI>
<LI>\ref SPR_SBOX "Mesh-to-UPI counters (M3UPI)"</LI>
<LI>\ref SPR_IBOX "IIO box counters"</LI>
<LI>\ref SPR_BBOX "Mesh2Memory counters"</LI>
<!--<LI>\ref SPR_PBOX "Mesh2PCIe counters"</LI>-->
<LI>\ref SPR_HBM "High bandwidth memory counters"</LI>
</UL>

<H2>Some units are currently not supported</H2>
<UL>
<LI>free-running IIO counters</LI>
<LI>fixed-purpose counters in \ref SPR_WBOX</LI>
<LI>free-running memory controller counters</LI>
<LI>M2HBM unit</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor SPR_FIXED
<H2>Fixed-purpose counters</H2>
<P>Since the Core2 microarchitecture, Intel&reg; provides a set of fixed-purpose counters. Each can measure only one specific event. The Intel&reg; SapphireRapids architecture adds a fourth fixed-purpose counter for the event TOPDOWN_SLOTS.</P>
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

\anchor SPR_METRICS
<H2>Performance metric counters</H2>
<P>With the Intel&reg; Icelake SP microarchitecture a new class of core-local counters was introduced, the so-called perf-metrics. The reflect the first level of the <A HREF="https://software.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html">Top-down Microarchitecture Analysis</A> tree. The events return the fraction of slots used by the event.</P>
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

\anchor SPR_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; SapphireRapids microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
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

<P>If HyperThreading is disabled, each core can use 8 general-purpose counters named &lt;4-7&gt;.


\anchor SPR_THERMAL
<H2>Thermal counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides one register for the current core temperature.</P>
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

\anchor SPR_VOLTAGE
<H2>Core voltage counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides one register for the current core voltage.</P>
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
\anchor SPR_POWER
<H2>Power counter</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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

\anchor SPR_UBOX
<H2>Uncore global counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements for the global uncore.</P>
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
<TR>
  <TD>UBOXFIX</TD>
  <TD>UNCORE_CLOCK</TD>
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

\anchor SPR_CBOX
<H2>Last level cache counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements for the last level cache segments.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0-55&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-55&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-55&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-55&gt;C3</TD>
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

\anchor SPR_MBOX
<H2>Memory channel counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements of the memory channels of each integrated Memory Controllers (iMC) in the Uncore.<BR>
The integrated Memory Controllers performance counters are exposed to the operating system through MMIO interfaces. There may be 8 memory controllers in the system. Each controller provides four memory channels. Each channel has 4 different general-purpose counters and one fixed counter for the DRAM clock. The channels of the first memory controller are MBOX0-3, the two channels of the second memory controller are named MBOX3-7, and so on. The name MBOX originates from the Nehalem EX Uncore monitoring where those functional units are called MBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX&lt;0-15&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-15&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-15&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-15&gt;C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-15&gt;FIX</TD>
  <TD>DRAM_CLOCKTICKS</TD>
</TR>
</TABLE>

<H3>Available Options (Only for counter MBOX&lt;0-15&gt;C&lt;0-3&gt;)</H3>
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

\anchor SPR_WBOX
<H2>Power control unit counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements of the power control unit (PCU) in the Uncore.
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
  <TD>PCU0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCU1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCU2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCU3</TD>
  <TD>*</TD>
</TR>
<!--
<TR>
  <TD>WBOX0FIX</TD>
  <TD>CORES_IN_C3</TD>
</TR>
<TR>
  <TD>WBOX1FIX</TD>
  <TD>CORES_IN_C6</TD>
</TR>
<TR>
  <TD>WBOX2FIX</TD>
  <TD>CORES_IN_P3</TD>
</TR>
<TR>
  <TD>WBOX3FIX</TD>
  <TD>CORES_IN_P6</TD>
</TR>
-->
</TABLE>

<H3>Available Options (Only for PCU&lt;0-3&gt; counters)</H3>
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


\anchor SPR_QBOX
<H2>UPI interface counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements of the Ultra Path Interconnect Link layer (UPI LL) in the Uncore.<BR>
The UPI hardware performance counters are exposed to the operating system through PCI interfaces. There are four of those interfaces/units/slots for the UPI. The actual amount of UPI units depend on the CPU core count of one socket. If your system has not all interfaces but interface 0 does not work, try the other ones. The QBOX was introduced for the Skylake microarchitecture.</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>UPI&lt;0-3&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPI&lt;0-3&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPI&lt;0-3&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPI&lt;0-3&gt;C3</TD>
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

\anchor SPR_SBOX
<H2>Mesh-to-UPI counters (M3UPI)</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements of the Mesh-to-UPI (M3UPI) interface in the Uncore.<BR>
The Mesh-to-UPI performance counters are exposed to the operating system through PCI interfaces. Since the RBOXes manage the traffic from the LLC-connecting mesh interface on the socket with the UPI interfaces (QBOXes), the amount is similar to the amount of QBOXes (4). See at UPI units how many are available for which system configuration.

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>RBOX&lt;0-3&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0-3&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0-3&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0-3&gt;C3</TD>
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


\anchor SPR_IBOX
<H2>IIO box counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements of the IIO box in the Uncore.<BR>
The IIO box counters are exposed to the operating system through the MSR interface.
</P>


<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IIO&lt;0-12&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IIO&lt;0-12&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IIO&lt;0-12&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IIO&lt;0-12&gt;C3</TD>
  <TD>*</TD>
</TR>
<!--
<TR>
  <TD>IBOX&lt;0-5&gt;PORT0</TD>
  <TD>IIO_BANDWIDTH_IN_PORT0</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;PORT1</TD>
  <TD>IIO_BANDWIDTH_IN_PORT1</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;PORT2</TD>
  <TD>IIO_BANDWIDTH_IN_PORT2</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;PORT3</TD>
  <TD>IIO_BANDWIDTH_IN_PORT3</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;PORT4</TD>
  <TD>IIO_BANDWIDTH_IN_PORT4</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;PORT5</TD>
  <TD>IIO_BANDWIDTH_IN_PORT5</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;PORT6</TD>
  <TD>IIO_BANDWIDTH_IN_PORT6</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;PORT7</TD>
  <TD>IIO_BANDWIDTH_IN_PORT7</TD>
</TR>
-->

</TABLE>

<H3>Available Options (Only for IIO&lt;0-12&gt; counters)</H3>
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
<!--
<TR>
  <TD>match0</TD>
  <TD>8 bit hex value</TD>
  <TD>Channel mask filter, sets bits 36-43 in config register</TD>
  <TD>Only for TCBOX&lt;0-5&gt; counters. Check <A HREF="https://cdrdv2.intel.com/v1/dl/getContent/639778">3rd Gen Intel&reg; Xeon&reg; Processor Scalable Family, Codename Ice Lake, Uncore Performance Monitoring Reference Manual</A> for more information.</TD>
</TR>
-->
</TABLE>


\anchor SPR_BBOX
<H2>Mesh2Memory counters</H2>
<P>The Intel&reg; SapphireRapids microarchitecture provides measurements of the mesh (M2M) which connects the cores with the Uncore devices.<br>
The M2M devices is first introduced in the Intel&reg; Skylake SP microarchitecture. There was no suitable unit name for this, so LIKWID calls them simply M2M.
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>M2M&lt;0-3&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0-3&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0-3&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0-3&gt;C3</TD>
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

<!--
\anchor SPR_PBOX
<H2>Mesh2PCIe counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements of the mesh to the IIO devices. The description from Intel&reg;:<br>
<I>M2PCIe blocks manage the interface between the Mesh and each IIO stack.</I><br>
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>P&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>P&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>P&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>P&lt;0-5&gt;C3</TD>
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
-->

\anchor SPR_HBM
<H2>High-Bandwidth Memory (HBM) counters</H2>
Some SapphireRapids systems provide on-chip HBM. If it is available, there are also corresponding perfmon units. The available events are almost similar to \ref SPR_MBOX.
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>HBM&lt;0-32&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>HBM&lt;0-32&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>HBM&lt;0-32&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>HBM&lt;0-32&gt;C3</TD>
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

