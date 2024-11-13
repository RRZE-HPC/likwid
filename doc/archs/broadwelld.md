\page broadwelld Intel&reg; Broadwell D

<P>This page is valid for Broadwell D.</P>

<H1>Available performance monitors for the Intel&reg; Broadwell D microarchitecture</H1>
<UL>
<LI>\ref BDW_DE_FIXED "Fixed-purpose counters"</LI>
<LI>\ref BDW_DE_PMC "General-purpose counters"</LI>
<LI>\ref BDW_DE_THERMAL "Thermal counters"</LI>
<LI>\ref BDW_DE_POWER "Power measurement counters"</LI>
<LI>\ref BDW_DE_UBOX "Uncore global counters"</LI>
<LI>\ref BDW_DE_CBOX "Last level cache counters"</LI>
<LI>\ref BDW_DE_BBOX "Home Agent counters"</LI>
<LI>\ref BDW_DE_WBOX "Power control unit counters"</LI>
<LI>\ref BDW_DE_IBOX "Coherency for IIO traffic counters"</LI>
<LI>\ref BDW_DE_MBOX "Integrated memory controller counters"</LI>
<LI>\ref BDW_DE_PBOX "Ring-to-PCIe interface counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor BDW_DE_FIXED
<H2>Fixed-purpose counters</H2>
<P>Since the Core2 microarchitecture, Intel&reg; provides a set of fixed-purpose counters. Each can measure only one specific event.</P>
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

\anchor BDW_DE_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; Broadwell D microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
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

<H3>Special handling for events</H3>
<P>The Intel&reg; Broadwell D microarchitecture provides measureing of offcore events in PMC counters. Therefore the stream of offcore events must be filtered using the OFFCORE_RESPONSE registers. The Intel&reg; Broadwell D microarchitecture has two of those registers. LIKWID defines some events that perform the filtering according to the event name. Although there are many bitmasks possible, LIKWID natively provides only the ones with response type ANY. Own filtering can be applied with the OFFCORE_RESPONSE_0_OPTIONS and OFFCORE_RESPONSE_1_OPTIONS events. Only for those events two more counter options are available:</P>
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
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/BDW-DE">https://download.01.org/perfmon/BDW-DE</A>.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>22 bit hex value</TD>
  <TD>Input value is written to bits 16-37 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/BDW-DE">https://download.01.org/perfmon/BDW-DE</A>.</TD>
</TR>
</TABLE>
<P>The event MEM_TRANS_RETIRED_LOAD_LATENCY is not available because it needs programming of PEBS registers. PEBS is a kernel-level measurement facility for performance monitoring. Although we can program it from user-space, the results are always 0.</P>

\anchor BDW_DE_THERMAL
<H2>Thermal counter</H2>
<P>The Intel&reg; Broadwell microarchitecture provides one register for the current core temperature.</P>
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
\anchor BDW_DE_POWER
<H2>Power counter</H2>
<P>The Intel&reg; Broadwell microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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
</TABLE>

\anchor BDW_DE_UBOX
<H2>Uncore management counters</H2>
<P>The Intel&reg; Broadwell D microarchitecture provides measurements of the management box in the Uncore. The description from Intel&reg;:<BR>
<I>The UBox serves as the system configuration controller for the Intel&reg; Xeon&reg; Processor D-1500 Product Family. In this capacity, the UBox acts as the central unit for a variety of functions:
<UL>
<LI>The master for reading and writing physically distributed registers across using the Message Channel.</LI>
<LI>The UBox is the intermediary for interrupt traffic, receiving interrupts from the system and dispatching interrupts to the appropriate core.</LI>
<LI>The UBox serves as the system lock master used when quiescing the platform (e.g., Intel&reg; QPI bus lock).</LI>
</UL>
</I><BR>
The Uncore management performance counters are exposed to the operating system through the MSR interface. The name UBOX originates from the Nehalem EX Uncore monitoring where those functional units are called UBOX.
</P>
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
  <TD>UBOX_CLOCKTICKS</TD>
</TR>
</TABLE>

<H3>Available Options (Only for UBOX&lt;0,1&gt; counters)</H3>
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
  <TD>threshold</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor BDW_DE_CBOX
<H2>Last level cache counters</H2>
<P>The Intel&reg; Broadwell microarchitecture provides measurements for the last level cache segments.The description from Intel&reg;:<BR>
<I>The LLC coherence engine (CBo) manages the interface between the core and the last level cache (LLC). All core transactions that access the LLC are directed from the core to a CBo via the ring interconnect. The CBo is responsible for managing data delivery from the LLC to the requesting core. It is also responsible for maintaining coherence between the cores within the socket that share the
LLC; generating snoops and collecting snoop responses from the local cores when the MESIF protocol requires it.
</I></P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0-15&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-15&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-15&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-15&gt;C3</TD>
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
<TR>
  <TD>tid</TD>
  <TD>6 bit hex value</TD>
  <TD>Set bits 0-5 in MSR_UNC_C&lt;0-15&gt;_PMON_BOX_FILTER register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>state</TD>
  <TD>7 bit hex value</TD>
  <TD>Set bits 17-23 in MSR_UNC_C&lt;0-15&gt;_PMON_BOX_FILTER register</TD>
  <TD>M': 0x40, D: 0x20, F: 0x10, M: 0x08, E: 0x04, S: 0x02, I: 0x01</TD>
</TR>
<TR>
  <TD>nid</TD>
  <TD>16 bit hex value</TD>
  <TD>Set bits 0-15 in MSR_UNC_C&lt;0-15&gt;_PMON_BOX_FILTER1 register</TD>
  <TD>Note: Node 0 has value 0x0001</TD>
</TR>
<TR>
  <TD>opcode</TD>
  <TD>9 bit hex value</TD>
  <TD>Set bits 20-28 in MSR_UNC_C&lt;0-15&gt;_PMON_BOX_FILTER1 register</TD>
  <TD>A list of valid opcodes can be found in the <A HREF="http://www.intel.com/content/www/us/en/processors/xeon/xeon-d-1500-uncore-performance-monitoring.html">Intel&reg; Xeon D-1500 Uncore Manual</A>.</TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>2 bit hex address</TD>
  <TD>Set bits 30-31 in MSR_UNC_C&lt;0-15&gt;_PMON_BOX_FILTER1 register</TD>
  <TD>See the <A HREF="http://www.intel.com/content/www/us/en/processors/xeon/xeon-d-1500-uncore-performance-monitoring.html">Intel&reg; Xeon D-1500 Uncore Manual</A> for more information.</TD>
</TR>
</TABLE>
<H3>Special handling for events</H3>
<P>The Intel&reg; Broadwell D microarchitecture provides an event LLC_LOOKUP which can be filtered with the 'state' option. If no 'state' is set, LIKWID sets the state to 0x1F, the default value to measure all lookups.</P>

\anchor BDW_DE_BBOX
<H2>Home Agent counters</H2>
<P>The Intel&reg; Broadwell D microarchitecture provides measurements of the Home Agent (HA) in the Uncore. The description from Intel&reg;:<BR>
<I>Each HA is responsible for the protocol side of memory interactions, including coherent and non-coherent home agent protocols (as defined in the Intel&reg; QuickPath Interconnect Specification). Additionally, the HA is responsible for ordering memory reads/writes, coming in from the modular Ring, to a given address such that the IMC (memory controller).
</I><BR>
The Home Agent performance counters are exposed to the operating system through PCI interfaces. There are two of those interfaces for the HA. For systems where each socket has 12 or more cores, there are both HAs available. The name BBOX originates from the Nehalem EX Uncore monitoring where this functional unit is called BBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>BBOX&lt;0,1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX&lt;0,1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX&lt;0,1&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX&lt;0,1&gt;C3</TD>
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
<TR>
  <TD>opcode</TD>
  <TD>6 bit hex value</TD>
  <TD>Set bits 0-5 in PCI_UNC_HA_PMON_OPCODEMATCH register of PCI device</TD>
  <TD></TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>46 bit hex address</TD>
  <TD>Extract bits 6-31 and set bits 6-31 in PCI_UNC_HA_PMON_ADDRMATCH0 register of PCI device<BR>Extract bits 32-45 and set bits 0-13 in PCI_UNC_HA_PMON_ADDRMATCH1 register of PCI device</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor BDW_DE_WBOX
<H2>Power control unit counters</H2>
<P>The Intel&reg; Broadwell D microarchitecture provides measurements of the power control unit (PCU) in the Uncore. The description from Intel&reg;:<BR>
<I>The PCU is the primary Power Controller for the Intel&reg; Xeon&reg; Processor D-1500 Product Family.<BR>
The uncore implements a power control unit acting as a core/uncore power and thermal manager. It runs its firmware on an internal microcontroller and coordinates the socket’s power states.
</I><BR>
The PCU performance counters are exposed to the operating system through the MSR interface. The name WBOX originates from the Nehalem EX Uncore monitoring where those functional units are called WBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>WBOX0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>WBOX1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>WBOX2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>WBOX3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>WBOX0FIX</TD>
  <TD>CORES_IN_C3</TD>
</TR>
<TR>
  <TD>WBOX1FIX</TD>
  <TD>CORES_IN_C6</TD>
</TR>
</TABLE>

<H3>Available Options (Only for WBOX&lt;0-3&gt; counters)</H3>
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
<TR>
  <TD>occupancy</TD>
  <TD>2 bit hex value</TD>
  <TD>Set bit 14-15 in config register</TD>
  <TD>Cores in C0: 0x1, in C3: 0x2, in C6: 0x3</TD>
</TR>
<TR>
  <TD>occupancy_filter</TD>
  <TD>32 bit hex value</TD>
  <TD>Set bits 0-31 in MSR_UNC_PCU_PMON_BOX_FILTER register</TD>
  <TD>Band0: bits 0-7, Band1: bits 8-15, Band2: bits 16-23, Band3: bits 24-31</TD>
</TR>
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
</TABLE>

\anchor BDW_DE_IBOX
<H2>IRP box counters</H2>
<P>The Intel&reg; Broadwell D microarchitecture provides measurements of the IRP box in the Uncore. The description from Intel&reg;:<BR>
<I>IRP is responsible for maintaining coherency for IIO traffic that needs to be coherent (e.g. cross-socket P2P).
</I>

The IRP box counters are exposed to the operating system through the PCI interface. The IBOX was introduced with the Intel&reg; IvyBridge EP/EN/EX microarchitecture.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IBOX&lt;0,1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0,1&gt;C1</TD>
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

\anchor BDW_DE_MBOX
<H2>Memory controller counters</H2>
<P>The Intel&reg; Broadwell D microarchitecture provides measurements of the integrated Memory Controllers (iMC) in the Uncore. The description from Intel&reg;:<BR>
<I>The Intel&reg; Xeon&reg; Processor D-1500 Product Family integrated Memory Controller provides the interface to DRAM and communicates to the rest of the Uncore through the Home Agent (i.e. the IMC does not connect to the Ring).

<BR>
In conjunction with the HA, the memory controller also provides a variety of RAS features.

</I><BR>
The integrated Memory Controllers performance counters are exposed to the operating system through PCI interfaces. There may be two memory controllers in the system. There are 4 different PCI devices per memory controller, but only 2 channels. Each channel has 4 different general-purpose counters and one fixed counter for the DRAM clock. The channels of the first memory controller are MBOX0-3, the four channels of the second memory controller (if available) are named MBOX4-7. The name MBOX originates from the Nehalem EX Uncore monitoring where those functional units are called MBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX&lt;0-7&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-7&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-7&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-7&gt;C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-7&gt;FIX</TD>
  <TD>DRAM_CLOCKTICKS</TD>
</TR>
</TABLE>

<H3>Available Options (Only for counter MBOX&lt;0-7&gt;C&lt;0-3&gt;)</H3>
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

\anchor BDW_DE_PBOX
<H2>Ring-to-PCIe counters</H2>
<P>The Intel&reg; Broadwell D microarchitecture provides measurements of the Ring-to-PCIe (R2PCIe) interface in the Uncore. The description from Intel&reg;:<BR>
<I>R2PCIe represents the interface between the Ring and IIO traffic to/from PCIe.</I><BR>
The Ring-to-PCIe performance counters are exposed to the operating system through a PCI interface. Independent of the system's configuration, there is only one Ring-to-PCIe interface per CPU socket.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PBOX0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PBOX1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PBOX2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PBOX3</TD>
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


*/
