\page sandybridgeep Intel&reg; SandyBridge EP/EN

<H1>Available performance monitors for the Intel&reg; IvyBridge microarchitecture</H1>
<UL>
<LI>\ref SNBEP_FIXED Fixed-purpose counters</LI>
<LI>\ref SNBEP_PMC General-purpose counters</LI>
<LI>\ref SNBEP_THERMAL Thermal counters</LI>
<LI>\ref SNBEP_POWER Power measurement counters</LI>
<LI>\ref SNBEP_MBOX Integrated memory controller counters</LI>
<LI>\ref SNBEP_CBOX Last Level cache counters</LI>
<LI>\ref SNBEP_UBOX Uncore management counters</LI>
<LI>\ref SNBEP_SBOX Intel&reg; QPI Link Layer counters</LI>
<LI>\ref SNBEP_BBOX Home Agent counters</LI>
<LI>\ref SNBEP_WBOX Power control unit counters</LI>
<LI>\ref SNBEP_RBOX Ring-to-QPI interface counters</LI>
<LI>\ref SNBEP_PBOX Ring-to-PCIe interface counters</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor SNBEP_FIXED
<H2>Fixed counters</H2>
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

\anchor SNBEP_PMC
<H2>PMC counters</H2>
<P>The Intel&reg; SandyBridge microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
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
<P>The Intel&reg; SandyBridge microarchitecture provides measureing of offcore events in PMC counters. Therefore the stream of offcore events must be filtered using the OFFCORE_RESPONSE registers. The Intel&reg; SandyBridge microarchitecture has two of those registers. LIKWID defines some events that perform the filtering according to the event name. Although there are many bitmasks possible, LIKWID natively provides only the ones with response type ANY. Own filtering can be applied with the OFFCORE_RESPONSE_0_OPTIONS and OFFCORE_RESPONSE_1_OPTIONS events. Only for those events two more counter options are available:</P>
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
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/JKT">https://download.01.org/perfmon/JKT</A>.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>22 bit hex value</TD>
  <TD>Input value is written to bits 16-37 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/JKT">https://download.01.org/perfmon/JKT</A>.</TD>
</TR>
</TABLE>

\anchor SNBEP_THERMAL
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
\anchor SNBEP_POWER
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
  <TD>PWR2</TD>
  <TD>PWR_PP1_ENERGY</TD>
</TR>
<TR>
  <TD>PWR3</TD>
  <TD>PWR_DRAM_ENERGY</TD>
</TR>
</TABLE>

\anchor SNBEP_MBOX
<H2>Memory controller counters</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the integrated Memory Controllers (iMC) in the uncore. The description from Intel&reg;:<BR>
<I>The integrated Memory Controller provides the interface to DRAM and communicates to the rest of the uncore through the Home Agent (i.e. the iMC does not connect to the Ring).<BR>
In conjunction with the HA, the memory controller also provides a variety of RAS features, such as ECC, lockstep, memory access retry, memory scrubbing, thermal throttling, mirroring, and rank sparing.
</I><BR>
The uncore management performance counters are exposed to the operating system through PCI interfaces. All SandyBridge based systems have one memory controller. There are 4 different PCI devices per memory controller, each covering one memory channel. Each channel has 4 different general-purpose counters and one fixed counter for the DRAM clock. The name MBOX originates from the Nehalem EX uncore monitoring.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX&lt;0-3&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-3&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-3&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-3&gt;C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-3&gt;FIX</TD>
  <TD>DRAM_CLOCKTICKS</TD>
</TR>
</TABLE>

<H3>Available Options (Only for counter MBOX&lt;0-3&gt;C&lt;0-3&gt;)</H3>
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

\anchor SNBEP_CBOX
<H2>Last Level cache counters</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the LLC coherency engine in the uncore. The description from Intel&reg;:<BR>
<I>The LLC coherence engine (CBo) manages the interface between the core and the last level cache (LLC). All core transactions that access the LLC are directed from the core to a CBo via the ring interconnect. The CBo is responsible for managing data delivery from the LLC to the requesting core. It is also responsible for maintaining coherence between the cores within the socket that share the
LLC; generating snoops and collecting snoop responses from the local cores when the MESIF protocol requires it.
</I><BR>
The Last Level cache performance counters are exposed to the operating system through the MSR interface. SandyBridge EN/EP systems have maximal 8 CBOXes, each with 4 general-purpose counters. The name CBOX originates from the Nehalem EX uncore monitoring.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0-7&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-7&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-7&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-7&gt;C3</TD>
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
  <TD>9 bit opcode identifier, see uncore performance monitoring guide for SandyBridge</TD>
  <TD>Set bits 23-31 in CBOX filter register MSR_UNC_C&lt;0-7&gt;_PMON_BOX_FILTER</TD>
  <TD>LIKWID checks whether the given value is a valid opcode. A list of all valid opcodes can be found in the <A HREF="http://www.intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-uncore-guide.html">Intel&reg; E5-2600 uncore monitoring guide</A></TD>
</TR>
<TR>
  <TD>state</TD>
  <TD>5 bit state representation</TD>
  <TD>Set bits 18-22 in CBOX filter register MSR_UNC_C&lt;0-7&gt;_PMON_BOX_FILTER</TD>
  <TD>F: 0x10,<BR>M: 0x08,<BR>E: 0x04,<BR>S: 0x02,<BR>I: 0x01</TD>
</TR>
<TR>
  <TD>nid</TD>
  <TD>8 bit node ID</TD>
  <TD>Set bits 10-17 in CBOX filter register MSR_UNC_C&lt;0-7&gt;_PMON_BOX_FILTER</TD>
  <TD>Note that for Node ID 0 the hex value should be 0x01.</TD>
</TR>
<TR>
  <TD>tid</TD>
  <TD>5 bit thread ID value</TD>
  <TD>Set bits 0-4 in CBOX filter register MSR_UNC_C&lt;0-7&gt;_PMON_BOX_FILTER</TD>
  <TD>Bit 0 means physical or logical thread, bits 1-3 the core ID</TD>
</TR>
</TABLE>

<H3>Special handling for events</H3>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides an event LLC_LOOKUP which can be filtered with the 'state' option. If no 'state' is set, LIKWID sets the state to 0x1F, the default value to measure all lookups.</P>

\anchor SNBEP_UBOX
<H2>Uncore management counters</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the management box in the uncore. The description from Intel&reg;:<BR>
<I>The UBox serves as the system configuration controller for the Intel&reg; Xeon Processor E5-2600 family uncore.<BR>
In this capacity, the UBox acts as the central unit for a variety of functions:<BR>
<UL>
<LI>The master for reading and writing physically distributed registers across the uncore using the Message Channel.</LI>
<LI>The UBox is the intermediary for interrupt traffic, receiving interrupts from the sytem and dispatching interrupts to the appropriate core.</LI>
<LI>The UBox serves as the system lock master used when quiescing the platform (e.g., Intel&reg; QPI bus lock).</LI>
</UL>
</I><BR>
The uncore management performance counters are exposed to the operating system through the MSR interface. The name UBOX originates from the Nehalem EX uncore monitoring.
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

<H3>Available Options (Only for counter UBOX&lt;0,1&gt;)</H3>
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

\anchor SNBEP_SBOX
<H2>Intel&reg; QPI Link Layer counters</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the QPI Link layer (QPI) in the uncore. The description from Intel&reg;:<BR>
<I>The Intel&reg; QPI Link Layer is responsible for packetizing requests from the caching agent on the way out to the system interface. As such, it shares responsibility with the CBo(s) as the Intel&reg; QPI caching agent(s). It is responsible for converting CBo requests to Intel&reg; QPI messages (i.e. snoop generation and data response messages from the snoop response) as well as converting/forwarding ring
messages to Intel&reg; QPI packets and vice versa.<BR>
The Intel&reg; QPI is split into two separate layers. The Intel&reg; QPI LL (link layer) is responsible for generating, transmitting, and receiving packets with the Intel&reg;® QPI link.
</I><BR>
The QPI hardware performance counters are exposed to the operating system through PCI interfaces. There are two of those interfaces for the QPI. If your system has not all interfaces but interface 0 does not work, try the other one. The name SBOX originates from the Nehalem EX uncore monitoring.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;FIX</TD>
  <TD>QPI_RATE, QPI_SLOW_MODE</TD>
</TR>
</TABLE>

<H3>Available Options (Only for counter SBOX&lt;0,1&gt;C&lt;0-3&gt;)</H3>
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
  <TD>match0</TD>
  <TD>32 bit hex address</TD>
  <TD>Input value masked with 0x8003FFF8 and written to bits 0-31 in the PCI_UNC_QPI_PMON_MATCH_0 register of PCI device</TD>
  <TD>Only if corresponding device available. See <A HREF="http://www.intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-uncore-guide.html">Intel&reg; E5-2600 uncore monitoring guide</A> for fields in PCI_UNC_QPI_PMON_MATCH_0</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>20 bit hex address</TD>
  <TD>Input value masked with 0x000F000F and written to bits 0-19 in the PCI_UNC_QPI_PMON_MATCH_1 register of PCI device</TD>
  <TD>Only if corresponding device available. See <A HREF="http://www.intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-uncore-guide.html">Intel&reg; E5-2600 uncore monitoring guide</A> for fields in PCI_UNC_QPI_PMON_MATCH_1</TD>
</TR>
<TR>
  <TD>mask0</TD>
  <TD>32 bit hex address</TD>
  <TD>Input value masked with 0x8003FFF8 and written to bits 0-31 in the PCI_UNC_QPI_PMON_MASK_0 register of PCI device</TD>
  <TD>Only if corresponding device available. See <A HREF="http://www.intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-uncore-guide.html">Intel&reg; E5-2600 uncore monitoring guide</A> for fields in PCI_UNC_QPI_PMON_MASK_0</TD>
</TR>
<TR>
  <TD>mask1</TD>
  <TD>20 bit hex address</TD>
  <TD>Input value masked with 0x000F000F and written to bits 0-19 in the PCI_UNC_QPI_PMON_MASK_1 register of PCI device</TD>
  <TD>Only if corresponding device available. See <A HREF="http://www.intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-uncore-guide.html">Intel&reg; E5-2600 uncore monitoring guide</A> for fields in PCI_UNC_QPI_PMON_MASK_1</TD>
</TR>
</TABLE>

\anchor SNBEP_BBOX
<H2>BBOX counter</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the Home Agent (HA) in the uncore. The description from Intel&reg;:<BR>
<I>The HA is responsible for the protocol side of memory interactions, including coherent and non-coherent home agent protocols (as defined in the Intel&reg;® QuickPath Interconnect Specification). Additionally, the HA is responsible for ordering memory reads/writes, coming in from the modular Ring, to a given address such that the iMC (memory controller).<BR>
In other words, it is the coherency agent responsible for guarding the memory controller. All requests for memory attached to the coupled iMC must first be ordered through the HA.
</I><BR>
The HA hardware performance counters are exposed to the operating system through PCI interfaces. The name BBOX originates from the Nehalem EX uncore monitoring.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>BBOX0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX3</TD>
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
  <TD>A table of all valid opcodes can be found in the <A HREF="http://www.intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-uncore-guide.html">Intel&reg; E5-2600 uncore monitoring guide</A>.</TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>46 bit hex address</TD>
  <TD>Extract bits 6-31 and set bits 6-31 in PCI_UNC_HA_PMON_ADDRMATCH0 register of PCI device<BR>Extract bits 32-45 and set bits 0-13 in PCI_UNC_HA_PMON_ADDRMATCH1 register of PCI device</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor SNBEP_WBOX
<H2>WBOX counter</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the power control unit (PCU) in the uncore. The description from Intel&reg;:<BR>
<I>The PCU is the primary Power Controller for the physical processor package.<BR>
The uncore implements a power control unit acting as a core/uncore power and thermal manager. It runs its firmware on an internal micro-controller and coordinates the socket’s power states.
</I><BR>
The PCU performance counters are exposed to the operating system through the MSR interface. The name WBOX originates from the Nehalem EX uncore monitoring.
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
  <TD>threshold</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>32 bit hex value</TD>
  <TD>Set bits 0-31 in MSR_UNC_PCU_PMON_BOX_FILTER register</TD>
  <TD>Band0: bits 0-7,<BR>Band1: bits 8-15,<BR>Band2: bits 16-23,<BR>Band3: bits 24-31</TD>
</TR>
<TR>
  <TD>occupancy</TD>
  <TD>2 bit hex value</TD>
  <TD>Set bit 14-15 in config register</TD>
  <TD>Cores<BR>in C0: 0x1,<BR>in C3: 0x2,<BR>in C6: 0x3</TD>
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

\anchor SNBEP_RBOX
<H2>RBOX counter</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the Ring-to-QPI (R3QPI) interface in the uncore. The description from Intel&reg;:<BR>
<I>R3QPI is the interface between the Intel&reg; QPI Link Layer, which packetizes requests, and the Ring.<BR>
R3QPI is the interface between the ring and the Intel&reg; QPI Link Layer. It is responsible for translating between ring protocol packets and flits that are used for transmitting data across the Intel&reg; QPI interface. It performs credit checking between the local Intel&reg; QPI LL, the remote Intel&reg; QPI LL and other agents on the local ring.
</I><BR>
The R3QPI performance counters are exposed to the operating system through PCI interfaces. Since the RBOXes manage the traffic from the LLC-connecting ring interface on the socket with the QPI interfaces (SBOXes), the amount is similar to the amount of SBOXes. See at SBOXes how many are available for which system configuration. The name RBOX originates from the Nehalem EX uncore monitoring.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>RBOX&lt;0,1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0,1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0,1&gt;C2</TD>
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

\anchor SNBEP_PBOX
<H2>PBOX counter</H2>
<P>The Intel&reg; SandyBridge EP/EN microarchitecture provides measurements of the Ring-to-PCIe (R2PCIe) interface in the uncore. The description from Intel&reg;:<BR>
<I>R2PCIe represents the interface between the Ring and IIO traffic to/from PCIe.
</I><BR>
The R2PCIe performance counters are exposed to the operating system through a PCI interface. Independent of the system's configuration, there is only one Ring-to-PCIe interface. The name PBOX originates from the Nehalem EX uncore monitoring.
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

