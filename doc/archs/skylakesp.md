/*! \page skylakesp Intel&reg; Skylake SP

<P>This page is valid for Skylake SP. The Skylake SP microarchitecture supports the UBOX and the CBOX Uncore devices.</P>

<H1>Available performance monitors for the Intel&reg; Skylake SP microarchitecture</H1>
<UL>
<LI>\ref SKX_FIXED "Fixed-purpose counters"</LI>
<LI>\ref SKX_PMC "General-purpose counters"</LI>
<LI>\ref SKX_THERMAL "Thermal counters"</LI>
<LI>\ref SKX_POWER "Power measurement counters"</LI>
<LI>\ref SKX_UBOX "Uncore global counters"</LI>
<LI>\ref SKX_CBOX "Last level cache counters"</LI>
<LI>\ref SKX_WBOX "Power control unit counters"</LI>
<LI>\ref SKX_MBOX "Integrated memory controller counters"</LI>
<LI>\ref SKX_SBOX "Intel&reg; UPI Link Layer counters"</LI>
<LI>\ref SKX_RBOX "Mesh-to-UPI interface counters"</LI>
<LI>\ref SKX_M2MBOX "Mesh-to-memory interface"</LI>
<LI>\ref SKX_IRP "Coherency for IIO traffic counters"</LI>
<LI>\ref SKX_IBOXGEN "General-purpose IIO counters"</LI>
<LI>\ref SKX_IBOXFIX "Fixed-purpose IIO counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor SKX_FIXED
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

\anchor SKX_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; Skylake SP microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
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
<P>If HyperThreading is disabled, you can additionally use the PMC registers of the disabled SMT thread and thus have 8 PMC registers</P>


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
<P>The Intel&reg; Skylake SP microarchitecture provides measureing of offcore events in PMC counters. Therefore the stream of offcore events must be filtered using the OFFCORE_RESPONSE registers. The Intel&reg; Skylake SP microarchitecture has two of those registers. LIKWID defines some events that perform the filtering according to the event name. Although there are many bitmasks possible, LIKWID natively provides only the ones with response type ANY. Own filtering can be applied with the OFFCORE_RESPONSE_0_OPTIONS and OFFCORE_RESPONSE_1_OPTIONS events. Only for those events two more counter options are available:</P>
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
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/SKX">https://download.01.org/perfmon/SKX</A>.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>22 bit hex value</TD>
  <TD>Input value is written to bits 16-37 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/SKX">https://download.01.org/perfmon/SKX</A>.</TD>
</TR>
</TABLE>
<P>The event MEM_TRANS_RETIRED_LOAD_LATENCY is not available because it needs programming of PEBS registers. PEBS is a kernel-level measurement facility for performance monitoring. Although we can program it from user-space, the results are always 0.</P>

\anchor SKX_THERMAL
<H2>Thermal counter</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides one register for the current core temperature.</P>
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
\anchor SKX_POWER
<H2>Power counter</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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
  <TD>PWR_SYS_ENERGY</TD>
</TR>
</TABLE>

\anchor SKX_UBOX
<H2>Uncore global counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements for the global uncore.</P>
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


\anchor SKX_CBOX
<H2>Last level cache counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements for the last level cache segments.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0-27&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-27&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-27&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-27&gt;C3</TD>
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
  <TD>8 bit hex value</TD>
  <TD>Set bits 0-7 in MSR_UNC_C&lt;0-27&gt;_PMON_BOX_FILTER register and bit 19 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>state</TD>
  <TD>10 bit hex value</TD>
  <TD>Set bits 17-27 in MSR_UNC_C&lt;0-27&gt;_PMON_BOX_FILTER register</TD>
  <TD>LLC F: 0x80, LLC M: 0x40, LLC E: 0x20, LLC S: 0x10, SF H: 0x08, SF E: 0x04, SF S: 0x02, LLC I: 0x01</TD>
</TR>
<TR>
  <TD>opcode</TD>
  <TD>20 bit hex value</TD>
  <TD>Set bits 9-28 and set bits 17,18,27,28 in MSR_UNC_C&lt;0-27&gt;_PMON_BOX_FILTER1 register</TD>
  <TD>A list of valid opcodes can be found in the <A HREF="https://www.intel.com/content/www/us/en/processors/xeon/scalable/xeon-scalable-uncore-performance-monitoring-manual.html">Intel&reg; Xeon SP (v6) Uncore Manual</A>.</TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>2 bit hex address</TD>
  <TD>Set bits 30-31 in MSR_UNC_C&lt;0-27&gt;_PMON_BOX_FILTER1 register</TD>
  <TD>See the <A HREF="https://www.intel.com/content/www/us/en/processors/xeon/scalable/xeon-scalable-uncore-performance-monitoring-manual.html">Intel&reg; Xeon SP (v6) Uncore Manual</A> for more information.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>6 bit hex address</TD>
  <TD>Set bits 0,1,4,5 in MSR_UNC_C&lt;0-27&gt;_PMON_BOX_FILTER1 register</TD>
  <TD>See the <A HREF="https://www.intel.com/content/www/us/en/processors/xeon/scalable/xeon-scalable-uncore-performance-monitoring-manual.html">Intel&reg; Xeon SP (v6) Uncore Manual</A> for more information.</TD>
</TR>
</TABLE>

\anchor SKX_MBOX
<H2>Memory controller counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the integrated Memory Controllers (iMC) in the Uncore. The description from Intel&reg;:<BR>
<I>Intel&reg; Xeon&reg; Processor Scalable Memory Family integrated Memory Controller provides the interface to DRAM and communicates to the rest of the Uncore through the Mesh2Mem block.
<BR>
The memory controller also provides a variety of RAS features, such as ECC, lockstep, memory access retry, memory scrubbing, thermal throttling, mirroring, and rank sparing.
</I><BR>
The integrated Memory Controllers performance counters are exposed to the operating system through PCI interfaces. There may be two memory controllers in the system. There are 3 different PCI devices per memory controller, each handling one memory channels. Each channel has 4 different general-purpose counters and one fixed counter for the DRAM clock. The channels of the first memory controller are MBOX0-2, the four channels of the second memory controller (if available) are named MBOX3-5. The name MBOX originates from the Nehalem EX Uncore monitoring where those functional units are called MBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-5&gt;C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0-5&gt;FIX</TD>
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


\anchor SKX_WBOX
<H2>Power control unit counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the power control unit (PCU) in the Uncore. The description from Intel&reg;:<BR>
<I>The PCU is the primary Power Controller for the Intel&reg; Xeon&reg; Processor Scalable Memory Family die, responsible for distributing power to core/uncore components and thermal management. It runs in firmware on an internal micro-controller and coordinates the socket’s power states.</I>
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
<TR>
  <TD>WBOX2FIX</TD>
  <TD>CORES_IN_P3</TD>
</TR>
<TR>
  <TD>WBOX3FIX</TD>
  <TD>CORES_IN_P6</TD>
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


\anchor SKX_SBOX
<H2>UPI interface counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the Ultra Path Interconnect Link layer (UPI) in the Uncore. The description from Intel&reg;:<BR>
<I>Intel&reg; Xeon&reg; Processor Scalable Memory Family uses a new coherent interconnect for scaling to multiple sockets known as Intel® Ultra Path Interconnect (Intel UPI). Intel&reg; UPI technology provides a cache coherent socket to socket external communication interface between processors.
</I><BR>
The UPI hardware performance counters are exposed to the operating system through PCI interfaces. There are three of those interfaces for the UPI. The actual amount of SBOX counters depend on the CPU core count of one socket. If your system has not all interfaces but interface 0 does not work, try the other ones. The SBOX was introduced for the Nehalem EX microarchitecture.</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>SBOX&lt;0-2&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0-2&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0-2&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0-2&gt;C3</TD>
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
  <TD>nid</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 40-43 and bit 45 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 32-39 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>nid</TD>
  <TD>10 bit hex value</TD>
  <TD>Set bits 46-55 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor SKX_RBOX
<H2>Ring-to-UPI counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the Mesh-to-UPI (M3UPI) interface in the Uncore. The description from Intel&reg;:<BR>
<I>M3UPI is the interface between the mesh and the Intel&reg; UPI Link Layer. It is responsible for translating between mesh protocol packets and flits that are used for transmitting data across the Intel&reg; UPI interface. It performs credit checking between the local Intel&reg; UPI LL, the remote Intel&reg; UPI LL and other agents on the local mesh.
</I><BR>
The Mesh-to-UPI performance counters are exposed to the operating system through PCI interfaces. Since the RBOXes manage the traffic from the LLC-connecting mesh interface on the socket with the UPI interfaces (SBOXes), the amount is similar to the amount of SBOXes. See at SBOXes how many are available for which system configuration. The name RBOX originates from the Nehalem EX Uncore monitoring where those functional units are called RBOX.

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>RBOX&lt;0,1,2&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0,1,2&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX&lt;0,1,2&gt;C2</TD>
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

\anchor SKX_M2MBOX
<H2>Mesh2Memory counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the mesh (M2M) which connects the cores with the Uncore devices. The description from Intel&reg;:<br>
<I>M2M blocks manage the interface between the Mesh (operating on both Mesh and the SMI3 protocol) and the Memory Controllers. M2M acts as intermediary between the local CHA issuing memory transactions to its attached Memory Controller. Commands from M2M to the MC are serialized by a scheduler and only one can cross the interface at a time.</I><br>
The M2M devices is first introduced in the Intel&reg; Skylake SP microarchitecture. There was no suitable unit name for this, so LIKWID calls them simply M2M.
</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>M2M&lt;0,1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0,1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0,1&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>M2M&lt;0,1&gt;C3</TD>
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


\anchor SKX_IBOXGEN
<H2>IIO box counters (general-purpose)</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the IIO box in the Uncore. The description from Intel&reg;:<BR>
<I>IIO stacks are responsible for managing traffic between the PCIe domain and the Mesh domain. The IIO PMON block is situated near the IIO stack’s traffic controller capturing traffic controller as well as PCIe root port information. The traffic controller is responsible for translating traffic coming in from the Mesh (through M2PCIe) and processed by IRP into the PCIe domain to IO agents such as CBDMA, PCIe and MCP.</I><BR>
The IIO box counters are exposed to the operating system through the MSR interface. The IBOX was introduced with the Intel&reg; IvyBridge EP/EN/EX microarchitecture.
</P>

<H3>Box description</H3>
<TABLE>
<TR>
  <TH>Unit number</TH>
  <TH>Unit description</TH>
</TR>
<TR>
  <TD>0</TD>
  <TD>CBDMA</TD>
</TR>
<TR>
  <TD>1</TD>
  <TD>PCIe0</TD>
</TR>
<TR>
  <TD>2</TD>
  <TD>PCIe1</TD>
</TR>
<TR>
  <TD>3</TD>
  <TD>PCIe2</TD>
</TR>
<TR>
  <TD>4</TD>
  <TD>MCP0</TD>
</TR>
<TR>
  <TD>5</TD>
  <TD>MCP1</TD>
</TR>
</TABLE>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX&lt;0-5&gt;CLK</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options (only for counters IBOX&lt;0-5&gt;C3&lt;0-3&gt;)</H3>
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
  <TD>12 bit hex value</TD>
  <TD>Set bits 24-35 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>mask0</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 36-43 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>mask1</TD>
  <TD>3 bit hex value</TD>
  <TD>Set bits 44-46 in config register</TD>
  <TD></TD>
</TR>
</TABLE>


\anchor SKX_IBOXFIX
<H2>IIO box counters (fixed-purpose)</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the IIO box in the Uncore. Besides the general-purpose counters of \ref SKX_IBOXGEN , there are fixed-purpose counters</P>
<H3>Box description</H3>
<TABLE>
<TR>
  <TH>Unit number</TH>
  <TH>Unit description</TH>
</TR>
<TR>
  <TD>0</TD>
  <TD>CBDMA</TD>
</TR>
<TR>
  <TD>1</TD>
  <TD>PCIe0</TD>
</TR>
<TR>
  <TD>2</TD>
  <TD>PCIe1</TD>
</TR>
<TR>
  <TD>3</TD>
  <TD>PCIe2</TD>
</TR>
<TR>
  <TD>4</TD>
  <TD>MCP0</TD>
</TR>
<TR>
  <TD>5</TD>
  <TD>MCP1</TD>
</TR>
</TABLE>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PI0</TD>
  <TD>BANDWIDTH_PORT0_IN</TD>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PI1</TD>
  <TD>BANDWIDTH_PORT1_IN</TD>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PI2</TD>
  <TD>BANDWIDTH_PORT2_IN</TD>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PI3</TD>
  <TD>BANDWIDTH_PORT3_IN</TD>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PO0</TD>
  <TD>BANDWIDTH_PORT0_OUT</TD>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PO1</TD>
  <TD>BANDWIDTH_PORT1_OUT</TD>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PO2</TD>
  <TD>BANDWIDTH_PORT2_OUT</TD>
</TR>
<TR>
  <TD>IBAND&lt;0-5&gt;PO3</TD>
  <TD>BANDWIDTH_PORT3_OUT</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PI0</TD></TD>
  <TD>UTLILIZATION_PORT0_IN</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PI1</TD></TD>
  <TD>UTLILIZATION_PORT1_IN</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PI2</TD></TD>
  <TD>UTLILIZATION_PORT2_IN</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PI3</TD></TD>
  <TD>UTLILIZATION_PORT3_IN</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PO0</TD></TD>
  <TD>UTLILIZATION_PORT0_OUT</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PO1</TD></TD>
  <TD>UTLILIZATION_PORT1_OUT</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PO2</TD></TD>
  <TD>UTLILIZATION_PORT2_OUT</TD>
</TR>
<TR>
  <TD><TD>IUTIL&lt;0-5&gt;PO3</TD></TD>
  <TD>UTLILIZATION_PORT3_OUT</TD>
</TR>
</TABLE>


\anchor SKX_IRP
<H2>IRP box counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the IRP box in the Uncore. The description from Intel&reg;:<BR>
<I>IRP is responsible for maintaining coherency for IIO traffic targeting coherent memory.</I><BR>
The IRP box counters are exposed to the operating system through the MSR interface. The IRP was introduced with the Intel&reg; IvyBridge EP/EN/EX microarchitecture.
</P>

<H3>Box description</H3>
<TABLE>
<TR>
  <TH>Unit number</TH>
  <TH>Unit description</TH>
</TR>
<TR>
  <TD>0</TD>
  <TD>CBDMA</TD>
</TR>
<TR>
  <TD>1</TD>
  <TD>PCIe0</TD>
</TR>
<TR>
  <TD>2</TD>
  <TD>PCIe1</TD>
</TR>
<TR>
  <TD>3</TD>
  <TD>PCIe2</TD>
</TR>
<TR>
  <TD>4</TD>
  <TD>MCP0</TD>
</TR>
<TR>
  <TD>5</TD>
  <TD>MCP1</TD>
</TR>
</TABLE>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IRP&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IRP&lt;0-5&gt;C1</TD>
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
  <TD>12 bit hex value</TD>
  <TD>Set bits 24-35 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

*/
