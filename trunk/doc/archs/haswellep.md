/*! \page haswellep Intel&reg; Haswell EP/EN/EX

<H1>Available performance monitors for the Intel&reg; Haswell EP/EN/EX microarchitecture</H1>
<UL>
<LI>\ref HASEP_FIXED "Fixed-purpose counters"</LI>
<LI>\ref HASEP_PMC "General-purpose counters"</LI>
<LI>\ref HASEP_THERMAL "Thermal counters"</LI>
<LI>\ref HASEP_POWER "Power measurement counters"</LI>
<LI>\ref HASEP_BBOX "Home Agent counters"</LI>
<LI>\ref HASEP_SBOX "Ring transfer counters"</LI>
<LI>\ref HASEP_QBOX "Intel&reg; QPI Link Layer counters"</LI>
<LI>\ref HASEP_CBOX "Last Level cache counters"</LI>
<LI>\ref HASEP_UBOX "Uncore management counters"</LI>
<LI>\ref HASEP_WBOX "Power control unit counters"</LI>
<LI>\ref HASEP_IBOX "Coherency for IIO traffic counters"</LI>
<LI>\ref HASEP_MBOX "Integrated memory controller counters"</LI>
<LI>\ref HASEP_RBOX "Ring-to-QPI interface counters"</LI>
<LI>\ref HASEP_PBOX "Ring-to-PCIe interface counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor HASEP_FIXED
<H2>Fixed-purpose counters</H2>
<P>Since the Core2 microarchitecture, Intel&reg; provides a set of fixed-purpose counters. Each can measure only one specific event. They are core-local.</P>
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

\anchor HASEP_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; Haswell microarchitecture provides 4 general-purpose counters consiting of a config and a counter register. They are core-local.</P>
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
<P>The Intel&reg; Haswell microarchitecture provides measureing of offcore events in PMC counters. Therefore the stream of offcore events must be filtered using the OFFCORE_RESPONSE registers. The Intel&reg; Haswell microarchitecture has two of those registers. Although the PMC counters are core-local, the offcore filtering can only be done by one hardware thread attached to a shared L2 cache. LIKWID defines some events that perform the filtering according to the event name but also own filtering can be applied with the OFFCORE_RESPONSE_0_OPTIONS and OFFCORE_RESPONSE_1_OPTIONS events. Only for those events two more counter options are available:</P>
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
  <TD>Input value masked with 0x8077 and written to bits 0-15 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A>.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>22 bit hex value</TD>
  <TD>Input value masked with 0x3F807F and written to bits 16-37 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A>.</TD>
</TR>
</TABLE>
<P>The event MEM_TRANS_RETIRED_LOAD_LAT is not available because it needs programming of PEBS registers. PEBS is a kernel-level measurement facility. Although we can programm it from user-space, the results are always 0.</P>

\anchor HASEP_THERMAL
<H2>Thermal counter</H2>
<P>The Intel&reg; Haswell microarchitecture provides one register for the current core temperature.</P>
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
\anchor HASEP_POWER
<H2>Power counter</H2>
<P>The Intel&reg; Haswell microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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


\anchor HASEP_BBOX
<H2>BBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the Home Agent (HA) in the Uncore. The description from Intel&reg;:<BR>
<I>The HA is responsible for the protocol side of memory interactions, including coherent and non-coherent home agent protocols (as defined in the Intel&reg;® QuickPath Interconnect Specification). Additionally, the HA is responsible for ordering memory reads/writes, coming in from the modular Ring, to a given address such that the iMC (memory controller).</I><BR>
The HA hardware performance counters are exposed to the operating system through PCI interfaces. There are two of those interfaces for the HA. For systems where each socket has 12 or more cores, there are both HAs available. The name BBOX originates from the Nehalem EX Uncore monitoring where this functional unit is called BBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>BBOX<0,1>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX<0,1>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX<0,1>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX<0,1>C3</TD>
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

\anchor HASEP_SBOX
<H2>SBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture manages the socket internal traffic through ring-based networks. Depending on the system's configuration there are multiple rings in one socket. The SBOXes organizes the traffic between the rings. The description from Intel&reg;:<BR>
<I>The SBox manages the interface between the two Rings.<BR>
The processor is composed of two independent rings connected via two sets of bi-directional buffered switches. Each set of bi-directional buffered switches is partitioned into two ingress/egress pairs. Further, each ingress/egress pair is associated with a ring stop on adjacent rings. This ring stop is termed an Sbo. The processor has up to 4 SBos depending on SKU. The Sbo can be simply thought of as a conduit for the ring, but must also help maintain ordering of traffic to ensure functional correctness in certain cases.
</I><BR>
The SBOX hardware performance counters are exposed to the operating system through the MSR interface. There are maximal four of those interfaces but not all must be present. The name SBOX originates from the Nehalem EX Uncore monitoring where the functional unit to the QPI network is called SBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>SBOX<0-3>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX<0-3>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX<0-3>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX<0-3>C3</TD>
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
  <TD>tid</TD>
  <TD>N</TD>
  <TD>Set bit 19 in config register</TD>
  <TD>This option has no real effect because TID filtering can be activated but there is no possibility to specify the TID somewhere.</TD>
</TR>
</TABLE>

\anchor HASEP_QBOX
<H2>QBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the QPI Link layer (QPI) in the Uncore. The description from Intel&reg;:<BR>
<I>The Intel&reg; QPI Link Layer is responsible for packetizing requests from the caching agent on the way out to the system interface. As such, it shares responsibility with the CBo(s) as the Intel&reg; QPI caching agent(s). It is responsible for converting CBo requests to Intel&reg; QPI messages (i.e. snoop generation and data response messages from the snoop response) as well as converting/forwarding ring messages to Intel&reg; QPI packets and vice versa.On Ivy Bridge, Intel&reg; QPI is split into two separate layers. The Intel&reg; QPI LL (link layer) is responsible for generating, transmitting, and receiving packets with the Intel&reg; QPI link.
</I><BR>
The QPI hardware performance counters are exposed to the operating system through PCI interfaces. There are two of those interfaces for the QPI. The actual amount of QBOX counters depend on the CPU core count of one socket. If your system has not all interfaces but interface 0 does not work, try the other ones. The QBOX was introduced for the Haswell EP microarchitecture, for older uncore-aware architectures the QBOX and the SBOX are the same.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>QBOX<0,1>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>QBOX<0,1>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>QBOX<0,1>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>QBOX<0,1>C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>QBOX<0,1>FIX0</TD>
  <TD>QPI_RATE</TD>
</TR>
<TR>
  <TD>QBOX<0,1>FIX1</TD>
  <TD>QPI_RX_IDLE</TD>
</TR>
<TR>
  <TD>QBOX<0,1>FIX2</TD>
  <TD>QPI_RX_LLR</TD>
</TR>
</TABLE>

<H3>Available Options (Only for QBOX<0,1>C<0,1,2,3> counters)</H3>
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
  <TD>match0</TD>
  <TD>32 bit hex address</TD>
  <TD>Input value masked with 0x8003FFF8 and written to bits 0-31 in the PCI_UNC_V3_QPI_PMON_RX_MATCH_0 register of PCI device</TD>
  <TD>This option matches the receive side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>20 bit hex address</TD>
  <TD>Input value masked with 0x000F000F and written to bits 0-19 in the PCI_UNC_V3_QPI_PMON_RX_MATCH_1 register of PCI device</TD>
  <TD>This option matches the receive side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
<TR>
  <TD>match2</TD>
  <TD>32 bit hex address</TD>
  <TD>Input value masked with 0x8003FFF8 and written to bits 0-31 in the PCI_UNC_V3_QPI_PMON_TX_MATCH_0 register of PCI device</TD>
  <TD>This option matches the transmit side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
<TR>
  <TD>match3</TD>
  <TD>20 bit hex address</TD>
  <TD>Input value masked with 0x000F000F and written to bits 0-19 in the PCI_UNC_V3_QPI_PMON_TX_MATCH_1 register of PCI device</TD>
  <TD>This option matches the transmit side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
<TR>
  <TD>mask0</TD>
  <TD>32 bit hex address</TD>
  <TD>Input value masked with 0x8003FFF8 and written to bits 0-31 in the PCI_UNC_V3_QPI_PMON_RX_MASK_0 register of PCI device</TD>
  <TD>This option masks the receive side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
<TR>
  <TD>mask1</TD>
  <TD>20 bit hex address</TD>
  <TD>Input value masked with 0x000F000F and written to bits 0-19 in the PCI_UNC_V3_QPI_PMON_RX_MASK_1 register of PCI device</TD>
  <TD>This option masks the receive side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
<TR>
  <TD>mask2</TD>
  <TD>32 bit hex address</TD>
  <TD>Input value masked with 0x8003FFF8 and written to bits 0-31 in the PCI_UNC_V3_QPI_PMON_TX_MASK_0 register of PCI device</TD>
  <TD>This option masks the transmit side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
<TR>
  <TD>mask3</TD>
  <TD>20 bit hex address</TD>
  <TD>Input value masked with 0x000F000F and written to bits 0-19 in the PCI_UNC_V3_QPI_PMON_TX_MASK_1 register of PCI device</TD>
  <TD>This option masks the transmit side. Check <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for bit fields.</TD>
</TR>
</TABLE>

\anchor HASEP_CBOX
<H2>CBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the LLC coherency engine in the Uncore. The description from Intel&reg;:<BR>
<I>The LLC coherence engine (CBo) manages the interface between the core and the last level cache (LLC). All core transactions that access the LLC are directed from the core to a CBo via the ring interconnect. The CBo is responsible for managing data delivery from the LLC to the requesting core. It is also responsible for maintaining coherence between the cores within the socket that share the LLC; generating snoops and collecting snoop responses from the local cores when the MESIF protocol requires it.
</I><BR>
The LLC hardware performance counters are exposed to the operating system through the MSR interface. The maximal amount of supported coherency engines for the Intel&reg; Haswell EP/EN/EX microarchitecture is 17. E7-8800 v2 systems have all 17 engines, the E5-2600 v2 only 10 of them and the E5-1600 v2 only 6. It may be possible that your systems does not have all CBOXes, LIKWID will skip the unavailable ones in the setup phase. The name CBOX originates from the Nehalem EX Uncore monitoring where those functional units are called CBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX<0-17>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-17>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-17>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-17>C3</TD>
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
  <TD>5 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>tid</TD>
  <TD>5 bit hex value</TD>
  <TD>Set bits 0-4 in MSR_UNC_C<0-17>_PMON_BOX_FILTER register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>state</TD>
  <TD>6 bit hex value</TD>
  <TD>Set bits 17-22 in MSR_UNC_C<0-17>_PMON_BOX_FILTER register</TD>
  <TD>M: 0x28, F: 0x10, M: 0x08, E: 0x04, S: 0x02, I: 0x01</TD>
</TR>
<TR>
  <TD>nid</TD>
  <TD>16 bit hex value</TD>
  <TD>Set bits 0-15 in MSR_UNC_C<0-17>_PMON_BOX_FILTER1 register</TD>
  <TD>Note: Node 0 has value 0x0001</TD>
</TR>
<TR>
  <TD>opcode</TD>
  <TD>9 bit hex value</TD>
  <TD>Set bits 20-28 in MSR_UNC_C<0-17>_PMON_BOX_FILTER1 register</TD>
  <TD>A list of valid opcodes can be found in the <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A>.</TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>2 bit hex address</TD>
  <TD>Set bits 30-31 in MSR_UNC_C<0-17>_PMON_BOX_FILTER1 register</TD>
  <TD>See the <A HREF="http://www.Intel.de/content/www/de/de/processors/xeon/xeon-e5-2600-v2-uncore-manual.html">Intel&reg; Xeon E5-2600 v3 Uncore Manual</A> for more information.</TD>
</TR>
</TABLE>

<H3>Special handling for events</H3>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides an event LLC_LOOKUP which can be filtered with the 'state' option. If no 'state' is set, LIKWID sets the state to 0x1F, the default value to measure all lookups.</P>

\anchor HASEP_UBOX
<H2>UBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the management box in the Uncore. The description from Intel&reg;:<BR>
<I>The UBox serves as the system configuration controller within the physical processor.
</I><BR>
The uncore management performance counters are exposed to the operating system through the MSR interface. The name UBOX originates from the Nehalem EX Uncore monitoring where those functional units are called UBOX.
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

<H3>Available Options (Only for UBOX<0,1> counters)</H3>
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
</TABLE>

\anchor HASEP_WBOX
<H2>WBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the power control unit (PCU) in the Uncore. The description from Intel&reg;:<BR>
<I>The PCU is the primary Power Controller for the physical processor package.
The uncore implements a power control unit acting as a core/uncore power and thermal manager. It runs its firmware on an internal micro-controller and coordinates the socket’s power states.
</I><BR>
The uncore management performance counters are exposed to the operating system through the MSR interface. The name WBOX originates from the Nehalem EX Uncore monitoring where those functional units are called WBOX.
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

<H3>Available Options (Only for WBOX<0-3> counters)</H3>
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
  <TD>match0</TD>
  <TD>32 bit hex value</TD>
  <TD>Set bits 0-31 in MSR_UNC_PCU_PMON_BOX_FILTER register</TD>
  <TD>Band0: bits 0-7, Band1: bits 8-15, Band2: bits 16-23, Band3: bits 24-31</TD>
</TR>
<TR>
  <TD>occupancy</TD>
  <TD>2 bit hex value</TD>
  <TD>Set bit 14-15 in config register</TD>
  <TD>Cores in C0: 0x1, in C3: 0x2, in C6: 0x3</TD>
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

\anchor HASEP_IBOX
<H2>IBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the IRP box in the Uncore. The description from Intel&reg;:<BR>
<I>IRP is responsible for maintaining coherency for IIO traffic that needs to be coherent (e.g. cross-socket P2P).
</I><BR>
The uncore management performance counters are exposed to the operating system through the PCI interface. The IBOX was introduced with the Intel&reg; Haswell EP/EN/EX microarchitecture.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>IBOX<0,1>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>IBOX<0,1>C1</TD>
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
</TABLE>

\anchor HASEP_MBOX
<H2>MBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the integrated Memory Controllers (iMC) in the Uncore. The description from Intel&reg;:<BR>
<I>The integrated Memory Controller provides the interface to DRAM and communicates to the rest of the uncore through the Home Agent (i.e. the iMC does not connect to the Ring).<BR>
In conjunction with the HA, the memory controller also provides a variety of RAS features, such as ECC, lockstep, memory access retry, memory scrubbing, thermal throttling, mirroring, and rank sparing.
</I><BR>
The uncore management performance counters are exposed to the operating system through PCI interfaces. There may be two memory controllers in the system (E7-8800 v2). There are 4 different PCI devices per memory controller, each covering one memory channel. Each channel has 4 different general-purpose counters and one fixed counter for the DRAM clock. The four channels of the first memory controller are MBOX0-3, the four channels of the second memory controller (if available) are named MBOX4-7. The name MBOX originates from the Nehalem EX Uncore monitoring where those functional units are called MBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX<0-7>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0-7>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0-7>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0-7>C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0-7>FIX</TD>
  <TD>DRAM_CLOCKTICKS</TD>
</TR>
</TABLE>

<H3>Available Options (Only for counter MBOX<0-7>C<0-3>)</H3>
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
</TABLE>

\anchor HASEP_RBOX
<H2>RBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the Ring-to-QPI (R3QPI) interface in the Uncore. The description from Intel&reg;:<BR>
<I>R3QPI is the interface between the Intel&reg; QPI Link Layer, which packetizes requests, and the Ring.<BR>
R3QPI is the interface between the ring and the Intel&reg; QPI Link Layer. It is responsible for translating between ring protocol packets and flits that are used for transmitting data across the Intel&reg; QPI interface. It performs credit checking between the local Intel&reg; QPI LL, the remote Intel&reg; QPI LL and other agents on the local ring.
</I><BR>
The uncore management performance counters are exposed to the operating system through PCI interfaces. Since the RBOXes manage the traffic from the LLC-connecting ring interface on the socket with the QPI interfaces (SBOXes), the amount is similar to the amount of SBOXes. See at SBOXes how many are available for which system configuration. The name RBOX originates from the Nehalem EX Uncore monitoring where those functional units are called RBOX.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>RBOX<0,1,2>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1,2>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1,2>C2</TD>
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
</TABLE>

\anchor HASEP_PBOX
<H2>PBOX counter</H2>
<P>The Intel&reg; Haswell EP/EN/EX microarchitecture provides measurements of the Ring-to-PCIe (R2PCIe) interface in the Uncore. The description from Intel&reg;:<BR>
<I>R2PCIe represents the interface between the Ring and IIO traffic to/from PCIe.
</I><BR>
The uncore management performance counters are exposed to the operating system through a PCI interface. Independent of the system's configuration, there is only one Ring-to-PCIe interface. The name PBOX originates from the Nehalem EX Uncore monitoring where those functional units are called PBOX.
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
</TABLE>


*/
