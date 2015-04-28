/*! \page westmereex Intel&reg; Westmere EX

<P>The Intel&reg; Westmere EX microarchitecture has the same features as the Intel&reg; Westmere architecture. There are some additional features like a second OFFCORE_RESPONSE register and an addr/opcode matching unit for general-purpose counters in the Uncore.</P>

<H1>Available performance monitors for the Intel&reg; Westmere EX microarchitecture</H1>
<UL>
<LI>\ref WESEX_FIXED "Fixed-purpose counters"</LI>
<LI>\ref WESEX_PMC "General-purpose counters"</LI>
<LI>\ref WESEX_MBOX "Memory controller counters"</LI>
<LI>\ref WESEX_BBOX "Home Agent counters"</LI>
<LI>\ref WESEX_RBOX "Crossbar router counters"</LI>
<LI>\ref WESEX_CBOX "Last Level cache counters"</LI>
<LI>\ref WESEX_SBOX "LLC-to-QPI interface counters"</LI>
<LI>\ref WESEX_WBOX "Power control unit counters"</LI>
<LI>\ref WESEX_UBOX "Uncore management counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor WESEX_FIXED
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

\anchor WESEX_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; Westmere EX microarchitecture provides 4 general-purpose counters consisting of a config and a counter register. They are core-local, hence each hardware thread has its own set of general-purpose counters.</P>
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
<P>The Intel&reg; Westmere EX microarchitecture provides measureing of offcore events in PMC counters. Therefore the stream of offcore events must be filtered using the OFFCORE_RESPONSE registers. The Intel&reg; Westmere EX microarchitecture has two of those registers. Although the PMC counters are core-local, the offcore filtering can only be done by one hardware thread attached to a shared L2 cache. LIKWID defines some events that perform the filtering according to the event name but also own filtering can be applied with the OFFCORE_RESPONSE_0_OPTIONS and OFFCORE_RESPONSE_1_OPTIONS events. Only for those events two more counter options are available:</P>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>match0</TD>
  <TD>8 bit hex value</TD>
  <TD>Input value masked with 0xFF and written to bits 0-7 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A>.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>8 bit hex value</TD>
  <TD>Input value masked with 0xF7 and written to bits 8-15 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A>.</TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor WESEX_MBOX
<H2>MBOX counters</H2>
<P>The Intel&reg; Westmere EX microarchitecture provides measurements of the memory controllers in the Uncore. The description from Intel&reg;:<BR>
<I>The memory controller interfaces to the Intel&reg; 7500 Scalable Memory Buffers and translates read and write commands into specific Intel&reg; Scalable Memory Interconnect (Intel&reg; SMI) operations. Intel SMI is based on the FB-DIMM architecture, but the Intel 7500 Scalable Memory Buffer is not an AMB2 device and has significant exceptions to the FB-DIMM2 architecture. The memory controller also provides a variety of RAS features, such as ECC, memory scrubbing, thermal throttling, mirroring, and DIMM sparing. Each socket has two independent memory controllers, and each memory controller has two Intel SMI channels that operate in lockstep.
</I><BR>
The Intel&reg; Westmere EX microarchitecture has 2 memory controllers, each with 6 general-purpose counters. They are exposed through the MSR interface to the operating system kernel. The MBOX and RBOX setup routines are taken from Likwid 3, they are not as flexible as the newer setup routines but programming of the MBOXes and RBOXes is tedious for Westmere EX. It is not possible to specify a FVID (Fill Victim Index) for the MBOX or IPERF option for RBOXes.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX<0,1>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0,1>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0,1>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0,1>C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0,1>C4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX<0,1>C5</TD>
  <TD>*</TD>
</TR>
</TABLE>
<H3>Special handling for events</H3>
<P>For the events DRAM_CMD_ALL and DRAM_CMD_ILLEGAL two counter options are available:</P>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>match0</TD>
  <TD>34 bit address</TD>
  <TD>Set bits 0-33 in MSR_M<0,1>_PMON_ADDR_MATCH register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>mask0</TD>
  <TD>60 bit hex value</TD>
  <TD>Extract bits 6-33 from address and set bits 0-27 in MSR_M<0,1>_PMON_ADDR_MASK register</TD>
  <TD></TD>
</TR>
<P>For the events THERM_TRP_DN and THERM_TRP_UP you cannot measure events for all and one specific DIMM simultaneously because they programm the same filter register MSR_M<0,1>_PMON_MSC_THR and have contrary configurations.</P>
<P>Although the events FVC_EV<0-3> are available to measure multiple memory events, some overlap and do not allow simultaneous measureing. That's because they programm the same filter register MSR_M<0,1>_PMON_ZDP and have contrary configurations. One case are the FVC_EV<0-3>_BBOX_CMDS_READS and FVC_EV<0-3>_BBOX_CMDS_WRITES events that measure memory reads or writes but cannot be measured at the same time.</P>
</TABLE>


\anchor WESEX_BBOX
<H2>BBOX counters</H2>
<P>The Intel&reg; Westmere EX microarchitecture provides measurements of the Home Agent in the Uncore. The description from Intel&reg;:<BR>
<I>The B-Box is responsible for the protocol side of memory interactions, including coherent and non-coherent home agent protocols (as defined in the Intel® QuickPath Interconnect Specification). Additionally, the B-Box is responsible for ordering memory reads/writes to a given address such that the M-Box does not have to perform this conflict checking. All requests for memory attached to the coupled M-Box must first be ordered through the B-Box.
</I><BR>
The memory traffic in an Intel&reg; Westmere EX system is controller by the Home Agents. Each MBOX has a corresponding BBOX. Each BBOX offers 4 general-purpose counters. They are exposed through the MSR interface to the operating system kernel.
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
<H3>Special handling for events</H3>
<P>For the matching events MSG_IN_MATCH, MSG_ADDR_IN_MATCH, MSG_OPCODE_ADDR_IN_MATCH, MSG_OPCODE_IN_MATCH, MSG_OPCODE_OUT_MATCH, MSG_OUT_MATCH, OPCODE_ADDR_IN_MATCH, OPCODE_IN_MATCH, OPCODE_OUT_MATCH and ADDR_IN_MATCH two counter options are available:</P>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
  <TD>match0</TD>
  <TD>60 bit hex value</TD>
  <TD>Set bits 0-59 in MSR_B<0,1>_PMON_MATCH register</TD>
  <TD>For register layout and valid settings see <A HREF="http://www.intel.com/content/www/us/en/processors/xeon/xeon-e7-family-uncore-performance-programming-guide.html">Intel&reg; Xeon&reg; Processor E7 Family Uncore Performance Monitoring Guide</A></TD>
</TR>
<TR>
  <TD>mask0</TD>
  <TD>60 bit hex value</TD>
  <TD>Set bits 0-59 in MSR_B<0,1>_PMON_MASK register</TD>
  <TD>For register layout and valid settings see <A HREF="http://www.intel.com/content/www/us/en/processors/xeon/xeon-e7-family-uncore-performance-programming-guide.html">Intel&reg; Xeon&reg; Processor E7 Family Uncore Performance Monitoring Guide</A></TD>
</TR>
</TABLE>

\anchor WESEX_RBOX
<H2>RBOX counters</H2>
<P>The Intel&reg; Westmere EX microarchitecture provides measurements of the crossbar rounter in the Uncore. The description from Intel&reg;:<BR>
<I>The Crossbar Router (R-Box) is a 8 port switch/router implementing the Intel&reg; QuickPath Interconnect Link and Routing layers. The R-Box is responsible for routing and transmitting all intra- and inter-processor communication.
</I><BR>
The Intel&reg; Westmere EX microarchitecture has two interfaces to the RBOX although each socket contains only one crossbar router, RBOX0 is the left part and RBOX1 is the right part of the single RBOX. Each RBOX side offers 8 general-purpose counters. They are exposed through the MSR interface to the operating system kernel. The MBOX and RBOX setup routines are taken from Likwid 3, they are not as flexible as the newer setup routines but programming of the MBOXes and RBOXes is tedious for Westmere EX. It is not possible to specify a FVID (Fill Victim Index) for the MBOX or IPERF option for RBOXes. 
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>RBOX<0,1>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1>C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1>C4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1>C5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1>C6</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>RBOX<0,1>C7</TD>
  <TD>*</TD>
</TR>
</TABLE>

\anchor WESEX_CBOX
<H2>CBOX counters</H2>
<P>The Intel&reg; Westmere EX microarchitecture provides measurements of the LLC coherency engine in the Uncore. The description from Intel&reg;:<BR>
<I>For the Intel Xeon Processor 7500 Series, the LLC coherence engine (C-Box) manages the interface between the core and the last level cache (LLC). All core transactions that access the LLC are directed from the core to a C-Box via the ring interconnect. The C-Box is responsible for managing data delivery from the LLC to the requesting core. It is also responsible for maintaining coherence between the cores within the socket that share the LLC; generating snoops and collecting snoop responses to the local cores when the MESI protocol requires it.<BR>
The C-Box is also the gate keeper for all Intel&reg; QuickPath Interconnect (Intel&reg; QPI) messages that originate in the core and is responsible for ensuring that all Intel QuickPath Interconnect messages that pass through the socket’s LLC remain coherent.
</I><BR>
The Intel&reg; Westmere EX microarchitecture has 10 CBOX instances. Each CBOX offers 6 general-purpose counters. They are exposed through the MSR interface to the operating system kernel.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX<0-9>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-9>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-9>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-9>C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-9>C4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX<0-9>C5</TD>
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

\anchor WESEX_SBOX
<H2>SBOX counters</H2>
<P>The Intel&reg; Westmere EX microarchitecture provides measurements of the LLC-to-QPI interface in the Uncore. The description from Intel&reg;:<BR>
<I>The S-Box represents the interface between the last level cache and the system interface. It manages flow control between the C and R & B-Boxes. The S-Box is broken into system bound (ring to Intel&reg; QPI) and ring bound (Intel&reg; QPI to ring) connections.<BR>
As such, it shares responsibility with the C-Box(es) as the Intel&reg; QPI caching agent(s). It is responsible for converting C-box requests to Intel&reg; QPI messages (i.e. snoop generation and data response messages from the snoop response) as well as converting/forwarding ring messages to Intel&reg; QPI packets and vice versa.
</I><BR>
The Intel&reg; Westmere EX microarchitecture has 2 SBOX instances. Each SBOX offers 4 general-purpose counters. They are exposed through the MSR interface to the operating system kernel.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>SBOX<0,1>C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX<0,1>C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX<0,1>C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX<0,1>C3</TD>
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

<H3>Special handling for events</H3>
<P>Only for the TO_R_PROG_EV events two counter options are available:</P>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>match0</TD>
  <TD>64 bit hex value</TD>
  <TD>Set bit 0-63 in MSR_S<0,1>_PMON_MATCH register</TD>
  <TD>For register layout and valid settings see <A HREF="http://www.intel.com/content/www/us/en/processors/xeon/xeon-e7-family-uncore-performance-programming-guide.html">Intel&reg; Xeon&reg; Processor E7 Family Uncore Performance Monitoring Guide</A></TD>
</TR>
<TR>
  <TD>mask0</TD>
  <TD>39 bit hex value</TD>
  <TD>Set bit 0-38 in MSR_S<0,1>_PMON_MASK register</TD>
  <TD>For register layout and valid settings see <A HREF="http://www.intel.com/content/www/us/en/processors/xeon/xeon-e7-family-uncore-performance-programming-guide.html">Intel&reg; Xeon&reg; Processor E7 Family Uncore Performance Monitoring Guide</A></TD>
</TR>
</TABLE>

\anchor WESEX_WBOX
<H2>WBOX counters</H2>
<P>The Intel&reg; Westmere EX microarchitecture provides measurements of the power controller in the Uncore. The description from Intel&reg;:<BR>
<I>The W-Box is the primary Power Controller for the Intel&reg; Xeon&reg; Processor 7500 Series.
</I><BR>
The Intel&reg; Westmere EX microarchitecture has one WBOX and it offers 4 general-purpose counters and one fixed counter. They are exposed through the MSR interface to the operating system kernel.
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
  <TD>WBOXFIX</TD>
  <TD>UNCORE_CLOCKTICKS</TD>
</TR>
</TABLE>
<H3>Available Options (Only for WBOX<0-3> counters)</H3>
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

\anchor WESEX_UBOX
<H2>UBOX counters</H2>
<P>The Intel&reg; Westmere EX microarchitecture provides measurements of the system configuration controller in the Uncore. The description from Intel&reg;:<BR>
<I>The U-Box serves as the system configuration controller for the Intel&reg; Xeon&reg; Processor E7 Family.
</I><BR>
The Intel&reg; Westmere EX microarchitecture has one UBOX and it offers a single general-purpose counter. It is exposed through the MSR interface to the operating system kernel.
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
</TABLE>
<H3>Available Options (Only for WBOX<0-3> counters)</H3>
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
</TABLE>

*/
