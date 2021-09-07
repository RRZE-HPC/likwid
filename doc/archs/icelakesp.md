/*! \page skylake Intel&reg; Icelake SP

<P>This page is valid for Icelake SP.</P>

<H1>Available performance monitors for the Intel&reg; Icelake SP microarchitecture</H1>
<UL>
<LI>\ref ICX_FIXED "Fixed-purpose counters"</LI>
<LI>\ref ICX_METRICS "Performance metric counters"</LI>
<LI>\ref ICX_PMC "General-purpose counters"</LI>
<LI>\ref ICX_THERMAL "Thermal counters"</LI>
<LI>\ref ICX_VOLTAGE "Core voltage counters"</LI>
<LI>\ref ICX_POWER "Power measurement counters"</LI>
<LI>\ref ICX_UBOX "Uncore global counters"</LI>
<LI>\ref ICX_CBOX "Last level cache counters"</LI>
<LI>\ref ICX_MBOX "Memory channel counters"</LI>
<LI>\ref ICX_MBOXFREERUN "Memory controller counters"</LI>
<LI>\ref ICX_WBOX "Power control unit counters"</LI>
<LI>\ref ICX_QBOX "UPI interface counters"</LI>
<LI>\ref ICX_SBOX "Mesh-to-UPI counters (M3UPI)"</LI>
<LI>\ref ICX_IBOX "IIO box counters"</LI>
<LI>\ref ICX_BBOX "Mesh2Memory counters"</LI>
<LI>\ref ICX_PBOX "Mesh2PCIe counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor ICX_FIXED
<H2>Fixed-purpose counters</H2>
<P>Since the Core2 microarchitecture, Intel&reg; provides a set of fixed-purpose counters. Each can measure only one specific event. The Intel&reg; Icelake SP architecture adds a fourth fixed-purpose counter for the event TOPDOWN_SLOTS.</P>
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

\anchor ICX_METRICS
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

\anchor ICX_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; Icelake SP microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
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


\anchor ICX_THERMAL
<H2>Thermal counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides one register for the current core temperature.</P>
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

\anchor ICX_VOLTAGE
<H2>Core voltage counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides one register for the current core voltage.</P>
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
\anchor ICX_POWER
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

\anchor ICX_UBOX
<H2>Uncore global counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements for the global uncore.</P>
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

\anchor ICX_CBOX
<H2>Last level cache counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements for the last level cache segments.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0-39&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-39&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-39&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-39&gt;C3</TD>
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
  <TD>state</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 8-15 in config register. Similar to umask</TD>
  <TD>LLC F: 0x80, LLC M: 0x40, LLC E: 0x20, LLC S: 0x10, SF H: 0x08, SF E: 0x04, SF S: 0x02, LLC I: 0x01. Only for the LLC_LOOKUP events</TD>
</TR>
<TR>
  <TD>match0</TD>
  <TD>28 bit hex value</TD>
  <TD>Set bits 32-57 in config register named UmaskExt</TD>
  <TD>See the special event handling section and <A HREF="https://cdrdv2.intel.com/v1/dl/getContent/639778">3rd Gen Intel&reg; Xeon&reg; Processor Scalable Family, Codename Ice Lake, Uncore Performance Monitoring Reference Manual</A> for more information.</TD>
</TR>
</TABLE>

<H3>Special event handling</H3>
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

\anchor ICX_MBOX
<H2>Memory channel counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements of the memory channels of each integrated Memory Controllers (iMC) in the Uncore. The description from Intel&reg;:<BR>
<I>The Ice Lake integrated Memory Controller provides the interface to DRAM and communicates to the rest of the Uncore through the Mesh2Mem block.
<BR>
The memory controller also provides a variety of RAS features, such as ECC, lockstep, memory access retry, memory scrubbing, thermal throttling, mirroring, and rank sparing.
</I><BR>
The integrated Memory Controllers performance counters are exposed to the operating system through MMIO interfaces. There may be four memory controllers in the system. Each controller provides two memory channels. Each channel has 4 different general-purpose counters and one fixed counter for the DRAM clock. The channels of the first memory controller are MBOX0-1, the two channels of the second memory controller are named MBOX2-3, and so on. The name MBOX originates from the Nehalem EX Uncore monitoring where those functional units are called MBOX.
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


\anchor ICX_MBOXFREERUN
<H2>Memory controller counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements of the integrated Memory Controllers (iMC) in the Uncore. The description from Intel&reg;:<BR>
<I>The Ice Lake integrated Memory Controller provides the interface to DRAM and communicates to the rest of the Uncore through the Mesh2Mem block.
<BR>
The memory controller also provides a variety of RAS features, such as ECC, lockstep, memory access retry, memory scrubbing, thermal throttling, mirroring, and rank sparing.
</I><BR>
Each memory controller provides five free-running counters.
</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MDEV&lt;0-3&gt;C0</TD>
  <TD>DDR_READ_BYTES</TD>
</TR>
<TR>
  <TD>MDEV&lt;0-3&gt;C1</TD>
  <TD>DDR_WRITE_BYTES</TD>
</TR>
<TR>
  <TD>MDEV&lt;0-3&gt;C2</TD>
  <TD>PMM_READ_BYTES</TD>
</TR>
<TR>
  <TD>MDEV&lt;0-3&gt;C3</TD>
  <TD>PMM_WRITE_BYTES</TD>
</TR>
<TR>
  <TD>MDEV&lt;0-3&gt;C4</TD>
  <TD>IMC_DEV_CLOCKTICKS</TD>
</TR>
</TABLE>

<P><B>Info:</B> Despite the event names contain the unit BYTES in them, they are counting in a cache line granularity</P>

<H3>Mapping between MBOX&lt;0-7&gt;C&lt;0-3&gt; and MDEV&lt;0-3&gt;C&lt;0-4&gt;:</H3>
<TABLE>
<TR>
  <TH>MDEV</TH>
  <TH>MBOX</TH>
</TR>
<TR>
  <TD>MDEV0</TD>
  <TD>MBOX&lt;0-1&gt;</TD>
</TR>
<TR>
  <TD>MDEV1</TD>
  <TD>MBOX&lt;2-3&gt;</TD>
</TR>
<TR>
  <TD>MDEV2</TD>
  <TD>MBOX&lt;4-5&gt;</TD>
</TR>
<TR>
  <TD>MDEV3</TD>
  <TD>MBOX&lt;6-7&gt;</TD>
</TR>
</TABLE>

\anchor ICX_WBOX
<H2>Power control unit counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements of the power control unit (PCU) in the Uncore. The description from Intel&reg;:<BR>
<I>The PCU is the primary Power Controller for the Ice Lake die, responsible for distributing power to core/uncore components and thermal management. It runs in firmware on an internal micro-controller and coordinates the socket’s power states.</I>
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


\anchor ICX_QBOX
<H2>UPI interface counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the Ultra Path Interconnect Link layer (UPI) in the Uncore. The description from Intel&reg;:<BR>
<I>Snow Ridge uses a coherent interconnect for scaling to multiple sockets known as
Intel&reg; Ultra Path Interconnect (UPI). Intel&reg; UPI technology provides a cache coherent
socket to socket external communication interface between processors. Intel&reg; UPI is
also used as a coherent communication interface between processors and OEM 3rd
party Node Controllers (XNC).
</I><BR>
The UPI hardware performance counters are exposed to the operating system through PCI interfaces. There are three of those interfaces for the UPI. The actual amount of QBOX counters depend on the CPU core count of one socket. If your system has not all interfaces but interface 0 does not work, try the other ones. The QBOX was introduced for the Skylake microarchitecture.</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>QBOX&lt;0-2&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>QBOX&lt;0-2&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>QBOX&lt;0-2&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>QBOX&lt;0-2&gt;C3</TD>
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

\anchor ICX_SBOX
<H2>Mesh-to-UPI counters (M3UPI)</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the Mesh-to-UPI (M3UPI) interface in the Uncore. The description from Intel&reg;:<BR>
<I>M3UPI is the interface between the mesh and the Intel&reg; UPI Link Layer. It is
responsible for translating between mesh protocol packets and flits that are used for
transmitting data across the Intel&reg; UPI interface. It performs credit checking between
the local Intel&reg; UPI LL, the remote Intel&reg; UPI LL and other agents on the local mesh.
</I><BR>
The Mesh-to-UPI performance counters are exposed to the operating system through PCI interfaces. Since the SBOXes manage the traffic from the LLC-connecting mesh interface on the socket with the UPI interfaces (QBOXes), the amount is similar to the amount of QBOXes (3). See at QBOXes how many are available for which system configuration.

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


\anchor ICX_IBOX
<H2>IIO box counters</H2>
<P>The Intel&reg; Skylake SP microarchitecture provides measurements of the IIO box in the Uncore. The description from Intel&reg;:<BR>
<I>IIO stacks are responsible for managing traffic between the PCIe domain and the Mesh
domain. The IIO PMON block is situated near the IIO stack’s traffic controller capturing
traffic controller as well as PCIe root port information. The traffic controller is
responsible for translating traffic coming in from the Mesh (through M2PCIe) and
processed by IRP into the PCIe domain to IO agents such as CBDMA, DMA and PCIe.</I><BR>
The IIO box counters are exposed to the operating system through the MSR interface.
</P>


<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>TCBOX&lt;0-5&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>TCBOX&lt;0-5&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>TCBOX&lt;0-5&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>TCBOX&lt;0-5&gt;C3</TD>
  <TD>*</TD>
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


</TABLE>

<H3>Available Options (Only for TCBOX&lt;0-5&gt; and I&lt;0-5&gt; counters)</H3>
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
<TR>
  <TD>match0</TD>
  <TD>8 bit hex value</TD>
  <TD>Channel mask filter, sets bits 36-43 in config register</TD>
  <TD>Only for TCBOX&lt;0-5&gt; counters. Check <A HREF="https://cdrdv2.intel.com/v1/dl/getContent/639778">3rd Gen Intel&reg; Xeon&reg; Processor Scalable Family, Codename Ice Lake, Uncore Performance Monitoring Reference Manual</A> for more information.</TD>
</TR>
</TABLE>


\anchor SKX_BBOX
<H2>Mesh2Memory counters</H2>
<P>The Intel&reg; Icelake SP microarchitecture provides measurements of the mesh (M2M) which connects the cores with the Uncore devices. The description from Intel&reg;:<br>
<I>For all Boxes that must communicate with the Mesh, there are a common set of events
to capture various kinds of information about traffic flowing through their connection to
the Mesh. The same encodings are used to request the mesh events in each box.</I><br>
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

\anchor ICX_PBOX
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

*/
