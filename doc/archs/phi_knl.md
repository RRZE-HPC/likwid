/*! \page phi_knl Intel&reg; Xeon Phi (Knights Landing)

<H1>Available performance monitors for the Intel&reg; Xeon Phi (Knights Landing) microarchitecture</H1>
<UL>
<LI>\ref KNL_FIXED "Fixed-purpose counters"</LI>
<LI>\ref KNL_PMC "General-purpose counters"</LI>
<LI>\ref KNL_THERMAL "Thermal counters"</LI>
<LI>\ref KNL_POWER "Power measurement counters"</LI>
<LI>\ref KNL_UBOX "Uncore management counters"</LI>
<LI>\ref KNL_CBOX "Last level cache counters"</LI>
<LI>\ref KNL_WBOX "Power control unit general-purpose counters"</LI>
<LI>\ref KNL_MBOX "Memory controller (iMC) counters"</LI>
<LI>\ref KNL_EBOX "Embedded DRAM controller (EDC) counters"</LI>
<LI>\ref KNL_PBOX "Ring-to-PCIe counters"</LI>
<LI>\ref KNL_IBOX "IRP box counters"</LI>

</UL>

<H1>Counters available for each hardware thread</H1>
\anchor KNL_FIXED
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

\anchor KNL_PMC
<H2>General-purpose counters</H2>
<P>The Intel&reg; Xeon Phi (Knights Landing) microarchitecture provides 2 general-purpose counters consisting of a config and a counter register.</P>
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
<P>The Intel&reg; Xeon Phi (Knights Landing) microarchitecture provides measuring of offcore events in PMC counters. Therefore the stream of offcore events must be filtered using the OFFCORE_RESPONSE registers. The Intel&reg; Xeon Phi (Knights Landing) microarchitecture has two of those registers. LIKWID defines some events that perform the filtering according to the event name. Although there are many bitmasks possible, LIKWID natively provides only the ones with response type ANY. Own filtering can be applied with the OFFCORE_RESPONSE_0_OPTIONS and OFFCORE_RESPONSE_1_OPTIONS events. Only OFFCORE_RESPONSE_0_OPTIONS can be used to measure average latencies. Only for those events two more counter options are available:
</P>
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
  <TD>Input value masked with 0xFFFF and written to bits 0-15 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/SLM">https://download.01.org/perfmon/SLM</A>.</TD>
</TR>
<TR>
  <TD>match1</TD>
  <TD>22 bit hex value</TD>
  <TD>Input value is written to bits 16-38 in the OFFCORE_RESPONSE register</TD>
  <TD>Check the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual, Vol. 3, Chapter Performance Monitoring</A> and <A HREF="https://download.01.org/perfmon/SLM">https://download.01.org/perfmon/SLM</A>.</TD>
</TR>
</TABLE>

\anchor KNL_THERMAL
<H2>Thermal counter</H2>
<P>The Intel&reg; Xeon Phi (Knights Landing) microarchitecture provides one register for the current core temperature.</P>
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
\anchor KNL_POWER
<H2>Power counters</H2>
<P>The Intel&reg; Xeon Phi (Knights Landing) microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
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
  <TD>PWR3</TD>
  <TD>PWR_DRAM_ENERGY</TD>
</TR>
</TABLE>

\anchor KNL_UBOX
<H2>Uncore management counters</H2>
<P>The Intel&reg; Xeon Phi (Knights Landing) microarchitecture provides measurements of the management box in the Uncore. The description from Intel&reg;:<BR>
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

\anchor KNL_CBOX
<H2>Last level cache counters</H2>
<p>The Intel&reg; Xeon Phi (KNL) microarchitecture provides measurements for the last level cache segments.</p>
<H3>Counter and events</H3>
<table>
<tr>
  <th>Counter name</th>
  <th>Event name</th>
</tr>
<tr>
  <td>CBOX&lt;0-37&gt;C0</td>
  <td>*</td>
</tr>
<tr>
  <td>CBOX&lt;0-37&gt;C1</td>
  <td>*</td>
</tr>
<tr>
  <td>CBOX&lt;0-37&gt;C2</td>
  <td>*</td>
</tr>
<tr>
  <td>CBOX&lt;0-37&gt;C3</td>
  <td>*</td>
</tr>
</table>
<H3>Available Options</H3>
<table>
<tr>
  <th>Option</th>
  <th>Argument</th>
  <th>Description</th>
  <th>Comment</th>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#edgedetect">edgedetect</a></td>
  <td>N</td>
  <td>Set bit 18 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#invert">invert</a></td>
  <td>N</td>
  <td>Set bit 23 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#threshold0xxxxx">threshold</a></td>
  <td>8 bit hex value</td>
  <td>Set bits 24-31 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#opcode0xxxxx">opcode</a></td>
  <td>9 bit hex value</td>
  <td>Set bits 9-28 in PERF_UNIT_CTL_1_CHA_&lt;0-37&gt; register</td>
  <td>A list of valid opcodes can be found in the <a href="https://software.intel.com/en-us/articles/intel-xeon-phi-x200-family-processor-performance-monitoring-reference-manual">Intel® Xeon® Phi Processor Performance Monitoring Reference Manual</a>.</td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#state0xxxxx">state</a></td>
  <td>10 bit hex value</td>
  <td>Set bits 17-26 in PERF_UNIT_CTL_CHA_&lt;0-37&gt; register</td>
  <td>H: 0x08,<br>E: 0x04,<br>S: 0x02<br>All other bits reserved.</td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#tid0xxxxx">tid</a></td>
  <td>9 bit hex value</td>
  <td>Set bits 0-8 in PERF_UNIT_CTL_CHA_&lt;0-37&gt; register and enables TID filtering with bit 19 in config register</td>
  <td>0-2 ThreadID, 3-8 CoreID</td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#nid0xxxxx">nid</a></td>
  <td>2 bit hex value</td>
  <td>Set bits 0-1 in PERF_UNIT_CTL_1_CHA_&lt;0-37&gt; register</td>
  <td>Remote: 0x1<br>Local: 0x2</td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#match0-30xxxxx">match0</a></td>
  <td>3 bit hex address</td>
  <td>Set bits 29-31 in PERF_UNIT_CTL_1_CHA_&lt;0-37&gt; register</td>
  <td>C6Opcode: 0x1<br>NonCohOpcode: 0x2<br>IsocOpcode: 0x3</td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#match0-30xxxxx">match1</a></td>
  <td>2 bit hex address</td>
  <td>Set bits 4-5 in PERF_UNIT_CTL_1_CHA_&lt;0-37&gt; register</td>
  <td>Count near memory cache events: 0x1<br>Count non-near memory cache events: 0x2</td>
</tr>
</table>

<H3>Special handling for events</H3>
<p>The Intel® Xeon Phi (KNL) microarchitecture provides an event LLC_LOOKUP which can be filtered with the 'state' option. If no 'state' is set, LIKWID sets the state to 0xE, the default value to measure all lookups.<br>
If the match1 option is not used, bits 4 and 5 in PERF_UNIT_CTL_1_CHA_&lt;0-37&gt; are set.<br>
If no opcode option is set, the bit 3 in PERF_UNIT_CTL_1_CHA_&lt;0-37&gt; is set.</p>


\anchor KNL_WBOX
<H2>Power control unit general-purpose counters</H2>
<p>The Intel® Xeon Phi (KNL) microarchitecture provides measurements of the power control unit (PCU) in the uncore.</p>

<p>The PCU performance counters are exposed to the operating system through the MSR interface. The name WBOX originates from the Nehalem EX uncore monitoring.</p>
<H3>Counter and events</H3>
<table>
<tr>
  <th>Counter name</th>
  <th>Event name</th>
</tr>
<tr>
  <td>WBOX0</td>
  <td>*</td>
</tr>
<tr>
  <td>WBOX1</td>
  <td>*</td>
</tr>
<tr>
  <td>WBOX2</td>
  <td>*</td>
</tr>
<tr>
  <td>WBOX3</td>
  <td>*</td>
</tr>
</table>
<H3>Available Options</H3>
<table>
<tr>
  <th>Option</th>
  <th>Argument</th>
  <th>Operation</th>
  <th>Comment</th>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#edgedetect">edgedetect</a></td>
  <td>N</td>
  <td>Set bit 18 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#invert">invert</a></td>
  <td>N</td>
  <td>Set bit 23 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#threshold0xxxxx">threshold</a></td>
  <td>5 bit hex value</td>
  <td>Set bits 24-28 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#match0-30xxxxx">match0</a></td>
  <td>32 bit hex value</td>
  <td>Set bits 0-31 in<br>MSR_UNC_PCU_PMON_BOX_FILTER register</td>
  <td>Band0: bits 0-7,<br>Band1: bits 8-15,<br>Band2: bits 16-23,<br>Band3: bits 24-31</td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#occupancy0xxxxx">occupancy</a></td>
  <td>2 bit hex value</td>
  <td>Set bit 14-15 in config register</td>
  <td>Cores<br>in C0: 0x1,<br>in C3: 0x2,<br>in C6: 0x3</td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#occ_edgedetect">occ_edgedetect</a></td>
  <td>N</td>
  <td>Set bit 31 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#occ_invert">occ_invert</a></td>
  <td>N</td>
  <td>Set bit 30 in config register</td>
  <td></td>
</tr>
</table>

\anchor KNL_MBOX
<H2>Memory controller (iMC) counters</H2>
<p>The Intel&reg; Xeon Phi (KNL) microarchitecture provides measurements of the integrated Memory Controllers (iMC) in the uncore. The description from Intel&reg;:<br>
<i>The processor implements two Memory Controllers on the processor die. Each memory
controller is capable of controlling three DDR4 memory channels. The MC design is
derived from the EDC (Near-Memory (MCDRAM) controller) and is a sub-set of EDC in
functionality. The main difference from EDC is that the physical interface for MC will be
DDR4 IOs. The processor MC will interface with the rest of the Untile via the mesh
interface (R2Mem -&gt; Ring-to-MC interface). Therefore, the MC agent is broken into
three regions: The front-end ring/mesh interface called the "R2Mem", the core "EDC
controller" logic, and three individual "DDR channel controllers/schedulers."
</i><br></p>
<p>The integrated Memory Controllers performance counters are exposed to the operating system through PCI interfaces. There may be two memory controllers in the system. There are four different PCI devices per memory controller, three for each memory channel and one for the controller. Each device has four different general-purpose counters. The three channels of the first memory controller are MBOX0-2, the memory controller itself is MBOX3. The three channels of the second memory controller (if available) are named MBOX4-6 and the corresponding controller MBOX7. The name MBOX originates from the Nehalem EX uncore monitoring.</p>
<H3>Counter and events</H3>
<table>
<tr>
  <th>Counter name</th>
  <th>Event name</th>
</tr>
<tr>
  <td>MBOX&lt;0-2,4-6&gt;C0</td>
  <td>MC_DCLK, MC_CAS*</td>
</tr>
<tr>
  <td>MBOX&lt;0-2,4-6&gt;C1</td>
  <td>MC_DCLK, MC_CAS*</td>
</tr>
<tr>
  <td>MBOX&lt;0-2,4-6&gt;C2</td>
  <td>MC_DCLK, MC_CAS*</td>
</tr>
<tr>
  <td>MBOX&lt;0-2,4-6&gt;C3</td>
  <td>MC_DCLK, MC_CAS*</td>
</tr>
<tr>
  <td>MBOX&lt;3,7&gt;C0</td>
  <td>MC_UCLK</td>
</tr>
<tr>
  <td>MBOX&lt;3,7&gt;C1</td>
  <td>MC_UCLK</td>
</tr>
<tr>
  <td>MBOX&lt;3,7&gt;C2</td>
  <td>MC_UCLK</td>
</tr>
<tr>
  <td>MBOX&lt;3,7&gt;C3</td>
  <td>MC_UCLK</td>
</tr>
<tr>
  <td>MBOX&lt;0-7&gt;FIX</td>
  <td>DRAM_CLOCKTICKS</td>
</tr>
</table>

<H3>Available Options</H3>
<table>
<tr>
  <th>Option</th>
  <th>Argument</th>
  <th>Operation</th>
  <th>Comment</th>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#edgedetect">edgedetect</a></td>
  <td>N</td>
  <td>Set bit 18 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#invert">invert</a></td>
  <td>N</td>
  <td>Set bit 23 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#threshold0xxxxx">threshold</a></td>
  <td>8 bit hex value</td>
  <td>Set bits 24-31 in config register</td>
  <td></td>
</tr>
</table>

\anchor KNL_EBOX
<H2>Embedded DRAM controller (EDC) counters</H2>
<p>The Intel® Xeon Phi (KNL) microarchitecture provides measurements of the Embedded DRAM Controllers (EDC) in the uncore, the interface to the MCDRAM. The description from Intel®:<br>
<i>The EDC is the high bandwidth near-memory controller for the processor. EDC refers
to "Embedded DRAM Controller" (i.e. DRAM that is embedded in the processor
package). The technology that is used to implement the embedded DRAM for the
processor is MCDRAM (Multi-Chip (Stacked) DRAM). Eight channels of MCDRAM are
supported by 8 MCDRAM Controllers (EDC). The EDC's are connected to the other
components (clusters) within the processor by the internal mesh interconnect fabric.
</i><br></p>

<p>The Embedded DRAM Controllers (EDC) performance counters are exposed to the operating system through PCI interfaces. There are eight embedded memory controllers in the system. There are two different PCI devices per memory controller, one for the mesh side (EUBOX<em>C</em>)and one on the DRAM side (EDBOX<em>C</em>). Each device has four different general-purpose counters. </p>
<p>The fixed-purpose counters are exposed to the operating system through PCI interfaces. There are eight embedded memory controllers in the system. There are two different PCI devices per memory controller, one for the mesh side (EUBOX<em>FIX)and one on the DRAM side (EDBOX</em>FIX). </p>
<H3>Counter and events</H3>
<table>
<tr>
  <th>Counter name</th>
  <th>Event name</th>
</tr>
<tr>
  <td>EUBOX&lt;0-7&gt;C0</td>
  <td>EDC_UCLK, EDC_HIT_*, EDC_MISS_*</td>
</tr>
<tr>
  <td>EUBOX&lt;0-7&gt;C1</td>
  <td>EDC_UCLK, EDC_HIT_*, EDC_MISS_*</td>
</tr>
<tr>
  <td>EUBOX&lt;0-7&gt;C2</td>
  <td>EDC_UCLK, EDC_HIT_*, EDC_MISS_*</td>
</tr>
<tr>
  <td>EUBOX&lt;0-7&gt;C3</td>
  <td>EDC_UCLK, EDC_HIT_*, EDC_MISS_*</td>
</tr>
<tr>
  <td>EUBOX&lt;0-7&gt;FIX</td>
  <td>EDC_CLOCKTICKS</td>
</tr>
<tr>
  <td>EDBOX&lt;0-7&gt;C0</td>
  <td>EDC_ECLK, EDC_WPQ_INSERTS, EDC_RPQ_INSERTS</td>
</tr>
<tr>
  <td>EDBOX&lt;0-7&gt;C1</td>
  <td>EDC_ECLK, EDC_WPQ_INSERTS, EDC_RPQ_INSERTS</td>
</tr>
<tr>
  <td>EDBOX&lt;0-7&gt;C2</td>
  <td>EDC_ECLK, EDC_WPQ_INSERTS, EDC_RPQ_INSERTS</td>
</tr>
<tr>
  <td>EDBOX&lt;0-7&gt;C3</td>
  <td>EDC_ECLK, EDC_WPQ_INSERTS, EDC_RPQ_INSERTS</td>
</tr>
<tr>
  <td>EDBOX&lt;0-7&gt;FIX</td>
  <td>MCDRAM_CLOCKTICKS</td>
</tr>
</table>
<H3>Available Options</H3>
<table>
<tr>
  <th>Option</th>
  <th>Argument</th>
  <th>Operation</th>
  <th>Comment</th>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#edgedetect">edgedetect</a></td>
  <td>N</td>
  <td>Set bit 18 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#invert">invert</a></td>
  <td>N</td>
  <td>Set bit 23 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#threshold0xxxxx">threshold</a></td>
  <td>8 bit hex value</td>
  <td>Set bits 24-31 in config register</td>
  <td></td>
</tr>
</table>


\anchor KNL_PBOX
<H2>Ring-to-PCIe counters</H2>
<p>The Intel® Xeon Phi (KNL) microarchitecture provides measurements of the Ring-to-PCIe (R2PCIe) interface in the uncore. The description from Intel®:<br>
<i>The M2PCI is the logic which interfaces the IIO modules to the mesh and includes the mesh stop.</i></p>

<p>The Ring-to-PCIe performance counters are exposed to the operating system through a PCI interface. Independent of the system's configuration, there is only one Ring-to-PCIe interface per CPU socket. </p>
<H3>Counter and events</H3>
<table>
<tr>
  <th>Counter name</th>
  <th>Event name</th>
</tr>
<tr>
  <td>PBOX0</td>
  <td>*</td>
</tr>
<tr>
  <td>PBOX1</td>
  <td>*</td>
</tr>
<tr>
  <td>PBOX2</td>
  <td>*</td>
</tr>
<tr>
  <td>PBOX3</td>
  <td>*</td>
</tr>
</table>
<H3>Available Options</H3>
<table>
<tr>
  <th>Option</th>
  <th>Argument</th>
  <th>Operation</th>
  <th>Comment</th>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#edgedetect">edgedetect</a></td>
  <td>N</td>
  <td>Set bit 18 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#invert">invert</a></td>
  <td>N</td>
  <td>Set bit 23 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#threshold0xxxxx">threshold</a></td>
  <td>8 bit hex value</td>
  <td>Set bits 24-31 in config register</td>
  <td></td>
</tr>
</table>

\anchor KNL_IBOX
<H2>IRP box counters</H2>
<p>The Intel® Xeon Phi (KNL) microarchitecture provides measurements of the IRP box in the uncore. The description from Intel®:<br>
<i>IRP is responsible for maintaining coherency for IIO traffic that needs to be coherent (e.g. cross-socket P2P).
</i></p>

<p>The IRP box counters are exposed to the operating system through the PCI interface. The IBOX was introduced with the Intel® IvyBridge EP/EN/EX microarchitecture.</p>
<H3>Counter and events</H3>
<table>
<tr>
  <th>Counter name</th>
  <th>Event name</th>
</tr>
<tr>
  <td>IBOX0</td>
  <td>*</td>
</tr>
<tr>
  <td>IBOX1</td>
  <td>*</td>
</tr>
</table>
<H3>Available Options</H3>
<table>
<tr>
  <th>Option</th>
  <th>Argument</th>
  <th>Operation</th>
  <th>Comment</th>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#edgedetect">edgedetect</a></td>
  <td>N</td>
  <td>Set bit 18 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#invert">invert</a></td>
  <td>N</td>
  <td>Set bit 23 in config register</td>
  <td></td>
</tr>
<tr>
  <td><a href="https://github.com/RRZE-HPC/likwid/wiki/DescOptions#threshold0xxxxx">threshold</a></td>
  <td>8 bit hex value</td>
  <td>Set bits 24-31 in config register</td>
  <td></td>
</tr>
</table>
*/
