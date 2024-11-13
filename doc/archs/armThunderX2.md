\page armThunderX2 Marvell&reg;/Cavium&reg; Thunder X2

<P>This page is valid for Marvell&reg;/Cavium&reg; Thunder X2.</P>

<H1>Available performance monitors for the Marvell&reg;/Cavium&reg; Thunder X2 microarchitecture</H1>
<UL>
<LI>\ref TX2_PMC "General-purpose counters"</LI>
<LI>\ref TX2_LLC "Last-level-cache counters"</LI>
<LI>\ref TX2_DMC "DDR4 Memory Controller counters"</LI>
<LI>\ref TX2_CCPI "Coherent Processor Interconnect counters"</LI>
</UL>




\anchor TX2_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Marvell&reg;/Cavium&reg; Thunder X2 microarchitecture provides 6 general-purpose counters consisting of a config and a counter register.</P>
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
<P>Currently no options are available for the general-purpose counters of Marvell&reg;/Cavium&reg; Thunder X2.</P>


<H1>Counters available for one hardware thread per socket</H1>
\anchor TX2_LLC
<H2>Last-level-cache counters</H2>
<P>The Marvell&reg;/Cavium&reg; Thunder X2 Thunder X2 (ARMv8) microarchitecture has a last level cache compared to reference implementations of ARMv8.</P>
<H3>Counter and events</H3>

<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0,1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0,1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0,1&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0,1&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the last-level-cache counters of Marvell&reg;/Cavium&reg; Thunder X2.</P>

\anchor TX2_DMC
<H2>DDR4 Memory Controller counters</H2>
<P>In the Marvell&reg; Thunder X2 (ARMv8) microarchitecture each CPU socket is equipped with two memory controllers. Each controller provides 4 counters.</P>
<H3>Counter and events</H3>

<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX&lt;0,1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0,1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0,1&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>MBOX&lt;0,1&gt;C3</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the DDR4 memory controller of Marvell&reg;/Cavium&reg; Thunder X2.</P>

\anchor TX2_CCPI
<H2>Coherent Processor Interconnect counters</H2>
<P>The Marvell&reg; Thunder X2 (ARMv8) microarchitecture connects CPU sockets with the CCPI interconnect. There are two endpoints per CPU socket, each providing 8 counters.</P>
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
  <TD>SBOX&lt;0,1&gt;C4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;C5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;C6</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX&lt;0,1&gt;C7</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the Coherent Processor Interconnect (CCPI) counters of Marvell&reg;/Cavium&reg; Thunder X2.</P>
