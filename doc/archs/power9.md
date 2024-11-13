\page power9 IBM&reg; POWER9

<H1>Available performance monitors for the IBM&reg; POWER9 microarchitecture</H1>
<UL>
<LI>\ref PWR9_PMC "General-purpose counters"</LI>
<LI>\ref PWR9_MEMORY_SOCKET "Memory controller general-purpose counters"</LI>
<LI>\ref PWR9_XLINK_SOCKET "Xlink interface counters"</LI>
<LI>\ref PWR9_POWERBUS_SOCKET "PowerBus interface counters"</LI>
</UL>



\anchor PWR9_PMC
<H2>General-purpose counters</H2>
<P>The IBM&reg; POWER9 microarchitecture provides 4 general-purpose counters consisting of a config and a counter register. Additionally, there are two registers marked as general-purpose but they are fixed to specific events (like the fixed-purpose counters of Intel&reg; systems). Be aware that IBM names the counters PMC1-PMC6 while LIKWID uses PMC0-PMC5. Events like PM_PMCx_OVERFLOW are adapted accordingly.</P>
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
  <TD>PM_RUN_INST_CMPL</TD>
</TR>
<TR>
  <TD>PMC5</TD>
  <TD>PM_RUN_CYC</TD>
</TR>
</TABLE>
<H3>Available Options</H3>
<P>No options<P>

<H1>Counters available for one hardware thread per socket</H1>


\anchor PWR9_MEMORY_SOCKET
<H2>Memory controller general-purpose counters</H2>
<P>The IBM&reg; POWER9 microarchitecture provides 4 counters per memory channel.</P>

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
</TABLE>

<H3>Available Options</H3>
<P>No options<P>


\anchor PWR9_XLINK_SOCKET
<H2>Xlink interface counters</H2>
<P>The IBM&reg; POWER9 microarchitecture provides 32 counters for all three Xlink interfaces. LIKWID reduced it to 3 per interface. The names QBOX originates from Intel&reg;'s QPI.</P>

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
</TABLE>

<H3>Available Options</H3>
<P>No options<P>


\anchor PWR9_POWERBUS_SOCKET
<H2>PowerBus interface counters</H2>
<P>The IBM&reg; POWER9 microarchitecture provides 32 counters for the PowerBus (SMP Interconnect). LIKWID reduces it to 3 counters. The names SBOX originates from Intel&reg; architectures.</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>SBOX0C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX0C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SBOX0C2</TD>
  <TD>*</TD>
</TR>

</TABLE>

<H3>Available Options</H3>
<P>No options provided for IBM&reg; POWER9 Xlink interface counters.<P>


\anchor PWR9_MCS_SOCKET
<H2>Memory controller synchronous counters</H2>
<P>The IBM&reg; POWER9 microarchitecture provides some counters for the Memory controller synchronous. LIKWID reduces it to 3 counters per port pair (0+1 and 2+3).</P>

<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>BBOX&lt;0-1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX&lt;0-1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>BBOX&lt;0-1&gt;C2</TD>
  <TD>*</TD>
</TR>

</TABLE>

<H3>Available Options</H3>
<P>No options provided for IBM&reg; POWER9 Memory controller synchronous counters.<P>
