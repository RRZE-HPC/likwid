\page armGrace Nvidia Grace CPU

<P>This page is valid for Nvidia Grace CPU</P>

<H1>Available performance monitors for the Nvidia Grace CPU</H1>
<UL>
<LI>\ref GRACE_PMC "General-purpose counters"</LI>
</UL>


<H1>Counters available for each hardware thread</H1>
\anchor GRACE_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Nvidia Grace CPU microarchitecture provides 6 general-purpose counters consisting of a config and a counter register.</P>
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
<P>Currently no options are available for the general-purpose counters of the Nvidia Grace CPU</P>


<H1>Counters available for one hardware thread per socket</H1>
\anchor GRACE_SCF
<H2>NVIDIA Scalable Coherency Fabric (SCF)</H2>
<P>The Nvidia Grace CPU microarchitecture one SCF interface per socket which provides 6 general-purpose counters consisting of a config and a counter register and one fixed-purpose counter.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>SCF0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SCF1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SCF2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SCF3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SCF4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SCF5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>SCFFIX</TD>
  <TD>CYCLES</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the NVIDIA Scalable Coherency Fabric (SCF) of the Nvidia Grace CPU</P>


\anchor GRACE_CNV
<H2>NVIDIA (C)NvLink</H2>
<P>The Nvidia Grace CPU microarchitecture one (C)NvLink units per socket which provides 6 general-purpose counters consisting of a config and a counter register and one fixed-purpose counter.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CNV0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CNV1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CNV2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CNV3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CNV4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CNV5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CNVFIX</TD>
  <TD>CYCLES</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the (C)NvLink units of the Nvidia Grace CPU</P>



\anchor GRACE_NV
<H2>NVIDIA NvLink</H2>
<P>The Nvidia Grace CPU microarchitecture up to 2 NvLink units per socket which provides 6 general-purpose counters consisting of a config and a counter register and one fixed-purpose counter.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>NV&lt;0-1&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>NV&lt;0-1&gt;C1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>NV&lt;0-1&gt;C2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>NV&lt;0-1&gt;C3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>NV&lt;0-1&gt;C4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>NV&lt;0-1&gt;C5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>NV&lt;0-1&gt;FIX</TD>
  <TD>CYCLES</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the NvLink units of the Nvidia Grace CPU</P>

\anchor GRACE_PCIE
<H2>PCI Express interface</H2>
<P>The Nvidia Grace CPU microarchitecture one PCI Express interface per socket which provides 6 general-purpose counters consisting of a config and a counter register and one fixed-purpose counter.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PCIE0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCIE1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCIE2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCIE3</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCIE4</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCIE5</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PCIEFIX</TD>
  <TD>CYCLES</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<P>Currently no options are available for the PCI Express interface of the Nvidia Grace CPU</P>
