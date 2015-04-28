/*! \page kabini AMD&reg; Kabini

<H1>Available performance monitors for the AMD&reg; Kabini microarchitecture</H1>
<UL>
<LI>\ref KAB_PMC "General-purpose counters"</LI>
<LI>\ref KAB_CPMC "L2 cache general-purpose counters"</LI>
<LI>\ref KAB_UPMC "Northbridge general-purpose counters"</LI>
</UL>


\anchor KAB_PMC
<H2>General-purpose counters</H2>
<P>The AMD&reg; Kabini microarchitecture provides 4 general-purpose counters consiting of a config and a counter register. They are core-local, hence each hardware thread has its own set of general-purpose counters.</P>
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
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD>The value for threshold can range between 0x0 and 0x3</TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
</TABLE>


<H1>Counters available for one hardware thread per shared L2 cache</H1>
\anchor KAB_CPMC
<H2>General-purpose counters</H2>
<P>The AMD&reg; Kabini microarchitecture provides 4 general-purpose counters for measuring L2 cache events. They consist of a config and a counter register. They are tile-local, hence one hardware thread of all sharing a L2 cache can measure the events.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CPMC0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CPMC3</TD>
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
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD>The value for threshold can range between 0x0 and 0x3</TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>tid</TD>
  <TD>4 bit hex value</TD>
  <TD>Set bits 56-59 in config register</TD>
  <TD>If bit equals 0, the events of the thread are counted. See <A HREF="http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2012/10/48751_16h_bkdg.pdf">BIOS and Kernel Developer’s Guide (BKDG) for AMD Family 16h Processors</A> for details.</TD>
</TR>
<TR>
  <TD>nid</TD>
  <TD>4 bit hex value</TD>
  <TD>Set bits 48-51 in config register</TD>
  <TD>If bit equals 0, the events of the thread are counted. See <A HREF="http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2012/10/48751_16h_bkdg.pdf">BIOS and Kernel Developer’s Guide (BKDG) for AMD Family 16h Processors</A> for details.</TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor KAB_UPMC
<H2>Northbridge general-purpose counters</H2>
<P>The AMD&reg; Kabini microarchitecture provides 4 general-purpose counters for the Northbridge consiting of a config and a counter register.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>UPMC0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPMC1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPMC2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UPMC3</TD>
  <TD>*</TD>
</TR>
</TABLE>


*/
