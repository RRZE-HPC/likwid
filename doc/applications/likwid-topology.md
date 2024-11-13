\page likwid-topology likwid-topology

<H1>Information</H1>
<CODE>likwid-topology</CODE> is a command line application to print the thread and cache topology on multicore x86 processors. Used with mono spaced fonts it can
draw the processor topology of a machine in ASCII art. Beyond topology <CODE>likwid-topology</CODE> determines the nominal clock of a processor and prints detailed informations about the caches hierarchy.<BR>

<H1>Options</H1>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>-h, --help</TD>
  <TD>Print help message</TD>
</TR>
<TR>
  <TD>-v, --version</TD>
  <TD>Print version information</TD>
</TR>
<TR>
  <TD>-V, --verbose &lt;level&gt;</TD>
  <TD>Verbose output during execution for debugging. Possible values for &lt;level&gt;:
  <TABLE>
    <TR>
      <TD>0</TD>
      <TD>Output only errors</TD>
    </TR>
    <TR>
      <TD>1</TD>
      <TD>Output some information</TD>
    </TR>
    <TR>
      <TD>2</TD>
      <TD>Output detailed information</TD>
    </TR>
    <TR>
      <TD>3</TD>
      <TD>Output developer information</TD>
    </TR>
  </TABLE>
  </TD>
</TR>
<TR>
  <TD>-c, --caches</TD>
  <TD>Print detailed information about all cache levels</TD>
</TR>
<TR>
  <TD>-C, --clock</TD>
  <TD>Measure the nominal clock frequency and print it</TD>
</TR>
<TR>
  <TD>-g</TD>
  <TD>ASCII art output of the system's topology</TD>
</TR>
<TR>
  <TD>-O</TD>
  <TD>Print output in CSV format (conform to <A HREF="https://tools.ietf.org/html/rfc4180">RFC 4180</A>).</TD>
</TR>
<TR>
  <TD>-o, --output &lt;file&gt;</TD>
  <TD>Write the output to file &lt;file&gt; instead of stdout. According to the used filename suffix, LIKWID tries to reformat the output to the specified format.<BR>By now, LIKWID ships with one filter script <CODE>xml</CODE> written in Perl and a Perl template for developing own output scripts. If the suffix is <CODE>.csv</CODE>, the internal CSV printer is used for file output.<BR>If <CODE>\%h</CODE> is in the filename, it is replaced by the host name.</TD>
</TR>
</TABLE>


