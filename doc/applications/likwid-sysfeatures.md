\page likwid-sysfeatures likwid-sysfeatures

<H1>Information</H1>
<CODE>likwid-sysfeatures</CODE> is a command line application to get and set various
system features, mostly related to the hardware but also the operating system. It
is the successor or \ref likwid-features , \ref likwid-powermeter and
\ref likwid-setFrequencies and will provide the same information.

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
  <TD>-d &lt;arg&gt;</TD>
  <TD>Specify devices for manipulation. Seperate multiple specifications with <CODE>&#64;</CODE>. The same domain specifiers as for the legacy -c/-C options can be used, except that <CODE>C</CODE> specifies are CPU core, not the last level cache..
  <TABLE>
    <TR>
      <TH>Option</TH>
      <TH>Description</TH>
    </TR>
    <TR>
      <TD>0,1,2,4-5</TD>
      <TD>If no domain specifier is given, HW thread IDs are assumed</TD>
    </TR>
    <TR>
      <TD>T:0,1,2,4-5</TD>
      <TD>The domain specifier <CODE>T</CODE> explicitly marks the following ID as HW thread IDs.</TD>
    </TR>
    <TR>
      <TD>T:0,1&#64;S:0</TD>
      <TD>Create devices for the HW threads with IDs 0 and 1 and a separate device for socket 0</TD>
    </TR>
    <TR>
      <TD>GN:0</TD>
      <TD>Create devices for Nvidia GPU 0</TD>
    </TR>
    <TR>
      <TD>GA:0</TD>
      <TD>Create devices for AMD GPU 0</TD>
    </TR>
  </TABLE>
  </TD>
</TR>
<TR>
  <TD>-a, --all</TD>
  <TD>List all provides system features with their device scope, access possibilities and description.</TD>
</TR>
<TR>
  <TD>-l, --list</TD>
  <TD>List the value of all system features for the specified devices (-d)</TD>
</TR>
<TR>
  <TD>-g, --get &lt;feature(s)&gt;</TD>
  <TD>Get the values of the specified features (comma-separated list) for the specified devices (-d)</TD>
</TR>
<TR>
  <TD>-s, --set &lt;feature(s)&gt;</TD>
  <TD>Set the values of the specified features (feature=value, comma-separated list) for the specified devices (-d)</TD>
</TR>
<TR>
  <TD>-O</TD>
  <TD>Print output in CSV format (conform to <A HREF="https://tools.ietf.org/html/rfc4180">RFC 4180</A>). The output contains some markers that help to parse the output.</TD>
</TR>
</TABLE>

