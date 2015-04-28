/*! \page likwid-perfscope <CODE>likwid-perfscope</CODE>

<H1>Information</H1>
<CODE>likwid-perfscope</CODE> is a command line application written in Lua that uses the timeline daemon mode of \ref likwid-perfctr
to create on-the-fly pictures with the current measurements. It uses the <A HREF="https://github.com/dkogan/feedgnuplot">feedGnuplot</A> Perl script to send the current data to  <A HREF="http://www.gnuplot.info/">gnuplot</A>. In order to make it more conveniet for users, preconfigured plots of interesting metrics are embedded into <CODE>likwid-perfscope</CODE>. Since the plot windows are normally closed directly after the execution of the monitored applications, <CODE>likwid-perfscope</CODE> waits until Ctrl+c is pressed.

<H1>Options</H1>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>-h, --help</TD>
  <TD>Print help message.</TD>
</TR>
<TR>
  <TD>-v, --version</TD>
  <TD>Print version information.</TD>
</TR>
<TR>
  <TD>-a</TD>
  <TD>Print available predefined plot configurations for current processor.</TD>
</TR>
<TR>
  <TD>-d, --dump</TD>
  <TD>Print measurements to stdout.</TD>
</TR>
<TR>
  <TD>-c &lt;arg&gt;</TD>
  <TD>Defines the CPUs that should be measured<BR>See \ref CPU_expressions on the \ref likwid-pin page for information about the syntax.</TD>
</TR>
<TR>
  <TD>-C &lt;arg&gt;</TD>
  <TD>Defines the CPUs that should be measured and pin the executable to the CPUs<BR>See \ref CPU_expressions on the \ref likwid-pin page for information about the syntax.</TD>
</TR>
<TR>
  <TD>-f, --freq &lt;time&gt;</TD>
  <TD>Specify the measurement time for each plot. &lt;time&gt; is handled over to \ref likwid-perfctr with the -t option. <BR>Examples for &lt;time&gt; are 1s, 250ms, 500us.</TD>
</TR>
<TR>
  <TD>-g, --group &lt;arg&gt;</TD>
  <TD>Specify a predefined plot with optional changes or an eventset with plot configuration. See \ref plot_configuration for details.</TD>
</TR>
</TABLE>

\anchor plot_configuration
<H1>Plot configurations</H1>
<CODE>likwid-perfscope</CODE> extends the format of the eventset option of \ref likwid-perfctr to make it more conveniet for the users. It accepts either a plot configuration of interesting metrics which are embedded into <CODE>likwid-perfscope</CODE> or a custom eventset suitable for \ref likwid-perfctr extended by the plot configuration. A plot configuration can be set with key=value pairs separated by ':' and has to contain at least a definition of a formula for plotting. If specifyed a predefined plot configuration, the configuration can be changed. Options for the plot configuration:
<TABLE>
<TR>
  <TH>Option
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  </TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>title=&lt;string&gt;<BR>TITLE=&lt;string&gt;</TD>
  <TD>Use &lt;string&gt; as title for the plot. The string has to be quoted if it contains spaces. No ':' are allowed in the string</TD>
</TR>
<TR>
  <TD>xtitle=&lt;string&gt;<BR>XTITLE=&lt;string&gt;</TD>
  <TD>Use &lt;string&gt; as label for the x-axis. The default label is 'Time'. The string has to be quoted if it contains spaces. No ':' are allowed in the string</TD>
</TR>
<TR>
  <TD>ytitle=&lt;string&gt;<BR>YTITLE=&lt;string&gt;</TD>
  <TD>Use &lt;string&gt; as label for the left y-axis. The string has to be quoted if it contains spaces. No ':' are allowed in the string</TD>
</TR>
<TR>
  <TD>&lt;string&gt;=&lt;string&gt;</TD>
  <TD>Use the first &lt;string&gt; as legend entry and the second &lt;string&gt; as input forumla for the plot. The result is printed over the run time. The names of the specified counters can be used as variables in the formula. Additional variables are 'time' for the measurement time and 'inverseClock' for the inverted clock frequency. No spaces are allowed in the formula.</TD>
</TR>
<TR>
  <TD>y2title=&lt;string&gt;<BR>Y2TITLE=&lt;string&gt;<BR>y2title=&lt;id-string&gt;<BR>Y2TITLE=&lt;id-string&gt;</TD>
  <TD>Use &lt;string&gt; as label for the right y-axis. If &lt;id-string&gt; is given, the formula with id is associated with the y2-axis. If used with predefined plot configurations, be aware that the formula 1 is part of the plot configuration. If no id is given, the y2-axis is associated with the last given formula. The string has to be quoted if it contains spaces. No ':' are allowed in the string</TD>
</TR>
</TABLE>

<H1>Examples</H1>
<UL>
<LI><CODE>likwid-perfscope -g L3_BAND -C 0-2 -f 1s ./a.out</CODE><BR>
Pin the executable <CODE>./a.out</CODE> to CPUs 0,1,2 and use the predefined plot configuration <CODE>L3_BAND</CODE> The plot is updated ever second.
</LI>
<LI><CODE>likwid-perfscope -g L3_VOL:TITLE="My Title" -C S0:1 -f 500ms ./a.out</CODE><BR>
Pin the executable <CODE>./a.out</CODE> to CPU 1 on Socket 0 and use the predefined plot configuration <CODE>L3_VOL</CODE> but change the title for the plot to "My Title".
</LI>
<LI><CODE>likwid-perfscope -g INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,FORMULA=FIXC0/FIXC1:YTITLE="CPI" -C 0 -freq 2s ./a.out</CODE><BR>
Pin the executable <CODE>./a.out</CODE> to CPUs 0 and use the custom event set <CODE>INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1</CODE>. The last event set entry specifies custom plot options. The plotted formula is <CODE>FIXC0/FIXC1</CODE> and the plot title is set to 'CPI'.
</LI>
<LI><CODE>likwid-perfscope -g L3_VOL,CPI=FIXC0/FIXC1:Y2TITLE="2-Cycles per Instruction" -C 0 -freq 2s ./a.out</CODE><BR>
Pin the executable <CODE>./a.out</CODE> to CPUs 0 and use the predefined plot configuration  <CODE>L3_VOL</CODE> to measure every 2 seconds. Additionally, a formula <CODE>FIXC0/FIXC1</CODE> with the name <CODE>CPI</CODE> is given. The right y-axis is associated to the given function and labeled with <CODE>Cycles per Instruction</CODE>. The formula ID 2 is not needed in this case as the default behavior is to associate the right y-axis to the last formula given.
</LI>
</UL>

*/
