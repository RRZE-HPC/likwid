/*! \page likwid-agent <CODE>likwid-agent</CODE>

<H1>Information</H1>
<CODE>likwid-agent</CODE> is a daemon application that uses \ref likwid-perfctr to measure hardware performance counters and write them to various output backends. The basic configuration is in a global configuration file that must be given on commandline. The configuration of the hardware event sets is done with extra files suitable for each architecture. Besides the hardware event configuration, the raw data can be transformed using formulas to interested metrics. In order to output to much data, the data can be further filtered or aggregated. <CODE>likwid-agent</CODE> provides multiple store backends like logfiles, <A HREF="https://oss.oetiker.ch/rrdtool/">RRD</A> (Round Robin Database) or gmetric (<A HREF="http://ganglia.sourceforge.net/">Ganglia Monitoring System</A>).

<H1>Config file</H1>
The global configuration file has the following options:
<TABLE>
<TR>
  <TH>Option
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>GROUPPATH &lt;path&gt;</TD>
  <TD>Path to the group files containing event set and output defintitions. See section <B>Group files</B> for information.</TD>
</TR>
<TR>
  <TD>EVENTSET &lt;group1&gt; &lt;group2&gt; ...</TD>
  <TD>Space separated list of groups (without .txt) that should be monitored.</TD>
</TR>
<TR>
  <TD>DURATION &lt;time&gt;</TD>
  <TD>Measurement duration in seconds for each group.</TD>
</TR>
<TR>
  <TD>LOGPATH &lt;path&gt;</TD>
  <TD>Sets the output logfile path for the measured data. Each monitoring group logs to its own file likwid.&lt;group&gt;.log</TD>
</TR>
<TR>
  <TD>LOGSTYLE &lt;update/log&gt;</TD>
  <TD>Specifies whether new data should be appended to the files (log) or the file should be emptied first (update).<BR> Update is a common option if you read in the data afterwards by some monitoring tool like cacti, nagios, ... Default is log</TD>
</TR>
<TR>
  <TD>GMETRIC &lt;True/False&gt;</TD>
  <TD>Activates the output to gmetric.</TD>
</TR>
<TR>
  <TD>GMETRICPATH &lt;path&gt;</TD>
  <TD>Set path to the gmetric executable.</TD>
</TR>
<TR>
  <TD>GMETRICCONFIG &lt;path&gt;</TD>
  <TD>Activates the output to RRD files (Round Robin Database).</TD>
</TR>
<TR>
  <TD>RRD &lt;True/False&gt;</TD>
  <TD>Sets the output logfile for the measured data.</TD>
</TR>
<TR>
  <TD>RRDPATH &lt;path&gt;</TD>
  <TD>Output path for the RRD files. The files are named according to the group and each output metric is saved as DS with function GAUGE. The RRD is configured with RRA entries to store average, minimum and maximum of 10 minutes for one hour, of 60 min for one day and daily data for one month.</TD>
</TR>
<TR>
  <TD>SYSLOG &lt;True/False&gt;</TD>
  <TD>Activates the output to system log using logger.</TD>
</TR>
<TR>
  <TD>SYSLOGPRIO &lt;prio&gt;</TD>
  <TD>Set the priority for the system log. The default priority is 'local0.notice'.</TD>
</TR>
</TABLE>

<H1>Group files</H1>
The group files are adapted performance group files as used by
.B likwid-perfctr(1).
This makes it easy to uses the predefined and often used performance groups as basis for the monitoring. The folder structure of for the groups is <CODE>&lt;GROUPPATH&gt;/&lt;SHORT_ARCH_NAME&gt;/</CODE> with &lt;SHORT_ARCH_NAME&gt; similar to the ones for the performance groups, like 'sandybridge' or 'haswellEP'.


<TABLE>
<TR>
  <TH>Option
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</TH>
  <TH>Description</TH>
</TR>
<TR>
  <TD>SHORT &lt;string&gt;</TD>
  <TD>A short descriptive information about the group.</TD>
</TR>
<TR>
  <TD>EVENTSET<BR>&lt;counter1&gt; &lt;event1&gt;<BR>&lt;counter2&gt;:&lt;option&gt; &lt;event2&gt;</TD>
  <TD>Defintion of the eventset similar to the performance groups. See performance_groups for details.</TD>
</TR>
<TR>
  <TD>METRICS<BR>&lt;metricname&gt; &lt;formula&gt;<BR>&lt;filter&gt; &lt;metricname&gt; &lt;formula&gt;</TD>
  <TD>Defintion of the output metrics. The syntax follows the METRICS defintion of the performance groups as used by \ref likwid-perfctr . If no function is set at the beginning of the line, .B &lt;formula&gt; is evaluated for every CPU and send to the output backends. The .B &lt;metricname&gt; gets the prefix "T&lt;cpuid&gt; ". To avoid writing to much data to the backends, the data can be reduced by .B &lt;filter&gt;. The possible filter options are MIN, MAX, AVG, SUM, ONCE. The ONCE filter sends only the data from the first CPU to the output backends commonly used for the measurement duration.</TD>
</TR>

</TABLE>

<H1>Notice</H1>
There is currently no predefined init script for <CODE>likwid-agent</CODE>, you have to create it yourself for your distribution.
*/
