/*! \page likwid-accessD <CODE>likwid-accessD</CODE>

<H1>Information</H1>

<CODE>likwid-accessD</CODE> is a command line application that opens a UNIX file socket and waits for access
operations from LIKWID tools that require access to the MSR and PCI device
files. The MSR and PCI device files are commonly only accessible for users with root
privileges, therefore <CODE>likwid-accessD</CODE> requires the suid-bit set or a suitable libcap setting.
Depending on the current system architecture, <CODE>likwid-accessD</CODE> permits only access to registers defined for the architecture.

<!--<H1>Security concerns</H1>
The <CODE>likwid-accessD</CODE> is a critical part of LIKWID. The accesses to the MSR and often also PCI devices are restricted to users with root privileges. In order to allow users the access to the MSR/PCI devices, the users have to get temporarily elevated privileges. There are currently two ways of achieving this in the Linux operating system. The convenient method are the suid/guid bits that allow an application to execute with the privileges of the owner (suid) or group (guid). The other method are extended capabilities (libcap) which allows a finer selection of allowed operations. The <CODE>cap_sys_rawio</CODE> capability gives executables the right to do raw input/output like reading from and writing to /dev/mem.<BR>
Both methods should be safe but there are exploits for the MSR devices, general suid applications and the <CODE>cap_sys_rawio</CODE>. We checked all exploits we found and built the access daemon so that it is not vulnerable for the exploits. By restricting the accessible registers and closing all file handles -->

<H1>Build</H1>
The building of <CODE>likwid-accessD</CODE> can be controlled through the <CODE>config.mk</CODE> file. Depending on the variable <CODE>BUILDDAEMON</CODE> the daemon code is built or not. The path to <CODE>likwid-accessD</CODE> is compiled into the LIKWID library, so if you want to use the access daemon from an uncommon path, you have to set the <CODE>ACCESSDAEMON</CODE> variable.

<H1>Setup</H1>
In order to allow <CODE>likwid-accessD</CODE> to run with elevated priviledges, there are three ways
<UL>
<LI>SUID Method:<BR>
<CODE>
root: # chown root:root likwid-accessD<BR>
root: # chmod u+s likwid-accessD<BR>
</CODE>
</LI>
<LI>GUID Method: (PCI devices cannot be accesses with this method but we are working on it)<BR>
<CODE>
root: # groupadd likwid<BR>
root: # chown root:likwid likwid-accessD<BR>
root: # chmod g+s likwid-accessD<BR>
</CODE>
</LI>
<LI>Libcap Method:<BR>
<CODE>
root: # setcap cap_sys_rawio+ep likwid-accessD
</CODE>
</LI>
</UL>
There are Linux distributions where settings the suid permission on <CODE>likwid-accessD</CODE> is not enough. Try also to set the capabilities for <CODE>likwid-accessD</CODE>. 

<H1>Protocol</H1>
Every likwid instance will start its own daemon. This client-server pair will communicate with a socket file in <CODE>/tmp</CODE>  named <CODE>likwid-$PID</CODE>. The daemon only accepts one connection. As soon as the connect is successful the socket file will be deleted.

From there the communication consists of write read pairs issued from the client. The daemon will ensure allowed register ranges relevant for the likwid applications. Other register access will be silently dropped and logged to <CODE>syslog</CODE>.

On shutdown the client will terminate the daemon with a exit message.

The daemon has the following error handling:
<UL>
<LI>To prevent daemons not stopped correctly the daemon has a timeout on startup.</LI>
<LI>If the client prematurely disconnects the daemon terminates.</LI>
<LI>If the client disconnects between a read and write the daemon catches <CODE>SIGPIPE</CODE>  and disconnects.</LI>
</UL>
*/
