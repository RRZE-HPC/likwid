/*! \page likwid-setFreq <CODE>likwid-setFreq</CODE>

<H1>Information</H1>
<CODE>likwid-setFreq</CODE> is a command line application that mediates the actual setting of CPU cores' frequency and governor for \ref likwid-setFrequencies. Since only users with root priviledges are allowed to change the frequency of CPU cores, <CODE>likwid-setFreq</CODE> needs to be suid-root.

<H1>Setup</H1>
Setting the suid-root bit:<BR>
<CODE>
root: # chown root:root likwid-setFreq<BR>
root: # chmod u+s likwid-setFreq
</CODE>

*/
