\page likwid-bridge likwid-bridge

<H1>Information</H1>
<CODE>likwid-bridge</CODE> is proxy for using LIKWID inside of containers. For
LIKWID installations inside containers with <CODE>ACCESSMODE=accessdaemon</CODE>,
the access daemon within the containers run with user-priviledges (despite
being owned by root inside the container). In order to provide also measurement
functionality inside containers, the <CODE>likwid-bridge</CODE> starts a
daemon outside of the container and forwards all requests of LIKWID inside the
container to this outside daemon.

<H1>Usage</H1>

<P>In order to use LIKWID inside the container, you have to wrap the container
startup with <CODE>likwid-bridge</CODE>:</P>

Apptainer/Singularity:


<UL>
<LI>Apptainer/Singularity:<BR>
<CODE>likwid-bridge apptainer run &lt;image&gt;</CODE>
</LI>
<LI>Docker<BR>
<CODE>likwid-bridge docker run &lt;image&gt;</CODE>
</LI>
</UL>
