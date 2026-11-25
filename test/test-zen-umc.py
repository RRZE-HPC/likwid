#!/usr/bin/env python3
"""Hardware regression: Zen4/5 perf_event UMC filters and socket mapping.

Run with a perf_event build: python3 test/test-zen-umc.py /path/to/likwid-perfctr
Requires readable amd_umc_* PMUs and perf_event_paranoid <= 0.
Checks the actual perf_event_open arguments against the kernel PMU topology.
"""
import collections
import pathlib
import re
import subprocess
import sys

pmus = collections.defaultdict(list)
for path in pathlib.Path('/sys/bus/event_source/devices').glob('amd_umc_*'):
    cpu = int(re.split(r'[,\-]', (path / 'cpumask').read_text().strip())[0])
    socket = int(pathlib.Path(
        f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id'
    ).read_text())
    pmus[socket].append((int(path.name.rsplit('_', 1)[1]), cpu,
                         int((path / 'type').read_text())))
if not pmus:
    sys.exit('SKIP: no AMD UMC perf PMUs exposed by the kernel')

# Test every channel, including the final channel on each socket.
expected = collections.Counter()
events = []
cpus = []
for socket, units in sorted(pmus.items()):
    units.sort()
    cpus.append(str(units[0][1]))
    for channel, (_, cpu, pmu_type) in enumerate(units):
        for counter, config in enumerate((0x10a, 0x20a)):
            expected[(cpu, pmu_type, config)] += 1
            if socket == min(pmus):
                events.append(f'CAS_CMD_{("RD", "WR")[counter]}:UMC{channel}C{counter}')

result = subprocess.run([
    sys.argv[1] if len(sys.argv) > 1 else 'likwid-perfctr',
    '-V', '3', '-c', ','.join(cpus), '-g', ','.join(events), 'sleep', '0.1'
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
actual = collections.Counter(
    (int(cpu), int(pmu), int(config, 16))
    for cpu, pmu, config in re.findall(
        r'perf_event_open: cpu_id=(\d+).*? type=(\d+) config=0x([0-9a-fA-F]+)',
        result.stdout)
)
errors = re.findall(r'^.*(?:ERROR|failed:).*$', result.stdout, re.MULTILINE)
if result.returncode or errors or actual != expected:
    print('FAIL: UMC read/write filters or socket PMU mapping')
    print('Missing calls:', expected - actual)
    print('Unexpected calls:', actual - expected)
    print('\n'.join(errors))
    sys.exit(1)
print(f'PASS: {sum(expected.values())} UMC events across {len(pmus)} sockets; '
      'read=0x10a, write=0x20a, correct kernel PMU types')
