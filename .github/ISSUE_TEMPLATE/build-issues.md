---
name: Build issues
about: In case you have problems building LIKWID
title: "[Build]"
labels: ''
assignees: ''

---

**Which version do you want to build?**
Please name version and source of the tarball/package.

**Which architecture do you want to build for?**
- [ ] x86_64
- [ ] ARMv8
- [ ] POWER9
- [ ] Nvidia GPUs
- [ ] x86 (32 bit)
- [ ] ARMv7
- [ ] POWER8

**To Reproduce**
- Changes to `config.mk` (you may leave out `PREFIX` changes)
- Changes to build configurations `make/include_<COMPILER>.mk`
- Any other changes

**Output log**
Error output of `make distclean && make`
