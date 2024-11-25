#!/bin/sh

set -eu

if [ -z ${FROM_MAKEFILE+1} ]; then
    echo "$0 is not intended to be run manually. Please run 'make RPM' instead."
    exit 1
fi

# pwd matters, so only call from Makefile directory
SPECFILE="packaging/rpm/likwid.spec"
# Setup RPM build tree
eval $(rpm --eval "ARCH='%{_arch}' RPMDIR='%{_rpmdir}' SOURCEDIR='%{_sourcedir}' SPECDIR='%{_specdir}' SRPMDIR='%{_srcrpmdir}' BUILDDIR='%{_builddir}'")
mkdir --parents --verbose "${RPMDIR}" "${SOURCEDIR}" "${SPECDIR}" "${SRPMDIR}" "${BUILDDIR}"
# Create source tarball
COMMITISH="HEAD"
VERS=$(git describe --tags --abbrev=0 ${COMMITISH})
VERS=${VERS#v}
VERS=$(echo ${VERS} | sed -e s/-/_/g)
if [ "${VERS}" = "" ]; then
    VERS="$(VERSION).$(RELEASE).$(MINOR)"
fi
eval $(rpmspec --query --queryformat "NAME='%{name}' VERSION='%{version}' RELEASE='%{release}' NVR='%{NVR}' NVRA='%{NVRA}'" --define="VERS ${VERS}" "${SPECFILE}")
PREFIX="${NAME}-${VERSION}"
FORMAT="tar.gz"
SRCFILE="${SOURCEDIR}/${PREFIX}.${FORMAT}"
git archive --verbose --format "${FORMAT}" --prefix="${PREFIX}/" --output="${SRCFILE}" ${COMMITISH}
# Build RPM and SRPM
rpmbuild -ba --define="VERS ${VERS}" --rmsource --clean "${SPECFILE}"
# Report RPMs and SRPMs when in GitHub Workflow
if [[ "${GITHUB_ACTIONS}" == true ]]; then
     RPMFILE="${RPMDIR}/${ARCH}/${NVRA}.rpm"
     SRPMFILE="${SRPMDIR}/${NVR}.src.rpm"
     echo "RPM: ${RPMFILE}"
     echo "SRPM: ${SRPMFILE}"
     echo "::set-output name=SRPM::${SRPMFILE}"
     echo "::set-output name=RPM::${RPMFILE}"
fi
