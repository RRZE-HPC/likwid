#!/bin/sh

set -eu

if [ -z ${FROM_MAKEFILE+1} ]; then
    echo "$0 is not intended to be run manually. Please run 'make RPM' instead."
    exit 1
fi

BASEDIR=${PWD}
WORKSPACE=${PWD}/.dpkgbuild
DEBIANDIR=${WORKSPACE}/debian
DEBIANBINDIR=${WORKSPACE}/DEBIAN
mkdir --parents --verbose ${WORKSPACE} ${DEBIANBINDIR}
make PREFIX=${WORKSPACE} INSTALLED_PREFIX=${PREFIX}
#mkdir --parents --verbose ${DEBIANDIR}
CONTROLFILE="${BASEDIR}/packaging/deb/likwid.deb.control"
COMMITISH="HEAD"
VERS=$(git describe --tags --abbrev=0 ${COMMITISH})
VERS=${VERS#v}
VERS=$(echo ${VERS} | sed -e s/-/_/g)
ARCH=$(uname -m)
ARCH=$(echo ${ARCH} | sed -e s/-/_/g)
if [ "${ARCH}" = "x86-64" ]; then
    ARCH=amd64
fi
if [ "${VERS}" = "" ]; then
    VERS="$(VERSION).$(RELEASE).$(MINOR)"
fi
PREFIX="${NAME}-${VERSION}_${ARCH}"
SIZE_BYTES=$(du -bcs --exclude=.dpkgbuild "$WORKSPACE"/ | awk '{print $1}' | head -1 | sed -e 's#^0\+##')
SIZE="$(awk -v size="${SIZE_BYTES}" 'BEGIN {print (size/1024)+1}' | awk '{print int($0)}')"
#sed -e s#"{VERSION}"#"$VERS"#g -e s#"{INSTALLED_SIZE}"#"$SIZE"#g -e s#"{ARCH}"#"$ARCH"#g ${CONTROLFILE} > ${DEBIANDIR}/control
sed -e s#"{VERSION}"#"${VERS}"#g -e s#"{INSTALLED_SIZE}"#"${SIZE}"#g -e s#"{ARCH}"#"${ARCH}"#g ${CONTROLFILE} > ${DEBIANBINDIR}/control
sudo make PREFIX=${WORKSPACE} INSTALLED_PREFIX=${PREFIX} install
DEB_FILE="likwid_${VERS}_${ARCH}.deb"
dpkg-deb -b ${WORKSPACE} "${DEB_FILE}"
sudo rm -r "${WORKSPACE}"
if [ "${GITHUB_ACTIONS}" = "true" ]; then
     echo "::set-output name=DEB::${DEB_FILE}"
fi
