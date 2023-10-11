# Maintainer: Holger Obermaier <holgerob[at]gmx[dot]de>
# Contributor: Steven Vancoillie <steven<dot>vancoillie[at]gmail[dot]com>
pkgname=likwid
pkgver=5.2.1
pkgrel=7
pkgdesc="Lightweight performance tools"
url="https://hpc.fau.de/research/tools/likwid/"
arch=('x86_64' 'i686')
license=('GPL3')
depends=()
makedepends=()
optdepends=('perl: for likwid-mpirun and likwid-perfscope'
            'openmpi: for likwid-mpirun'
            'gnuplot: for likwid-perfscope')
conflicts=()
source=("${pkgname}-${pkgver}.tar.gz::https://github.com/RRZE-HPC/likwid/archive/v${pkgver}.tar.gz")
sha256sums=('1b8e668da117f24302a344596336eca2c69d2bc2f49fa228ca41ea0688f6cbc2')

_prefix="/usr"
_mandir="${_prefix}/share/man"
_bindir="${_prefix}/bin"
_sbindir="${_prefix}/bin" # /usr/sbin links to /usr/bin
_libdir="${_prefix}/lib"
_accessdeamon="${_sbindir}/likwid-accessD"
_freqdaemon="${_sbindir}/likwid-setFreq"

build() {
  cd "${srcdir}/${pkgname}-${pkgver}"
  # Replace hard coded sbin in Makefile
  sed -i 's@/sbin@/bin@g' Makefile
  make -j "$(nproc)" \
    PREFIX="${_prefix}" \
    MANPREFIX="${_mandir}" \
    BINPREFIX="${_bindir}" \
    LIBPREFIX="${_libdir}" \
    ACCESSDAEMON="${_accessdeamon}" \
    FREQDAEMON="${_freqdaemon}" \
    INSTALLED_PREFIX="${_prefix}" \
    INSTALLED_BINPREFIX="${_bindir}" \
    INSTALLED_LIBPREFIX="${_libdir}" \
    INSTALLED_ACCESSDAEMON="${_accessdeamon}" \
    INSTALLED_FREQDAEMON="${_freqdaemon}"
}

package() {
  cd "${srcdir}/${pkgname}-${pkgver}"
  make install \
    PREFIX="${pkgdir}/${_prefix}" \
    MANPREFIX="${pkgdir}/${_mandir}" \
    BINPREFIX="${pkgdir}/${_bindir}" \
    LIBPREFIX="${pkgdir}/${_libdir}" \
    ACCESSDAEMON="${pkgdir}/${_accessdeamon}" \
    FREQDAEMON="${pkgdir}/${_freqdaemon}" \
    INSTALLED_PREFIX="${_prefix}" \
    INSTALLED_BINPREFIX="${_bindir}" \
    INSTALLED_LIBPREFIX="${_libdir}" \
    INSTALLED_ACCESSDAEMON="${_accessdeamon}" \
    INSTALLED_FREQDAEMON="${_freqdaemon}"
}
