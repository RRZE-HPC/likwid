#!/bin/bash
set -x

LUA_VERSION=5.4.2
EXT_FOLDER=/home/hpc/unrz/unrz139/Work/likwid-branches/ext

TEMPDIR=/tmp/$$



URL=https://www.lua.org/ftp/lua-${LUA_VERSION}.tar.gz

mkdir -p "${TEMPDIR}"
cd "${TEMPDIR}"
wget ${URL}
tar xf lua-"${LUA_VERSION}".tar.gz

mkdir -p lua"${LUA_VERSION}"/src
mkdir -p lua"${LUA_VERSION}"/includes

cp lua-"${LUA_VERSION}"/src/*.c lua"${LUA_VERSION}"/src
cp lua-"${LUA_VERSION}"/src/*.h lua"${LUA_VERSION}"/includes
rm -f lua"${LUA_VERSION}"/src/luac.c
sed -i -e s+'#define LUA_USE_READLINE'+'//#define LUA_USE_READLINE'+g lua"${LUA_VERSION}"/includes/luaconf.h


cp "${EXT_FOLDER}/lua/Makefile" lua"${LUA_VERSION}"
cp "${EXT_FOLDER}/lua/LICENSE" lua"${LUA_VERSION}"
mv "${EXT_FOLDER}/lua" "${EXT_FOLDER}/lua.old"
cd "${TEMPDIR}"
cp -r lua"${LUA_VERSION}" "${EXT_FOLDER}/lua"

rm -rf lua"${LUA_VERSION}"
rm -rf lua-"${LUA_VERSION}"

