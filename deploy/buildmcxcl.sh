#!/bin/bash

###############################################################################
#
#  MCX-CL Nightly-Build Script
#
#  by Qianqian Fang <q.fang at neu.edu>
#
#  Format:
#     ./buildmcxcl.sh <releasetag> <branchname>
#                   releasetag defaults to "nightly" if not given
#                   branchname defaults to "master" if not given
#
#  Dependency:
#   - To compile mcxcl binary, mcxlabcl for octave
#
#     sudo apt-get install gcc ocl-icd-opencl-dev liboctave-dev vim-common upx-ucl
#
#   - To compile mcxlabcl for MATLAB, one must install MATLAB first, also search
#     and replace R20xx in this script to match your system's MATLAB version
#   - One can also install vendor-specific OpenCL libraries, such as nvidia-opencl-dev
#   - For Windows, first install Cygwin64, and install x86_64-w64-mingw32-gcc/g++
#     or install MSYS2 with mingw64 based gcc compilers
#
###############################################################################

## setting up environment

BUILD='nightly'
if [ ! -z "$1" ]; then
	BUILD=$1
fi
DATE=$(date +'%Y%m%d')
BUILDROOT=~/space/autobuild/$BUILD/mcxcl
OSID=$(uname -s)
MACHINE=$(uname -m)

if [ "$OSID" == "Linux" ]; then
	OS=linux
	source ~/.bashrc
elif [ "$OSID" == "Darwin" ]; then
	OS=macos
	source ~/.bash_profile
elif [[ $OSID == CYGWIN* ]] || [[ $OSID == MINGW* ]] || [[ $OSID == MSYS* ]]; then
	OS=win
fi

if [ "$BUILD" == "nightly" ]; then
	TAG=${OS}-${MACHINE}-${BUILD}build
else
	TAG=${OS}-${MACHINE}-${BUILD}
fi

## setting up upload server (blank if no need to upload)

SERVER=
REMOTEPATH=

## checking out latest github code

mkdir -p $BUILDROOT
cd $BUILDROOT

rm -rf mcxcl
git clone https://github.com/fangq/mcxcl.git

## automatically update revision/version number

cat <<EOF >>mcxcl/.git/config
[filter "rcs-keywords"]
        clean  = .git_filters/rcs-keywords.clean
        smudge = .git_filters/rcs-keywords.smudge %f
EOF

cd mcxcl

if [ ! -z "$2" ]; then
	git checkout $2
fi

rm -rf *
git checkout .

rm -rf .git

## zip and upload source code package

cd ..
zip -FSr $BUILDROOT/mcxcl-src-${BUILD}.zip mcxcl
if [ "$OS" == "linux" ] && [ ! -z "$SERVER" ]; then
	scp $BUILDROOT/mcxcl-src-${BUILD}.zip $SERVER:$REMOTEPATH/src/
fi

## build matlab mex file

cd mcxcl/src

rm -rf ../mcxlabcl/AUTO_BUILD_*
make clean

if [ "$OS" == "win" ]; then
	OLDPATH="$PATH"
	export PATH="C:\Octave\Octave-8.2.1\mingw64\bin":$PATH
	make mex CC=gcc &>../mcxlabcl/AUTO_BUILD_${DATE}.log
	export PATH="$OLDPATH"
else
	make mex &>../mcxlabcl/AUTO_BUILD_${DATE}.log
fi

## build octave mex file

make clean
if [ "$OS" == "win" ]; then
	OLDPATH="$PATH"
	export PATH="C:\Octave\Octave-8.2.1\mingw64\bin":$PATH
	make oct CC=gcc MEXLINKOPT='"C:\tools\msys64\mingw64\lib\gcc\x86_64-w64-mingw32\10.3.0\libgomp.a" "C:\cygwin64\usr\x86_64-w64-mingw32\sys-root\mingw\lib\libwinpthread.a" -static-libgcc -static-libstdc++' >>../mcxlabcl/AUTO_BUILD_${DATE}.log 2>&1
	export PATH="$OLDPATH"
elif [ "$OS" == "macos" ]; then
	make oct MEXLINKOPT="-L/opt/local/lib/octave/8.3.0/" >>../mcxlabcl/AUTO_BUILD_${DATE}.log 2>&1
else
	make oct >>../mcxlabcl/AUTO_BUILD_${DATE}.log 2>&1
fi

## test mex file dependencies

mexfile=(../mcxlabcl/mcxcl.mex*)

if [ -f "$mexfile" ]; then
	if [ "$OS" == "macos" ]; then
		otool -L ../mcxlabcl/mcxcl.mex* >>../mcxlabcl/AUTO_BUILD_${DATE}.log
	elif [ "$OS" == "win" ]; then
		objdump -p ../mcxlabcl/mcxcl.mex* | grep -H "DLL Name:" >>../mcxlabcl/AUTO_BUILD_${DATE}.log
	else
		ldd ../mcxlabcl/mcxcl.mex* >>../mcxlabcl/AUTO_BUILD_${DATE}.log
	fi
	echo "Build Successfully" >>../mcxlabcl/AUTO_BUILD_${DATE}.log
else
	echo "Build Failed" >>../mcxlabcl/AUTO_BUILD_${DATE}.log
fi

## compress mex files with upx

upx -9 ../mcxlabcl/mcxcl.mex* || true

## zip and upload mex package

rm -rf ../mcxlabcl/mcxlabcl.o*

if [ "$BUILD" != "nightly" ]; then
	rm -rf ../mcxlabcl/AUTO_BUILD_${DATE}.log
fi

cp $BUILDROOT/dlls/* ../mcxlabcl
cd ..
zip -FSr $BUILDROOT/mcxlabcl-${TAG}.zip mcxlabcl
cd src
[ ! -z "$SERVER" ] && scp $BUILDROOT/mcxlabcl-${TAG}.zip $SERVER:$REMOTEPATH/${OS}64/

## compile standalone binary/executable

make clean
if [ "$OS" == "macos" ]; then
	make &>$BUILDROOT/mcxcl_buildlog_${DATE}.log
elif [ "$OS" == "win" ]; then
	make CC=gcc &>$BUILDROOT/mcxcl_buildlog_${DATE}.log
else
	make static &>$BUILDROOT/mcxcl_buildlog_${DATE}.log
fi

## test binary dependencies

if [ -f "../bin/mcxcl" ]; then
	if [ "$OS" == "macos" ]; then
		otool -L ../bin/mcxcl >>$BUILDROOT/mcxcl_buildlog_${DATE}.log
	elif [ "$OS" == "win" ]; then
		objdump -p ../bin/mcxcl.exe | grep "DLL Name:" >>$BUILDROOT/mcxcl_buildlog_${DATE}.log
	else
		ldd ../bin/mcxcl >>$BUILDROOT/mcxcl_buildlog_${DATE}.log
	fi
	echo "Build Successfully" >>$BUILDROOT/mcxcl_buildlog_${DATE}.log
else
	echo "Build Failed" >>$BUILDROOT/mcxcl_buildlog_${DATE}.log
	exit 1
fi

## compress binary with upx

upx -9 ../bin/mcxcl* || true

if [ "$OS" == "macos" ]; then
	cat <<EOF >../MAC_USER_PLEASE_RUN_THIS_FIRST.sh
#/bin/sh
xattr -dr com.apple.quarantine *
EOF
	chmod +x ../MAC_USER_PLEASE_RUN_THIS_FIRST.sh
fi

## zip and upload binary package

#cp $BUILDROOT/dlls/*.dll ../bin

cd ../
rm -rf .git* .travis* mcxlabcl bin/mcxcl.dSYM src .git_filters .gitattributes deploy
cd ../
pwd
mv $BUILDROOT/mcxcl_buildlog_${DATE}.log mcxcl/AUTO_BUILD_${DATE}.log

if [ "$BUILD" != "nightly" ]; then
	rm -rf mcxcl/AUTO_BUILD_${DATE}.log
fi

if [ "$OS" == "win" ]; then
	zip -FSr mcxcl-${TAG}.zip mcxcl
else
	zip -FSry mcxcl-${TAG}.zip mcxcl
fi

#mv mcxcl-${TAG}.zip $BUILDROOT

cd $BUILDROOT

[ ! -z "$SERVER" ] && scp mcxcl-${TAG}.zip $SERVER:$REMOTEPATH/${OS}64/
