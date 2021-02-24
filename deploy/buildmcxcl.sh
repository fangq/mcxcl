#!/bin/bash

###############################################################################
#
#  MCX-CL Nightly-Build Script
#
#  by Qianqian Fang <q.fang at neu.edu>
#
#  Format:
#     ./buildmcxcl.sh <releasetag> <branch>
#                   releasetag defaults to "nightly" if not given
#                   branch defaults to "master" if not given
#
#  Dependency:
#   - To compile mcxcl binary, mcxlabcl for octave
#
#     sudo apt-get install gcc ocl-icd-opencl-dev liboctave-dev vim-common
#
#   - To compile mcxlabcl for MATLAB, one must install MATLAB first, also search
#     and replace R20xx in this script to match your system's MATLAB version
#   - One can also install vendor-specific OpenCL libraries, such as nvidia-opencl-dev
#   - For Windows, first install Cygwin64, and install x86_64-w64-mingw32-gcc/g++
#   
###############################################################################

BUILD='nightly';
if [ ! -z "$1" ]
then
	BUILD=$1
fi
DATE=`date +'%Y%m%d'`
BUILDROOT=~/space/autobuild/$BUILD/mcxcl
OSID=`uname -s`
MACHINE=`uname -m`

if [ "$OSID" == "Linux" ]
then
    OS=linux
    source ~/.bashrc
elif [ "$OSID" == "Darwin" ]; then
    OS=osx
    source ~/.bash_profile
elif [[ "$OSID" == CYGWIN* ]] || [[ "$OSID" == MINGW* ]] || [[ "$OSID" == MSYS* ]]; then
    OS=win
fi

if [ "$BUILD" == "nightly" ]
then
	TAG=${OS}-${MACHINE}-${BUILD}build
else
	TAG=${OS}-${MACHINE}-${BUILD}
fi

SERVER=
REMOTEPATH=

mkdir -p $BUILDROOT
cd $BUILDROOT

rm -rf mcxcl
git clone https://github.com/fangq/mcxcl.git

cat <<EOF >> mcxcl/.git/config
[filter "rcs-keywords"]
        clean  = .git_filters/rcs-keywords.clean
        smudge = .git_filters/rcs-keywords.smudge %f
EOF

cd mcxcl

if [ ! -z "$2" ]
then
      git checkout $2
fi

rm -rf *
git checkout .

rm -rf .git

cd ..
zip -FSr $BUILDROOT/mcxcl-src-${BUILD}.zip mcxcl
if [ "$OS" == "linux" ] && [ ! -z "$SERVER" ]
then
	scp $BUILDROOT/mcxcl-src-${BUILD}.zip $SERVER:$REMOTEPATH/src/
fi

cd mcxcl/src

rm -rf ../mcxlabcl/AUTO_BUILD_*
make clean

if [ "$OS" == "win" ]; then

    make mex &> ../mcxlabcl/AUTO_BUILD_${DATE}.log

    echo "Windows mcxcl build"
    cd ../mcxlabcl
    upx -9 mcxcl.mexw64
    cd ../src
else
    make mex &> ../mcxlabcl/AUTO_BUILD_${DATE}.log
fi

make clean

if [ "$OS" == "win" ]; then
    make oct CXXFLAGS="-m64" LIBOPENCL='/cygdrive/c/Windows/System32/OpenCL.dll' USERLINKOPT='-L"/cygdrive/c/cygwin64/lib/octave/4.4.1" libzmat.a -lz' >>  ../mcxlabcl/AUTO_BUILD_${DATE}.log 2>&1
else
    make oct  >>  ../mcxlabcl/AUTO_BUILD_${DATE}.log 2>&1
fi

if [ -f "../mcxlabcl/mcxcl.mex" ]
then
        echo "Build Successfully" >> ../mcxlabcl/AUTO_BUILD_${DATE}.log
else
        echo "Build Failed" >> ../mcxlabcl/AUTO_BUILD_${DATE}.log
fi

rm -rf ../mcxlabcl/mcxlabcl.o*

if [ "$BUILD" != "nightly" ]
then
	rm -rf ../mcxlabcl/AUTO_BUILD_${DATE}.log
fi

#cp $BUILDROOT/dlls/* ../mcxlabcl
cd ..
zip -FSr $BUILDROOT/mcxlabcl-${TAG}.zip mcxlabcl
cd src
[ ! -z "$SERVER" ] && scp $BUILDROOT/mcxlabcl-${TAG}.zip $SERVER:$REMOTEPATH/${OS}64/


make clean
if [ "$OS" == "osx" ] || [ "$OS" == "win" ]
then
	make &> $BUILDROOT/mcxcl_buildlog_${DATE}.log
else
	make static &> $BUILDROOT/mcxcl_buildlog_${DATE}.log
fi

if [ -f "../bin/mcxcl" ]
then
	echo "Build Successfully" >> $BUILDROOT/mcxcl_buildlog_${DATE}.log
else
	echo "Build Failed" >> $BUILDROOT/mcxcl_buildlog_${DATE}.log
	exit 1;
fi

cp $BUILDROOT/dlls/*.dll ../bin

#upx -9 ../bin/mcxcl

cd ../
rm -rf .git mcxlabcl bin/mcxcl.dSYM src .git_filters .gitattributes deploy
#mv src/bin .
#rm -rf src .git_filters .gitattributes 
#mkdir -p src/bin
#cd src/bin
#if [ "$OS" == "win" ]
#then
#    cmd /c "mklink /h mmc.exe ..\..\bin\mmc.exe"
#else
#    ln -s ../../bin/mmc .
#fi
#cd ../../../../

cd ../
pwd
mv $BUILDROOT/mcxcl_buildlog_${DATE}.log mcxcl/AUTO_BUILD_${DATE}.log

if [ "$BUILD" != "nightly" ]
then
	rm -rf mcxcl/AUTO_BUILD_${DATE}.log
fi

if [ "$OS" == "win" ]
then
   zip -FSr mcxcl-${TAG}.zip mcxcl
else
   zip -FSry mcxcl-${TAG}.zip mcxcl
fi

#mv mcxcl-${TAG}.zip $BUILDROOT

cd $BUILDROOT

[ ! -z "$SERVER" ] && scp mcxcl-${TAG}.zip $SERVER:$REMOTEPATH/${OS}64/

