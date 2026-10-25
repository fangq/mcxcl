#!/bin/bash
# Builds the Linux release binaries of mcxcl inside the manylinux2014 builder:
#
#   docker build --build-arg WITH_CUDA=0 -t mcx-linux-release-builder .github/linux_release_builder
#   docker run --rm -v $PWD:/src \
#       [-v $MATLAB_ROOT:$MATLAB_ROOT:ro -e MATLAB_ROOT=$MATLAB_ROOT] \
#       [-v /usr/include/octave-X.Y.Z:/octinc:ro -e OCTAVE_INCDIR=/octinc] \
#       mcx-linux-release-builder bash .github/linux_release_builder/build_mcxcl.sh
#
# MATLAB is the runner's (matlab-actions/setup-matlab), mounted read-only at
# the same path; the Octave headers are the runner's too.  Either may be left
# out to skip that mex file.  Every artifact is checked at the end: none may
# need a glibc symbol newer than 2.17, the oldest the image has, or link the
# GCC runtime libraries dynamically, or the build fails.

set -e
cd /src

# git refuses a checkout owned by another user, which a mounted one is
git config --global --add safe.directory '*' 2>/dev/null || true

export PATH=/src/.github/linux_release_builder:$PATH   # the mkoctfile stand-in
chmod +x /src/.github/linux_release_builder/mkoctfile

# the check at the end must only ever see what this run built
rm -f bin/mcxcl mcxlabcl/mcxcl.mex mcxlabcl/mcxcl.mexa64

cd src

if [ -n "$OCTAVE_INCDIR" ]; then
    echo "== mcxlabcl for Octave (headers: $OCTAVE_INCDIR)"
    make clean
    make oct MEXLINKOPT="-static-libgcc -static-libstdc++"
fi

if [ -n "$MATLAB_ROOT" ]; then
    echo "== mcxlabcl for MATLAB ($MATLAB_ROOT)"
    make clean
    make mex MEX="$MATLAB_ROOT/bin/mex" MEXLINKOPT="-static-libgcc -static-libstdc++"
    strip ../mcxlabcl/mcxcl.mex*
fi

echo "== mcxcl"
make clean
# "static" links libstdc++, libgcc and libm statically, OpenCL dynamically; zmat's
# threads need libpthread, which glibc 2.17 still keeps apart from libc
make static USERLINKOPT="lib/libzmat.a -Wl,-Bdynamic -lpthread"

cd /src
bash .github/linux_release_builder/check_glibc.sh bin/mcxcl mcxlabcl/mcxcl.mex mcxlabcl/mcxcl.mexa64
