#################################################################
#  Makefile for Monte Carlo eXtreme (MCX) - Dual Backend
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#
#  Build targets:
#    make           - OpenCL only (default, same as original mcxcl)
#    make cuda      - CUDA only (via nvcc, no OpenCL dependency)
#    make trinity   - Both OpenCL + CUDA backends in one binary
#    make mex       - MATLAB mex (OpenCL)
#    make cudamex   - MATLAB mex (CUDA)
#    make trinitymex - MATLAB mex (both backends)
#    make oct       - Octave mex (OpenCL)
#    make cudaoct   - Octave mex (CUDA)
#    make trinityoct - Octave mex (both backends)
#################################################################

#CXX=g++
AR=g++
CUDACC=nvcc
IOCC=ioc64
MEX=mex
DOXY=doxygen

BINARY=mcxcl
OUTPUT_DIR=../bin
DOCDIR=../doc
INCLUDEDIRS=-Izmat -Izmat/easylzma -Iubj

CUCCOPT:=-use_fast_math -Xptxas -O3,-v
CCOPT=-g -pedantic -Wall -O3 -DUSE_OS_TIMER -DCL_SILENCE_DEPRECATION
CPPOPT:=$(CCOPT) -Wno-variadic-macros
CCOPT+=-std=c99

ifdef DEBUG
    CUCCOPT += -DMCX_GPU_DEBUG
endif

DLLFLAG=-fPIC
OMP=#-fopenmp
OUTPUTFLAG:=-o
XXD=xxd

ZMATLIB    :=lib/libzmat.a
USERLINKOPT?=$(ZMATLIB)
DOXYCFG=mcx_doxygen.cfg

#################################################################
# Source file lists
#################################################################

# Common files (shared by all backends)
FILES_COMMON=mcx_utils mcx_tictoc mcxcl mcx_shapes mcx_neurojson mcx_mie cjson/cJSON ubj/ubjw

# OpenCL-specific host code
FILES_OCL=mcx_host

# CUDA-specific host code (compiled by nvcc)
FILES_CUDA=mcx_cu_host

# The OpenCL kernel to be embedded (mcx_gpu_compat.h is already merged into mcx_core.cl)
CLPROGRAM=mcx_core

#################################################################
# Backend selection via make targets
#
# Default: OpenCL only
# cuda:    CUDA only
# trinity: both backends
#################################################################

# Default: OpenCL only
USERCCFLAGS ?=
FILES = $(FILES_COMMON)
CUOBJS =
CLSOURCE =

# --- OpenCL backend (default, and when no goal specified) ---
ifneq (,$(filter $(MAKECMDGOALS),all static mex oct intelcpu))
    USERCCFLAGS += -DUSE_OPENCL -DMCX_OPENCL -DMCX_EMBED_CL
    FILES += $(FILES_OCL)
    CLSOURCE = $(addsuffix $(CLHEADER), $(CLPROGRAM))
endif

ifeq ($(MAKECMDGOALS),)
    USERCCFLAGS += -DUSE_OPENCL -DMCX_OPENCL -DMCX_EMBED_CL
    FILES += $(FILES_OCL)
    CLSOURCE = $(addsuffix $(CLHEADER), $(CLPROGRAM))
endif

# --- CUDA backend (no OpenCL, no .clh needed) ---
ifneq (,$(filter $(MAKECMDGOALS),cuda cudamex cudaoct))
    USERCCFLAGS += -DUSE_CUDA
    FILES += $(FILES_COMMON)
    FILES := $(filter-out mcxcl,$(sort $(FILES)))
    FILES += mcxcl
    CUFILES = $(FILES_CUDA)
    CUOBJS = $(addsuffix $(OBJSUFFIX), $(CUFILES))
    CUCCOPT += -DUSE_CUDA -DUSE_ATOMIC -DMCX_SAVE_DETECTORS -DMCX_DO_REFLECTION -Xcompiler -fPIC
    BINARY = mcx
    EXTRALIB += -lcudart
endif

# --- Trinity: both backends (.clh needed for OpenCL path) ---
ifneq (,$(filter $(MAKECMDGOALS),trinity trinitymex trinityoct))
    USERCCFLAGS += -DUSE_OPENCL -DMCX_OPENCL -DMCX_EMBED_CL -DUSE_CUDA
    FILES += $(FILES_OCL)
    CUFILES = $(FILES_CUDA)
    CUOBJS = $(addsuffix $(OBJSUFFIX), $(CUFILES))
    CUCCOPT += -DUSE_CUDA -DUSE_OPENCL -DUSE_ATOMIC -DMCX_SAVE_DETECTORS -DMCX_DO_REFLECTION -Xcompiler -fPIC
    BINARY = mcx
    EXTRALIB += -lcudart
    CLSOURCE = $(addsuffix $(CLHEADER), $(CLPROGRAM))
endif

#################################################################
# Platform detection and library paths
#################################################################

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)

# setup for amd
AMDAPPSDKROOT ?=/opt/AMDAPPSDK-3.0
HAS_AMD := $(shell [ -d $(AMDAPPSDKROOT) ] && echo "1" )
ifeq ($(HAS_AMD), 1)
  LIBOPENCLDIR=$(AMDAPPSDKROOT)/lib/x86_64
  INCLUDEDIRS +=-I$(AMDAPPSDKROOT)/include
endif

# setup for cuda
CUDA_PATH ?= /usr/local/cuda

ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  CUDA_LIB:="$(shell echo $$CUDA_PATH | sed 's:\\:/:g')"
else ifeq ($(findstring MINGW64,$(PLATFORM)), MINGW64)
  CUDA_LIB:="$(shell echo $$CUDA_PATH | sed 's:\\:/:g')"
else
  CUDA_LIB:="$(shell echo $(CUDA_PATH) | sed 's:\\:/:g')"
endif

HAS_CUDA := $(shell [ -d $(CUDA_LIB) ] && echo "1" )
ifeq ($(HAS_CUDA), 1)
  LIBOPENCLDIR ?= $(CUDA_LIB)/lib64
  INCLUDEDIRS +=-I$(CUDA_LIB)/include
endif

LIBOPENCLDIR ?= /usr/lib/x86_64-linux-gnu

LIBOPENCL?=-lOpenCL

OBJSUFFIX=.o
EXESUFFIX=
CLHEADER=.clh

MAKE       ?= make
ECHO       := echo
MKDIR      := mkdir

MEXLINKLIBS=-L"\$$MATLABROOT/extern/lib/\$$ARCH" -L"\$$MATLABROOT/bin/\$$ARCH" -lmx -lmex

#################################################################
# Platform-specific overrides
#################################################################

ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  EXTRALIB   +=-static
  LINKOPT=-L$(LIBOPENCLDIR) $(LIBOPENCL) $(EXTRALIB)
  INCLUDEDIRS +=-I"./mingw64/include"
  CPPOPT =-D_CRT_SECURE_NO_DEPRECATE -DWIN32
  CCOPT +=-D__USE_MINGW_ANSI_STDIO=1
  OBJSUFFIX=.o
  EXESUFFIX=.exe
  LIBOPENCL   ="c:\Windows\System32\OpenCL.dll"
  MEXCCOPT+=-DMX_COMPAT_32
  MEX        :=cmd /c mex.bat -v -f mexopts_msys2_gcc.xml
  DLLFLAG=
  ZMATLIB    :=zmat/zmatlib.o zmat/miniz/miniz.o zmat/lz4/lz4.o zmat/lz4/lz4hc.o zmat/easylzma/*.o zmat/easylzma/pavlov/*.o
  USERLINKOPT:=$(ZMATLIB)
  MEXLINKLIBS="\$$LINKLIBS"
else ifeq ($(findstring MINGW64,$(PLATFORM)), MINGW64)
  MW_MINGW64_LOC?=/c/msys64/usr
  MEX        :=cmd //c mex.bat -f mexopts_msys2_gcc.xml
  CCOPT +=-D__USE_MINGW_ANSI_STDIO=1
  INCLUDEDIRS+=-I"./mingw64/include"
  LIBOPENCL   ="c:\Windows\System32\OpenCL.dll"
  EXTRALIB   +=-static
  LINKOPT=-L$(LIBOPENCLDIR) $(LIBOPENCL) $(EXTRALIB)
  MEXCCOPT+=-DMX_COMPAT_32
  DLLFLAG    =
  MEXLINKLIBS="\$$LINKLIBS"
else ifeq ($(findstring MSYS,$(PLATFORM)), MSYS)
  MEX        :=cmd //c mex.bat -f mexopts_msys2_gcc.xml
  CCOPT +=-D__USE_MINGW_ANSI_STDIO=1
  INCLUDEDIRS+=-I"./mingw64/include"
  LIBOPENCL   ="c:\Windows\System32\OpenCL.dll"
  EXTRALIB   +=-static
  LINKOPT=-L$(LIBOPENCLDIR) $(LIBOPENCL) $(EXTRALIB)
  MEXCCOPT+=-DMX_COMPAT_32
  DLLFLAG    =
  MEXLINKLIBS="\$$LINKLIBS"
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  INCLUDEDIRS+=-I/System/Library/Frameworks/OpenCL.framework/Headers
  LIBOPENCL=-framework OpenCL
  LIBOPENCLDIR=/System/Library/Frameworks/OpenCL.framework/Versions/A
  AR+=-g -L$(LIBOPENCLDIR) $(LIBOPENCL)
else
  LINKOPT=-g -L$(LIBOPENCLDIR) $(LIBOPENCL)
endif

ifeq ($(findstring x86_64,$(ARCH)), x86_64)
  CCOPT  +=-m64
  CPPOPT +=-m64
endif

#################################################################
# CUDA-only targets: remove OpenCL link dependency
#################################################################

ifneq (,$(filter $(MAKECMDGOALS),cuda cudamex cudaoct))
  LINKOPT=-g -L$(CUDA_LIB)/lib64 -lcudart
  LIBOPENCL=
  EXTRALIB:=$(filter-out -lOpenCL,$(EXTRALIB))
endif

# Trinity needs both
ifneq (,$(filter $(MAKECMDGOALS),trinity trinitymex trinityoct))
  LINKOPT=-g -L$(LIBOPENCLDIR) $(LIBOPENCL) -L$(CUDA_LIB)/lib64 -lcudart
endif

#################################################################
# Mex/Oct file list overrides
#################################################################

ifneq (,$(filter $(MAKECMDGOALS),mex oct cudamex cudaoct trinitymex trinityoct))
  CCOPT+=-fPIC
endif

ifeq ($(MAKECMDGOALS),mex)
  FILES=mcx_host mcx_utils mcx_tictoc mcx_shapes mcx_mie cjson/cJSON
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),oct)
  FILES=mcx_host mcx_utils mcx_tictoc mcx_shapes mcx_mie cjson/cJSON
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),cudamex)
  FILES=mcx_utils mcx_tictoc mcx_shapes mcx_mie cjson/cJSON
  CUFILES=mcx_cu_host
  CUOBJS=$(addsuffix $(OBJSUFFIX), $(CUFILES))
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),cudaoct)
  FILES=mcx_utils mcx_tictoc mcx_shapes mcx_mie cjson/cJSON
  CUFILES=mcx_cu_host
  CUOBJS=$(addsuffix $(OBJSUFFIX), $(CUFILES))
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),trinitymex)
  FILES=mcx_host mcx_utils mcx_tictoc mcx_shapes mcx_mie cjson/cJSON
  CUFILES=mcx_cu_host
  CUOBJS=$(addsuffix $(OBJSUFFIX), $(CUFILES))
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),trinityoct)
  FILES=mcx_host mcx_utils mcx_tictoc mcx_shapes mcx_mie cjson/cJSON
  CUFILES=mcx_cu_host
  CUOBJS=$(addsuffix $(OBJSUFFIX), $(CUFILES))
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),static)
  LINKOPT:=-static-libgcc -static-libstdc++ $(LINKOPT) -Wl,-Bstatic -lm
endif

#################################################################
# Mex/Oct compiler and flag overrides
#################################################################

mex trinitymex:        AR=$(MEX)
mex trinitymex:        LINKOPT=CXXFLAGS='$$CXXFLAGS -g -DMCX_CONTAINER -DMATLAB_MEX_FILE $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LINKLIBS="$(MEXLINKLIBS) $(MEXLINKOPT)" COMPFLAGS='' DEFINES='' CXXLIBS='$$CXXLIBS $(MEXLINKOPT) -L$(LIBOPENCLDIR) $(LIBOPENCL)'
mex oct trinitymex trinityoct:    OUTPUT_DIR=../mcxlabcl
mex trinitymex:        OUTPUTFLAG:=-output
mex trinitymex:        BINARY=mcxcl
mex oct trinitymex trinityoct:    CCOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
mex oct trinitymex trinityoct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
mex oct trinitymex trinityoct:    CUCCOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
mex trinitymex:        LINKOPT+=mcxlabcl.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

cudamex:        AR=$(MEX)
cudamex:        OUTPUT_DIR=../mcxlabcl
cudamex:        OUTPUTFLAG:=-output
cudamex:        BINARY=mcxcl
cudamex:        CCOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
cudamex:        CUCCOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
cudamex:        LINKOPT=CXXFLAGS='$$CXXFLAGS -g -DMCX_CONTAINER -DMATLAB_MEX_FILE $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LINKLIBS="$(MEXLINKLIBS) $(MEXLINKOPT)" COMPFLAGS='' DEFINES='' CXXLIBS='$$CXXLIBS $(MEXLINKOPT) -L$(CUDA_LIB)/lib64 -lcudart'
cudamex:        LINKOPT+=mcxlabcl.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

trinitymex:     LINKOPT+=CXXLIBS='$$CXXLIBS $(MEXLINKOPT) -L$(LIBOPENCLDIR) $(LIBOPENCL) -L$(CUDA_LIB)/lib64 -lcudart'

OCT_LDFLAGS := $(shell mkoctfile -p LDFLAGS 2>/dev/null)

oct:            BINARY=mcxcl.mex
oct:            AR=XTRA_CFLAGS=' ' XTRA_CXXFLAGS=' ' CXXFLAGS='-std=c++11 $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LFLAGS='$(LIBOPENCL)' LDFLAGS='$(OCT_LDFLAGS) -g -L$(LIBOPENCLDIR) $(LIBOPENCL)' mkoctfile -v
oct:            LINKOPT=--mex -DMATLAB_MEX_FILE mcxlabcl.cpp $(INCLUDEDIRS) $(MEXLINKOPT)

cudaoct:        BINARY=mcxcl.mex
cudaoct:        OUTPUT_DIR=../mcxlabcl
cudaoct:        AR=XTRA_CFLAGS=' ' XTRA_CXXFLAGS=' ' CXXFLAGS='-std=c++11 $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LFLAGS='-L$(CUDA_LIB)/lib64 -lcudart' LDFLAGS='$(OCT_LDFLAGS) -g -L$(CUDA_LIB)/lib64 -lcudart' mkoctfile -v
cudaoct:        LINKOPT=--mex -DMATLAB_MEX_FILE mcxlabcl.cpp $(INCLUDEDIRS) $(MEXLINKOPT)

trinityoct:     BINARY=mcxcl.mex
trinityoct:     AR=XTRA_CFLAGS=' ' XTRA_CXXFLAGS=' ' CXXFLAGS='-std=c++11 $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LFLAGS='$(LIBOPENCL) -L$(CUDA_LIB)/lib64 -lcudart' LDFLAGS='$(OCT_LDFLAGS) -g -L$(LIBOPENCLDIR) $(LIBOPENCL) -L$(CUDA_LIB)/lib64 -lcudart' mkoctfile -v
trinityoct:     LINKOPT=--mex -DMATLAB_MEX_FILE mcxlabcl.cpp $(INCLUDEDIRS) $(MEXLINKOPT)

#################################################################
# Object lists
#################################################################

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))
# CLSOURCE is set conditionally in the backend selection block above

#################################################################
# CUDA gen-code (override with CUGENCODE=... on command line)
#################################################################

CUGENCODE ?= -arch=sm_50

#################################################################
# Build targets
#################################################################

all mex oct intelcpu static: $(OUTPUT_DIR)/$(BINARY)

cuda cudamex cudaoct: $(OUTPUT_DIR)/$(BINARY)

trinity trinitymex trinityoct: $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) -p $(OUTPUT_DIR); fi
makedocdir:
	@if test ! -d $(DOCDIR); then $(MKDIR) -p $(DOCDIR); fi

#################################################################
# Embed OpenCL kernel via xxd
#
# For OpenCL and trinity targets, mcx_core.cl (which already
# includes the compat macros inline) is converted to a .clh
# header via xxd for embedding.
#
# For CUDA-only builds, nvcc #includes mcx_core.cl directly.
#################################################################

$(CLPROGRAM)$(CLHEADER): $(CLPROGRAM).cl
	@$(ECHO) "Generating embedded OpenCL kernel header..."
	$(XXD) -i $(CLPROGRAM).cl | sed 's/\([0-9a-f]\)$$/&, 0x00/' > $(CLPROGRAM)$(CLHEADER)

#################################################################
# Compile rules
#################################################################

##  Compile .c files  ##
%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CCOPT) $(USERCCFLAGS) -c -o $@ $<

##  Compile .cpp files (for g++ compiled files)  ##
%$(OBJSUFFIX): %.cpp
	$(CXX) $(INCLUDEDIRS) $(CPPOPT) $(USERCCFLAGS) -c -o $@ $<

##  Compile .cu files with nvcc  ##
%$(OBJSUFFIX): %.cu
	$(CUDACC) $(CUGENCODE) -c $(CUCCOPT) $(INCLUDEDIRS) -o $@ $<

#################################################################
# Link
#################################################################

$(OUTPUT_DIR)/$(BINARY): makedirs $(CLSOURCE) $(ZMATLIB) $(OBJS) $(CUOBJS)
	@$(ECHO) "Linking $@ ..."
	$(AR) $(OBJS) $(CUOBJS) $(LINKOPT) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY) $(USERLINKOPT)

$(ZMATLIB):
	-$(MAKE) -C zmat lib AR=ar CPPOPT="-O3 $(DLLFLAG)" CCOPT="-O3 $(DLLFLAG)" USERLINKOPT=

#################################################################
# Utilities
#################################################################

intelcpu:
	$(IOCC) -cmd=build -input=mcx_core.cl -device=cpu -spir64=mcx_core_intelcpu.bc -bo="-cl-std=CL1.2"

##  Documentation  ##
doc: makedocdir
	$(DOXY) $(DOXYCFG)

clean:
	-$(MAKE) -C zmat clean
	-rm -f $(OBJS) $(addsuffix $(OBJSUFFIX), $(FILES_CUDA) $(FILES_OCL)) $(CLSOURCE)
	-rm -f $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX)
	-rm -f $(OUTPUT_DIR)/mcx$(EXESUFFIX)
	-rm -f $(OUTPUT_DIR)/mcxcl$(EXESUFFIX)
	-rm -f $(ZMATLIB)

xxd:
	@if [ -z `which ${XXD}` ]; then \
	    echo "Please first install 'xxd' utility."; exit 1;\
	fi

pretty:
	astyle \
	    --style=attach \
	    --indent=spaces=4 \
	    --indent-modifiers \
	    --indent-switches \
	    --indent-preproc-block \
	    --indent-preproc-define \
	    --indent-col1-comments \
	    --pad-oper \
	    --pad-header \
	    --align-pointer=type \
	    --align-reference=type \
	    --add-brackets \
	    --convert-tabs \
	    --close-templates \
	    --lineend=linux \
	    --preserve-date \
	    --suffix=none \
	    --formatted \
	    --break-blocks \
	    --exclude=mcx_bench.h \
	   "*.c" "*.h" "*.cpp" "*.cu" "*.cl"

.PHONY: all clean doc makedirs makedocdir xxd pretty \
        cuda trinity static intelcpu \
        mex oct cudamex cudaoct trinitymex trinityoct
