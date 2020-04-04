#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#################################################################

#CXX=g++
AR=g++
IOCC=ioc64
MEX=mex

BINARY=mcxcl
OUTPUT_DIR=../bin
INCLUDEDIRS=

CUCCOPT:=
CCOPT=-g -pedantic -Wall -O3 -DMCX_EMBED_CL -DMCX_OPENCL -DUSE_OS_TIMER 
CPPOPT:=$(CCOPT) -Wno-variadic-macros
CCOPT+=-std=c99

DLLFLAG=-fPIC
OMP=-fopenmp
OUTPUTFLAG:=-o
XXD=xxd

# setup for amd
AMDAPPSDKROOT ?=/opt/AMDAPPSDK-3.0
HAS_AMD := $(shell [ -d $(AMDAPPSDKROOT) ] && echo "1" )
ifeq ($(HAS_AMD), 1)
  LIBOPENCLDIR=$(AMDAPPSDKROOT)/lib/x86_64
  INCLUDEDIRS +=-I$(AMDAPPSDKROOT)/include
endif

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)

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
  CUCCOPT +=-DUSE_OS_TIMER #-m32 -msse2 -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -g3 -ffor-scope 
  CCOPT +=-DUSE_OS_TIMER
  LIBOPENCLDIR=$(CUDA_LIB)/lib64
  ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
     LIBOPENCLDIR=$(CUDA_LIB)/lib/x64
  endif
  INCLUDEDIRS +=-I$(CUDA_LIB)/include
endif

LIBOPENCLDIR ?= /usr/lib/x86_64-linux-gnu

LIBOPENCL=-lOpenCL

OBJSUFFIX=.o
EXESUFFIX=
CLHEADER=.clh

MAKE       := make
ECHO       := echo
MKDIR      := mkdir

FILES=mcx_host mcx_utils tictoc mcxcl mcx_shapes cjson/cJSON
CLPROGRAM=mcx_core

ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  CCC=nvcc
  LINKOPT=-L$(LIBOPENCLDIR) -lOpenCL
  INCLUDEDIRS +=-I"./mingw64/include"
  CPPOPT =-c  -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  LIBOPENCL   ="c:\Windows\System32\OpenCL.dll"
  MEXCCOPT+=-DMX_COMPAT_32
  EXTRALIB   +=-static
  MEX        :=cmd /c mex.bat -v -f mexopts_msys2_gcc.xml
  DLLFLAG=
else ifeq ($(findstring MINGW64,$(PLATFORM)), MINGW64)
  MW_MINGW64_LOC?=/c/msys64/usr
  MEX        :=cmd //c mex.bat -f mexopts_msys2_gcc.xml
  INCLUDEDIRS+=-I"./mingw64/include"
  LIBOPENCL   ="c:\Windows\System32\OpenCL.dll"
  LINKOPT=-L$(LIBOPENCLDIR) $(LIBOPENCL) $(EXTRALIB)
  MEXCCOPT+=-DMX_COMPAT_32
  EXTRALIB   +=-static
  DLLFLAG    =
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  INCLUDEDIRS=-I/System/Library/Frameworks/OpenCL.framework/Headers
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

all static: CUCCOPT+=

mex:        AR=$(MEX)
mex:        LINKOPT= CXXFLAGS='$$CXXFLAGS -g -DMCX_CONTAINER $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LDFLAGS='$$LDFLAGS -g $(USERLINKOPT)' COMPFLAGS='' DEFINES='' CXXLIBS='$$CXXLIBS -L$(LIBOPENCLDIR) $(LIBOPENCL)'
mex oct:    OUTPUT_DIR=../mcxlabcl
mex:        OUTPUTFLAG:=-output
mex:        BINARY=mcxcl
mex oct:    CUCCOPT+=$(DLLFLAG) -DMCX_CONTAINER
mex oct:    CCOPT+=$(DLLFLAG) -DMCX_CONTAINER
mex:        LINKOPT+=mcxlabcl.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

oct:        BINARY=mcxcl.mex
oct:        AR= CXXFLAGS='$(MEXCCOPT) $(USERCCOPT)' LFLAGS='-g -L$(LIBOPENCLDIR) $(LIBOPENCL) $(USERLINKOPT)' LDFLAGS='$(LFLAGS)' mkoctfile
oct:        LINKOPT=--mex mcxlabcl.cpp $(INCLUDEDIRS)

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))
CLSOURCE  := $(addsuffix $(CLHEADER), $(CLPROGRAM))

all mex oct intelcpu static : $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(CLSOURCE) $(OBJS)
	$(AR) $(OBJS) $(LINKOPT) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY)

%$(CLHEADER): mcx_core.cl xxd %.cl
	$(XXD) -i $(CLPROGRAM).cl | sed 's/\([0-9a-f]\)$$/\0, 0x00/' > $(CLPROGRAM).clh

%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CCOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cpp
	$(CXX) $(INCLUDEDIRS) $(CPPOPT) -c $(CUCCOPT) -o $@  $<

intelcpu:
	$(IOCC) -cmd=build -input=mcx_core.cl -device=cpu -spir64=mcx_core_intelcpu.bc -bo="-cl-std=CL1.2"

static: LINKOPT:=-Wl,-Bstatic  -Wl,-Bdynamic -static-libgcc -static-libstdc++ $(LINKOPT) -Wl,-Bstatic -lm -Wl,-Bdynamic

clean:
	-rm -f $(OBJS) $(CLSOURCE) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX)
xxd:
	@if [ -z `which ${XXD}` ]; then \
	    echo "Please first install 'xxd' utility. For Ubuntu/Debian, use 'sudo apt-get install vim-common'; for Windows, please select xxd in cygwin64 installer."; exit 1;\
	fi
