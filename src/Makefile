#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#################################################################

#CXX=g++
AR=g++
IOCC=ioc64
MEX=mex
DOXY=doxygen

BINARY=mcxcl
OUTPUT_DIR=../bin
DOCDIR=../doc
INCLUDEDIRS=-Izmat -Izmat/easylzma -I/usr/local/cuda/include -Iubj

CUCCOPT:=
CCOPT=-g -pedantic -Wall -O3 -DMCX_EMBED_CL -DMCX_OPENCL -DUSE_OS_TIMER -DCL_SILENCE_DEPRECATION
CPPOPT:=$(CCOPT) -Wno-variadic-macros
CCOPT+=-std=c99

DLLFLAG=-fPIC
OMP=#-fopenmp
OUTPUTFLAG:=-o
XXD=xxd

ZMATLIB    :=lib/libzmat.a
USERLINKOPT?=$(ZMATLIB)
DOXYCFG=mcx_doxygen.cfg

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

LIBOPENCL?=-lOpenCL

OBJSUFFIX=.o
EXESUFFIX=
CLHEADER=.clh

MAKE       ?= make
ECHO       := echo
MKDIR      := mkdir

MEXLINKLIBS=-L"\$$MATLABROOT/extern/lib/\$$ARCH" -L"\$$MATLABROOT/bin/\$$ARCH" -lmx -lmex

FILES=mcx_host mcx_utils mcx_tictoc mcxcl mcx_shapes mcx_neurojson cjson/cJSON ubj/ubjw
CLPROGRAM=mcx_core

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

ifeq ($(MAKECMDGOALS),mex)
  FILES=mcx_host mcx_utils mcx_tictoc mcx_shapes cjson/cJSON
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),oct)
  FILES=mcx_host mcx_utils mcx_tictoc mcx_shapes cjson/cJSON
  ZMATLIB=
  USERLINKOPT=
endif

ifeq ($(MAKECMDGOALS),static)
  LINKOPT:=-static-libgcc -static-libstdc++ $(LINKOPT) -Wl,-Bstatic -lm
endif

all static: CUCCOPT+=

mex:        AR=$(MEX)
mex:        LINKOPT=CXXFLAGS='$$CXXFLAGS -g -DMCX_CONTAINER -DMATLAB_MEX_FILE $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LINKLIBS="$(MEXLINKLIBS) $(MEXLINKOPT)" COMPFLAGS='' DEFINES='' CXXLIBS='$$CXXLIBS $(MEXLINKOPT) -L$(LIBOPENCLDIR) $(LIBOPENCL)'
mex oct:    OUTPUT_DIR=../mcxlabcl
mex:        OUTPUTFLAG:=-output
mex:        BINARY=mcxcl
mex oct:    CUCCOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
mex oct:    CCOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
mex:        LINKOPT+=mcxlabcl.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

OCT_LDFLAGS := $(shell mkoctfile -p LDFLAGS)

oct:        BINARY=mcxcl.mex
oct:        AR=XTRA_CFLAGS=' ' XTRA_CXXFLAGS=' ' CXXFLAGS='-std=c++11 $(CPPOPT) $(MEXCCOPT) $(USERCCOPT)' LFLAGS='$(LIBOPENCL)' LDFLAGS='$(OCT_LDFLAGS) -g -L$(LIBOPENCLDIR) $(LIBOPENCL)' mkoctfile -v
oct:        LINKOPT=--mex -DMATLAB_MEX_FILE mcxlabcl.cpp $(INCLUDEDIRS) $(MEXLINKOPT)

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))
CLSOURCE  := $(addsuffix $(CLHEADER), $(CLPROGRAM))

all mex oct intelcpu static : $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi
makedocdir:
	@if test ! -d $(DOCDIR); then $(MKDIR) $(DOCDIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(CLSOURCE) $(ZMATLIB) $(OBJS)
	$(AR) $(OBJS) $(LINKOPT) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY) $(USERLINKOPT)

%$(CLHEADER): mcx_core.cl xxd %.cl
	$(XXD) -i $(CLPROGRAM).cl | sed 's/\([0-9a-f]\)$$/\0, 0x00/' > $(CLPROGRAM).clh

%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CCOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cpp
	$(CXX) $(INCLUDEDIRS) $(CPPOPT) -c $(CUCCOPT) -o $@  $<

intelcpu:
	$(IOCC) -cmd=build -input=mcx_core.cl -device=cpu -spir64=mcx_core_intelcpu.bc -bo="-cl-std=CL1.2"

$(ZMATLIB):
	-$(MAKE) -C zmat lib AR=ar CPPOPT="-O3 $(DLLFLAG)" CCOPT="-O3 $(DLLFLAG)" USERLINKOPT=

##  Documentation  ##
doc: makedocdir
	$(DOXY) $(DOXYCFG)

clean:
	-$(MAKE) -C zmat clean
	-rm -f $(OBJS) $(CLSOURCE) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX) $(ZMATLIB)
xxd:
	@if [ -z `which ${XXD}` ]; then \
	    echo "Please first install 'xxd' utility. For Ubuntu/Debian, use 'sudo apt-get install vim-common'; for Windows, please select xxd in cygwin64 installer."; exit 1;\
	fi

# derived the astyle settings from https://github.com/nlohmann/json
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
	   "*.c" "*.h" "*.cpp" "*.cl"
