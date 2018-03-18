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

LIBOPENCLDIR:=

DLLFLAG=-fPIC
OMP=-fopenmp
OUTPUTFLAG:=-o

# setup for amd
AMDAPPSDKROOT ?=/opt/AMDAPPSDK-3.0
HAS_AMD := $(shell [ -d $(AMDAPPSDKROOT) ] && echo "1" )
ifeq ($(HAS_AMD), 1)                                                        
  LIBOPENCLDIR=$(AMDAPPSDKROOT)/lib/x86_64
  INCLUDEDIRS +=-I$(AMDAPPSDKROOT)/include
endif

# setup for cuda
CUDA_PATH ?= /usr/local/cuda
HAS_CUDA := $(shell [ -d $(CUDA_PATH) ] && echo "1" )
ifeq ($(HAS_CUDA), 1)                                                        
  CUCCOPT +=-I$(CUDA_PATH)/include #-m32 -msse2 -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -g3 -ffor-scope 
  CCOPT += -I$(CUDA_PATH)/include
  LIBOPENCLDIR=$(CUDA_PATH)/lib64
  INCLUDEDIRS +=-I$(CUDA_PATH)/include
endif

LIBOPENCL=-lOpenCL
LINKOPT=-g -L$(LIBOPENCLDIR) $(LIBOPENCL)

OBJSUFFIX=.o
EXESUFFIX=
CLHEADER=.clh

MAKE       := make
ECHO       := echo
MKDIR      := mkdir

FILES=mcx_host mcx_utils tictoc mcxcl mcx_shapes cjson/cJSON
CLPROGRAM=mcx_core

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)
ifeq ($(findstring Msys,$(PLATFORM)), CYGWIN)
  CCC=nvcc
  LINKOPT=-L/c/CUDA/lib -lcudart --link
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c  -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  MEX=cmd /c mex
  DLLFLAG=
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  INCLUDEDIRS=-I/System/Library/Frameworks/OpenCL.framework/Headers
  LIBOPENCL=-framework OpenCL
endif
  
all static: CUCCOPT+=

mex:        AR=$(MEX)
mex:        LINKOPT+= CXXFLAGS='$$CXXFLAGS -g -DMCX_CONTAINER  $(MEXCCOPT) $(USERCCOPT)' LDFLAGS='-L$$TMW_ROOT$$MATLABROOT/sys/os/$$ARCH $$LDFLAGS  $(USERLINKOPT)' 
mex oct:    OUTPUT_DIR=../mcxlabcl
mex oct:    OUTPUTFLAG:=-output
mex:        BINARY=mcxcl
mex oct:    CUCCOPT+=$(DLLFLAG) -DMCX_CONTAINER
mex oct:    CCOPT+=$(DLLFLAG) -DMCX_CONTAINER
mex oct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER
mex:        LINKOPT+=mcxlabcl.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

oct:        BINARY=mcxcl.mex
oct:        AR= CXXFLAGS='$(MEXCCOPT) $(USERCCOPT)' LFLAGS=' $(USERLINKOPT)' LDFLAGS='$(LFLAGS)' mkoctfile
oct:        LINKOPT+=--mex mcxlabcl.cpp $(INCLUDEDIRS)

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))
CLSOURCE  := $(addsuffix $(CLHEADER), $(CLPROGRAM))

all mex oct static : $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(CLSOURCE) $(OBJS)
	$(AR) $(OBJS) $(LINKOPT) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY)

%$(CLHEADER): %.cl
	xxd -i $(CLPROGRAM).cl | sed 's/\([0-9a-f]\)$$/\0, 0x00/' > $(CLPROGRAM).clh

%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CCOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cpp
	$(CXX) $(INCLUDEDIRS) $(CPPOPT) -c $(CUCCOPT) -o $@  $<

intelcpu:
	$(IOCC) -cmd=build -input=mcx_core.cl -device=cpu -spir64=mcx_core_intelcpu.bc -bo="-cl-std=CL1.2"

static: LINKOPT:=-Wl,-Bstatic -lm -Wl,-Bdynamic -static-libgcc -static-libstdc++ $(LINKOPT) -Wl,-Bstatic -lm -Wl,-Bdynamic

clean:
	-rm -f $(OBJS) $(CLSOURCE) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX)
