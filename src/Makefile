#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#################################################################

#CXX=g++
AR=g++
IOCC=ioc64

BINARY=mcxcl
OUTPUT_DIR=../bin
INCLUDEDIRS=

CUCCOPT:=
CCOPT=-g -pedantic -Wall -O3 -DMCX_EMBED_CL -DMCX_OPENCL -DUSE_OS_TIMER 
CPPOPT:=$(CCOPT)
CCOPT+=-std=c99

LIBOPENCLDIR:=

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
ifeq ($(findstring Msys,$(PLATFORM)), Msys)
  CCC=nvcc
  LINKOPT=-L/c/CUDA/lib -lcudart --link
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c  -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  INCLUDEDIRS=-I/System/Library/Frameworks/OpenCL.framework/Headers
  LIBOPENCL=-framework OpenCL
endif
  
all static: CUCCOPT+=
mt:         CUCCOPT+=-DUSE_MT_RAND
fast:       CUCCOPT+=-DUSE_MT_RAND
log:        CUCCOPT+=
debugmt:    CUCCOPT+=-DUSE_MT_RAND
debuglog:   CUCCOPT+=
racing:     CUCCOPT+=-DTEST_RACING
mtatomic:   CUCCOPT+=-DUSE_MT_RAND -DUSE_ATOMIC -use_fast_math -arch compute_11
logatomic:  CUCCOPT+=-DUSE_ATOMIC -use_fast_math -arch compute_11
mtbox logbox:		CUCCOPT+=-DCACHE_NON_ATOMIC
debugmt debuglog:	CUCCOPT+=-deviceemu
mtatomic logatomic:	BINARY:=$(BINARY)_atomic

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))
CLSOURCE  := $(addsuffix $(CLHEADER), $(CLPROGRAM))

all mt fast log logfast racing mtatomic logatomic mtbox logbox debugmt debuglog det static : $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(CLSOURCE) $(OBJS)
	$(AR) $(OBJS) $(LINKOPT) -o $(OUTPUT_DIR)/$(BINARY)

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
