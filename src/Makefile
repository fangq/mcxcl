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
AMDAPPSDKROOT ?=/opt/AMDAPPSDK-3.0
LIBOPENCLDIR ?=$(AMDAPPSDKROOT)/lib/x86_64
LIBOPENCL=-lOpenCL -fopenmp
LINKOPT=-g -L$(LIBOPENCLDIR) $(LIBOPENCL)

CUCCOPT=-I/usr/local/cuda/include #-m32 -msse2 -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -g3 -ffor-scope 
CCOPT=-g -pedantic -Wall -O3 -DMCX_OPENCL -DUSE_OS_TIMER -I/usr/local/cuda/include
CPPOPT:=$(CCOPT) -fopenmp
CCOPT+=-std=c99

OBJSUFFIX=.o
EXESUFFIX=

MAKE       := make
ECHO       := echo
MKDIR      := mkdir

FILES=mcx_host mcx_utils tictoc mcxcl mcx_shapes cjson/cJSON

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

ifdef AMDAPPSDKROOT
  INCLUDEDIRS +=-I$(AMDAPPSDKROOT)/include
endif
  
all logfast:CUCCOPT+=
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

all mt fast log logfast racing mtatomic logatomic mtbox logbox debugmt debuglog det: $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(OBJS)
	$(AR) $(OBJS) $(LINKOPT) -o $(OUTPUT_DIR)/$(BINARY)

%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CCOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cpp
	$(CXX) $(INCLUDEDIRS) $(CPPOPT) -c $(CUCCOPT) -o $@  $<

intelcpu:
	$(IOCC) -cmd=build -input=mcx_core.cl -device=cpu -spir64=mcx_core_intelcpu.bc -bo="-cl-std=CL1.2"

clean:
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX)
