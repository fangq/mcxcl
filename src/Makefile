#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
#  2009/04/02
#################################################################

CUDACC=g++
CCC=g++
BINARY=mcxcl
OUTPUT_DIR=../bin
INCLUDEDIRS=#-I/home/fangq/Download/ati-stream-sdk-v2.0-lnx32/include
#LINKOPT=-g -m32 -lpthread -ldl -L/usr/X11R6/lib -lOpenCL -lSDKUtil -L/home/fangq/Download/ati-stream-sdk-v2.0-lnx32/lib/x86
LINKOPT=-g -lOpenCL

CUCCOPT=#-m32 -msse2 -Wfloat-equal -Wpointer-arith  -DATI_OS_LINUX -g3 -ffor-scope 
CPPOPT=-g -Wall -DMCX_OPENCL -DUSE_OS_TIMER

OBJSUFFIX=.o
EXESUFFIX=

FILES=mcx_host mcx_utils tictoc mcextreme

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -o)
ifeq ($(findstring Msys,$(PLATFORM)), Msys)
  CCC=nvcc
  LINKOPT=-L/c/CUDA/lib -lcudart --link
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c  -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
else
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        #LINKOPT=-L/usr/local/cuda/lib64 -lcudart 
        #LINKOPT=-g -lOpenCL
     endif
  endif
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

all mt fast log logfast racing mtatomic logatomic mtbox logbox debugmt debuglog: $(OUTPUT_DIR)/$(BINARY)

$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(CCC) $(LINKOPT) $(OBJS) -o $(OUTPUT_DIR)/$(BINARY)

%$(OBJSUFFIX): %.c
	$(CCC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cpp
	$(CUDACC) $(INCLUDEDIRS) $(CPPOPT) -c $(CUCCOPT) -o $@  $<

clean:
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX)
