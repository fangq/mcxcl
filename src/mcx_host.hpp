#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
  #include <cl.h>
#else
  #include <CL/cl.h>
#endif

#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MCX_RNG_NAME       "Logistic-Lattice"
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define RO_MEM             (CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR)
#define WO_MEM             (CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR)
#define RW_MEM             (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR)
#define RW_PTR             (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR)

#define OCL_ASSERT(x)  ocl_assess((x),__FILE__,__LINE__)

#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV           0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV           0x4001
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV                0x4002
#define CL_DEVICE_WARP_SIZE_NV                          0x4003
#define CL_DEVICE_GPU_OVERLAP_NV                        0x4004
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV                0x4005
#define CL_DEVICE_INTEGRATED_MEMORY_NV                  0x4006

#define CL_DEVICE_BOARD_NAME_AMD                        0x4038
#define CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD             0x4040
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD                   0x4043
#define CL_DEVICE_GFXIP_MAJOR_AMD                       0x404A
#define CL_DEVICE_GFXIP_MINOR_AMD                       0x404B

typedef struct KernelParams {
  cl_float4 ps,c0;
  cl_float4 maxidx;
  cl_uint4  dimlen,cp0,cp1;
  cl_uint2  cachebox;
  cl_float  minstep;
  cl_float  twin0,twin1,tmax;
  cl_float  oneoverc0;
  cl_uint isrowmajor,save2pt,doreflect,dorefint,savedet;
  cl_float  Rtstep;
  cl_float  minenergy;
  cl_float  skipradius2;
  cl_float  minaccumtime;
  cl_uint maxdetphoton;
  cl_uint maxmedia;
  cl_uint detnum;
  cl_uint idx1dorig;
  cl_uint mediaidorig;
  cl_uint threadphoton;
  cl_uint oddphotons;
}MCXParam __attribute__ ((aligned (16)));

void mcx_run_simulation(Config *cfg,float *fluence,float *totalenergy);
cl_platform_id mcx_list_gpu(Config *cfg,unsigned int *activedev,cl_device_id *activedevlist,GPUInfo **info);
void ocl_assess(int cuerr,const char *file,const int linenum);

#ifdef  __cplusplus
}
#endif

#endif
