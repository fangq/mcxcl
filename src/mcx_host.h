/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas,
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504, 2018. https://doi.org/10.1117/1.JBO.23.1.010504
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel
**          based Monte Carlo algorithm for accurate and efficient photon transport
**          modeling in complex bio-tissues," Biomed. Opt. Express, 11(11)
**          pp. 6262-6270. https://doi.org/10.1364/BOE.409468
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_host.h

@brief   Header file for the OpenCL host code
*******************************************************************************/

#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#define CL_TARGET_OPENCL_VERSION 120

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include "mcx_utils.h"
#include "mcx_param.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define ABS(a)  ((a)<0?-(a):(a))

#define MIN(a,b)           ((a)<(b)?(a):(b))

#ifdef USE_LL5_RAND
#define MCX_RNG_NAME       "Logistic-Lattice"
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define RAND_BUF_LEN       5        //register arrays
#else
#define MCX_RNG_NAME       "xoroshiro128+"
#endif

#ifndef CL_MEM_LOCATION_HOST_NV
#define CL_MEM_LOCATION_HOST_NV                     (1 << 0)
typedef cl_bitfield         cl_mem_flags_NV;
#endif

#define RO_MEM             (CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR)
#define WO_MEM             (CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR)
#define RW_MEM             (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR)
#define RW_PTR             (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR)
#define NV_PIN             CL_MEM_LOCATION_HOST_NV

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

void mcx_run_simulation(Config* cfg, float* fluence, float* totalenergy);
cl_platform_id mcx_list_gpu(Config* cfg, unsigned int* activedev, cl_device_id* activedevlist, GPUInfo** info);
void ocl_assess(int cuerr, const char* file, const int linenum);

#ifdef  __cplusplus
}
#endif

#endif
