/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2023
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

#ifdef  __cplusplus
extern "C" {
#endif

#define ABS(a)  ((a)<0?-(a):(a))

#define MIN(a,b)           ((a)<(b)?(a):(b))

#ifdef USE_LL5_RAND
typedef float  RandType;
#define MCX_RNG_NAME       "Logistic-Lattice"
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define RAND_BUF_LEN       5        //register arrays
#else
typedef cl_ulong  RandType;
#define MCX_RNG_NAME       "xoroshiro128+"
#define RAND_SEED_LEN      4        //32bit seed length (32*5=160bits)
#define RAND_BUF_LEN       2        //register arrays
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

/**
 * @brief Simulation constant parameters stored in the constant memory
 *
 * This struct stores all constants used in the simulation.
 */

typedef struct KernelParams {
    cl_float4 ps;                   /**< initial position vector, for pencil beam */
    cl_float4 c0;                   /**< initial directon vector, for pencil beam */
    cl_float4 maxidx;               /**< maximum index in x/y/z directions for out-of-bound tests */
    cl_uint4  dimlen;               /**< maximum index used to convert x/y/z to 1D array index */
    cl_uint4  cp0;                  /**< 3D coordinates of one diagonal of the cached region  (obsolete) */
    cl_uint4  cp1;                  /**< 3D coordinates of the other diagonal of the cached region  (obsolete) */
    cl_uint2  cachebox;             /**< stride for cachebox data acess  (obsolete) */
    cl_float  minstep;              /**< minimum step of the 3, always 1 */
    cl_float  twin0;                /**< starting time of the current time gate, unit is s */
    cl_float  twin1;                /**< end time of the current time gate, unit is s  */
    cl_float  tmax;                 /**< maximum time gate length, same as cfg.tend */
    cl_float  oneoverc0;            /**< 1/(speed of light in the vacuum)*/
    cl_uint isrowmajor;
    cl_uint save2pt;                /**< flag if mcx outputs fluence volume */
    cl_uint doreflect;              /**< flag if mcx performs reflection calculations */
    cl_uint dorefint;               /**< flag if mcx perform reflection calculations at internal boundaries */
    cl_uint savedet;                /**< flag if mcx outputs detected photon partial length data */
    cl_float  Rtstep;               /**< reciprocal of the step size */
    cl_float  minenergy;            /**< threshold of weight to trigger Russian roulette */
    cl_float  skipradius2;          /**< square of the radius within which the data is cached (obsolete) */
    cl_float  minaccumtime;         /**< time steps for tMCimg like weight accummulation (obsolete) */
    cl_uint maxdetphoton;           /**< max number of detected photons */
    cl_uint maxmedia;               /**< max number of media labels */
    cl_uint detnum;                 /**< max number of detectors */
    cl_uint idx1dorig;              /**< pre-computed 1D index of the photon at launch for pencil/isotropic beams */
    cl_uint mediaidorig;            /**< pre-computed media index of the photon at launch for pencil/isotropic beams */
    cl_uint blockphoton;
    cl_uint blockextra;
    cl_uint voidtime;               /**< flag if the time-of-flight in the background is counted */
    cl_uint srctype;                /**< type of the source */
    cl_float4 srcparam1;            /**< source parameters set 1 */
    cl_float4 srcparam2;            /**< source parameters set 2 */
    cl_uint   maxvoidstep;          /**< max steps that photon can travel in the background before entering non-zero voxels */
    cl_uint   issaveexit;           /**<1 save the exit position and dir of a detected photon, 0 do not save*/
    cl_uint   issaveseed;           /**< flag if one need to save the detected photon seeds for replay */
    cl_uint   issaveref;            /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    cl_uint   isspecular;           /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    cl_uint   maxgate;              /**< max number of time gates */
    cl_int    seed;                 /**< RNG seed passted from the host */
    cl_uint   outputtype;           /**< Type of output to be accummulated */
    cl_uint   threadphoton;         /**< how many photons to be simulated in a thread */
    cl_int    oddphoton;            /**< how many threads need to simulate 1 more photon above the basic load (threadphoton) */
    cl_uint   debuglevel;           /**< debug flags */
    cl_uint   savedetflag;          /**< detected photon save flags */
    cl_uint   reclen;               /**< length of buffer per detected photon */
    cl_uint   partialdata;          /**< per-medium detected photon data length */
    cl_uint   w0offset;             /**< photon-sharing buffer offset */
    cl_uint   mediaformat;          /**< format of the media buffer */
    cl_uint   maxjumpdebug;         /**< max number of positions to be saved to save photon trajectory when -D M is used */
    cl_uint   gscatter;             /**< how many scattering events after which mus/g can be approximated by mus' */
    cl_uint   is2d;                 /**< is the domain a 2D slice? */
    cl_int    replaydet;            /**< select which detector to replay, 0 for all, -1 save all separately */
    cl_uint   srcnum;               /**< total number of source patterns */
    cl_uint   nphase;               /**< number of samples for inverse-cdf, will be added by 2 to include -1 and 1 on the two ends */
    cl_uint   nphaselen;            /**< even-rounded nphase so that shared memory buffer won't give an error */
    cl_uint   nangle;               /**< number of samples for launch angle inverse-cdf, will be added by 2 to include 0 and 1 on the two ends */
    cl_uint   nanglelen;            /**< even-rounded nangle so that shared memory buffer won't give an error */
    cl_char   bc[8];                /**< boundary conditions */
} MCXParam POST_ALIGN(16);

void mcx_run_simulation(Config* cfg, float* fluence, float* totalenergy);
cl_platform_id mcx_list_gpu(Config* cfg, unsigned int* activedev, cl_device_id* activedevlist, GPUInfo** info);
void ocl_assess(int cuerr, const char* file, const int linenum);

#ifdef  __cplusplus
}
#endif

#endif
