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

#define MCX_DEBUG_RNG       1                   /**< MCX debug flags */
#define MCX_DEBUG_MOVE      2
#define MCX_DEBUG_PROGRESS  4

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

typedef struct KernelParams {
    cl_float4 ps, c0;
    cl_float4 maxidx;
    cl_uint4  dimlen, cp0, cp1;
    cl_uint2  cachebox;
    cl_float  minstep;
    cl_float  twin0, twin1, tmax;
    cl_float  oneoverc0;
    cl_uint isrowmajor, save2pt, doreflect, dorefint, savedet;
    cl_float  Rtstep;
    cl_float  minenergy;
    cl_float  skipradius2;
    cl_float  minaccumtime;
    cl_uint maxdetphoton;
    cl_uint maxmedia;
    cl_uint detnum;
    cl_uint idx1dorig;
    cl_uint mediaidorig;
    cl_uint blockphoton;
    cl_uint blockextra;
    cl_uint voidtime;
    cl_uint srctype;                    /**< type of the source */
    cl_float4 srcparam1;                  /**< source parameters set 1 */
    cl_float4 srcparam2;                  /**< source parameters set 2 */
    cl_uint   maxvoidstep;
    cl_uint   issaveexit;    /**<1 save the exit position and dir of a detected photon, 0 do not save*/
    cl_uint   issaveseed;           /**< flag if one need to save the detected photon seeds for replay */
    cl_uint   issaveref;     /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    cl_uint   isspecular;     /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    cl_uint   maxgate;
    cl_int    seed;                          /**< RNG seed passted from the host */
    cl_uint   outputtype;           /**< Type of output to be accummulated */
    cl_uint   threadphoton;                  /**< how many photons to be simulated in a thread */
    cl_int    oddphoton;                    /**< how many threads need to simulate 1 more photon above the basic load (threadphoton) */
    cl_uint   debuglevel;           /**< debug flags */
    cl_uint   savedetflag;          /**< detected photon save flags */
    cl_uint   reclen;               /**< length of buffer per detected photon */
    cl_uint   partialdata;          /**< per-medium detected photon data length */
    cl_uint   w0offset;             /**< photon-sharing buffer offset */
    cl_uint   mediaformat;          /**< format of the media buffer */
    cl_uint   maxjumpdebug;         /**< max number of positions to be saved to save photon trajectory when -D M is used */
    cl_uint   gscatter;             /**< how many scattering events after which mus/g can be approximated by mus' */
    cl_uint   is2d;                 /**< is the domain a 2D slice? */
    cl_int    replaydet;                     /**< select which detector to replay, 0 for all, -1 save all separately */
    cl_uint   srcnum;               /**< total number of source patterns */
    cl_char   bc[8];               /**< boundary conditions */
} MCXParam POST_ALIGN(16);

void mcx_run_simulation(Config* cfg, float* fluence, float* totalenergy);
cl_platform_id mcx_list_gpu(Config* cfg, unsigned int* activedev, cl_device_id* activedevlist, GPUInfo** info);
void ocl_assess(int cuerr, const char* file, const int linenum);

#ifdef  __cplusplus
}
#endif

#endif
