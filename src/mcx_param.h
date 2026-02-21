/***************************************************************************//**
**  \file    mcx_param.h
**  \brief   Shared GPU parameter struct used by both OpenCL and CUDA backends
**
**  This header defines MCXParam, MCXSrc, Stokes, and RandType which are
**  needed by:
**    - mcx_host.h / mcx_host.cpp   (OpenCL host)
**    - mcx_cu_host.h / mcx_cu_host.cu (CUDA host)
**    - mcx_core.cl                  (GPU kernel, OpenCL-only path)
**
**  Extracted from mcx_host.h to avoid circular dependencies.
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025, GPL v3
*******************************************************************************/

#ifndef _MCX_PARAM_H
#define _MCX_PARAM_H

#include "vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Source data structure for the GPU kernel
 */
typedef struct MCXSource {
    float4 pos;      /**< initial position vector, for pencil beam */
    float4 dir;      /**< initial direction vector, for pencil beam */
    float4 param1;   /**< source parameters set 1 */
    float4 param2;   /**< source parameters set 2 */
} MCXSrc;

/**
 * @brief Stokes vector for polarized light simulation
 */
typedef struct StokesVector {
    float i; /**< total light intensity: IH + IV */
    float q; /**< IH - IV */
    float u; /**< I(+pi/4) - I(-pi/4) */
    float v; /**< IR - IL */
} Stokes;

/**
 * @brief Simulation constant parameters stored in GPU constant memory
 *
 * This struct must have IDENTICAL layout to the copy in mcx_core.cl
 * (inside the #ifndef __NVCC__ block) and the original mcx_host.h.
 * Any change here must be mirrored in both places.
 */
typedef struct KernelParams {
    MCXSrc src;                  /**< additional source data, including pos, dir, param1, param2 */
    uint   extrasrclen;          /**< number of additional sources */
    int    srcid;                /**< 0: all combined, -1: all separate, >0: single source */
    float4 maxidx;               /**< maximum index in x/y/z directions for out-of-bound tests */
    uint4  dimlen;               /**< maximum index used to convert x/y/z to 1D array index */
    uint4  cp0;                  /**< 3D coordinates of one diagonal of the cached region  (obsolete) */
    uint4  cp1;                  /**< 3D coordinates of the other diagonal of the cached region  (obsolete) */
    uint2  cachebox;             /**< stride for cachebox data acess  (obsolete) */
    float  minstep;              /**< minimum step of the 3, always 1 */
    float  twin0;                /**< starting time of the current time gate, unit is s */
    float  twin1;                /**< end time of the current time gate, unit is s  */
    float  tmax;                 /**< maximum time gate length, same as cfg.tend */
    float  oneoverc0;            /**< 1/(speed of light in the vacuum)*/
    uint   isrowmajor;
    uint   save2pt;              /**< flag if mcx outputs fluence volume */
    uint   doreflect;            /**< flag if mcx performs reflection calculations */
    uint   dorefint;             /**< flag if mcx perform reflection calculations at internal boundaries */
    uint   savedet;              /**< flag if mcx outputs detected photon partial length data */
    float  Rtstep;               /**< reciprocal of the step size */
    float  minenergy;            /**< threshold of weight to trigger Russian roulette */
    float  skipradius2;          /**< square of the radius within which the data is cached (obsolete) */
    float  minaccumtime;         /**< time steps for tMCimg like weight accummulation (obsolete) */
    uint   maxdetphoton;         /**< max number of detected photons */
    uint   maxmedia;             /**< max number of media labels */
    uint   detnum;               /**< max number of detectors */
    uint   blockphoton;
    uint   blockextra;
    uint   voidtime;             /**< flag if the time-of-flight in the background is counted */
    uint   srctype;              /**< type of the source */
    uint   maxvoidstep;          /**< max steps that photon can travel in the background before entering non-zero voxels */
    uint   issaveexit;           /**<1 save the exit position and dir of a detected photon, 0 do not save*/
    uint   issaveseed;           /**< flag if one need to save the detected photon seeds for replay */
    uint   issaveref;            /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    uint   isspecular;           /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    uint   maxgate;              /**< max number of time gates */
    int    seed;                 /**< RNG seed passed from the host */
    uint   outputtype;           /**< Type of output to be accummulated */
    uint   threadphoton;         /**< how many photons to be simulated in a thread */
    int    oddphoton;            /**< how many threads need to simulate 1 more photon above the basic load (threadphoton) */
    uint   debuglevel;           /**< debug flags */
    uint   savedetflag;          /**< detected photon save flags */
    uint   reclen;               /**< length of buffer per detected photon */
    uint   partialdata;          /**< per-medium detected photon data length */
    uint   w0offset;             /**< photon-sharing buffer offset */
    uint   mediaformat;          /**< format of the media buffer */
    uint   maxjumpdebug;         /**< max number of positions to be saved to save photon trajectory when -D M is used */
    uint   gscatter;             /**< how many scattering events after which mus/g can be approximated by mus' */
    uint   is2d;                 /**< is the domain a 2D slice? */
    int    replaydet;            /**< select which detector to replay, 0 for all, -1 save all separately */
    uint   srcnum;               /**< total number of source patterns */
    unsigned int nphase;         /**< number of samples for inverse-cdf, will be added by 2 to include -1 and 1 on the two ends */
    unsigned int nphaselen;      /**< even-rounded nphase so that shared memory buffer won't give an error */
    unsigned int nangle;         /**< number of samples for launch angle inverse-cdf, will be added by 2 to include 0 and 1 on the two ends */
    unsigned int nanglelen;      /**< even-rounded nangle so that shared memory buffer won't give an error */
    uint   maxpolmedia;          /**< max number of media labels for polarized light */
    uint   istrajstokes;         /**< 1 to save Stokes IQUV in trajectory data */
    float4 s0;                   /**< initial stokes parameters */
    float  omega;                /**< modulation angular frequency (2*pi*f), for FD/RF replay */
    unsigned char bc[12];        /**< boundary conditions */
} MCXParam POST_ALIGN(16);

/**
 * @brief RNG type definition
 */
#ifdef __CUDACC__
typedef unsigned long long RandType;
#else
#ifdef __APPLE__
typedef unsigned long long RandType;
#else
typedef unsigned long RandType;
#endif
#endif

#define RAND_BUF_LEN  2
#define RAND_SEED_LEN 4

#ifdef __cplusplus
}
#endif

#endif /* _MCX_PARAM_H */