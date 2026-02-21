/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**      -- OpenCL/CUDA dual-backend edition
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025
**
**  \section sref Reference:
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**         "Scalable and massively parallel Monte Carlo photon transport simulations
**         for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504 (2018)
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
**
**  \section sdualbackend Dual-Backend Support
**  This kernel compiles under both OpenCL (clBuildProgram) and CUDA (nvcc).
*******************************************************************************/

#ifndef MCX_GPU_COMPAT_H
#define MCX_GPU_COMPAT_H

/*============================================================================
 * Section 1: Address-space qualifiers and kernel launch
 *
 * OpenCL uses __global, __local, __constant, __private, __kernel.
 * CUDA uses none of these (flat address space) except __global__ for kernels.
 *==========================================================================*/

#ifdef __NVCC__
    /* CUDA: suppress OpenCL address-space qualifiers */
    #define __constant    const
    #define __private
    #define __local
    #define __global
    #define __kernel      __global__
#else
    /* OpenCL: __constant__ used in kernel source for CUDA-style declarations */
    #define __constant__  __constant
    #define __device__
#endif

/*============================================================================
 * Section 2: Vector type constructors and conversions
 *
 * OpenCL has built-in (float4)(a,b,c,d) syntax and convert_* functions.
 * CUDA needs make_float4() etc. We also provide operator overloads for CUDA.
 *==========================================================================*/

#ifdef __NVCC__

/* --- float3 operators --- */
inline __device__ __host__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ __host__ void operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __device__ __host__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __device__ __host__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}
inline __device__ __host__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __device__ __host__ float3 operator*(float f, float3 v) {
    return make_float3(v.x * f, v.y * f, v.z * f);
}
inline __device__ __host__ float3 operator*(float3 v, float f) {
    return make_float3(v.x * f, v.y * f, v.z * f);
}
inline __device__ __host__ void operator*=(float3& v, float f) {
    v.x *= f;
    v.y *= f;
    v.z *= f;
}

/* --- float4 operators --- */
inline __device__ __host__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __device__ __host__ void operator+=(float4& a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __device__ __host__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __device__ __host__ float4 operator*(float4 v, float f) {
    return make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
}
inline __device__ __host__ float4 operator*(float f, float4 v) {
    return make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
}
inline __device__ __host__ float4 operator/(float4 v, float4 d) {
    return make_float4(v.x / d.x, v.y / d.y, v.z / d.z, v.w / d.w);
}

/* --- Built-in function shims --- */
inline __device__ __host__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __device__ __host__ int get_global_id(int d) {
    return (d == 0) ? blockIdx.x * blockDim.x + threadIdx.x
           : (d == 1) ? blockIdx.y * blockDim.y + threadIdx.y
           :          blockIdx.z * blockDim.z + threadIdx.z;
}
inline __device__ __host__ int get_local_id(int d) {
    return (d == 0) ? threadIdx.x : (d == 1) ? threadIdx.y : threadIdx.z;
}
inline __device__ __host__ int get_local_size(int d) {
    return (d == 0) ? blockDim.x : (d == 1) ? blockDim.y : blockDim.z;
}
inline __device__ __host__ int get_num_groups(int d) {
    return (d == 0) ? gridDim.x : (d == 1) ? gridDim.y : gridDim.z;
}
inline __device__ __host__ int get_global_size(int d) {
    return get_local_size(d) * get_num_groups(d);
}

/* --- Type conversion shims --- */
inline __device__ float4 convert_float4_rtp(short4 v) {
    return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}
inline __device__ short3 convert_short3_rtn(float3 v) {
    return make_short3((short)floorf(v.x), (short)floorf(v.y), (short)floorf(v.z));
}
inline __device__ short convert_short_rtn(float v) {
    return (short)floorf(v);
}
inline __device__ short convert_short_rte(float v) {
    return (short)rintf(v);
}
inline __device__ float convert_float_rte(float v) {
    return rintf(v);
}

/* isgreater() - used in OpenCL branchless patterns */
inline __device__ float4 isgreater_f4(float4 a, float4 b) {
    return make_float4(a.x > b.x ? 1.f : 0.f, a.y > b.y ? 1.f : 0.f, a.z > b.z ? 1.f : 0.f, a.w > b.w ? 1.f : 0.f);
}

#endif /* __NVCC__ */

/*============================================================================
 * Section 3: Math function wrappers
 *
 * MCX can use native_* fast-math functions in OpenCL, or standard math in CUDA.
 * MCX_MATHFUN(log) expands to native_log or log depending on MCX_USE_NATIVE.
 * MCX_SINCOS provides sincos() which differs between OpenCL and CUDA.
 *==========================================================================*/

#ifdef __NVCC__
    #ifdef MCX_USE_NATIVE
        #define MCX_MATHFUN(fun)             fun##f   /* CUDA fast math via -use_fast_math */
    #else
        #define MCX_MATHFUN(fun)             fun##f
    #endif
    #define MCX_SINCOS(theta,osin,ocos)  sincosf((theta),&(osin),&(ocos))
#else
    #ifdef MCX_USE_NATIVE
        #define MCX_MATHFUN(fun)             native_##fun
        #define MCX_SINCOS(theta,osin,ocos)  {(osin)=native_sin(theta);(ocos)=native_cos(theta);}
    #else
        #define MCX_MATHFUN(fun)             fun
        #define MCX_SINCOS(theta,osin,ocos)  (ocos)=sincos((theta),&(osin))
    #endif
#endif

/*============================================================================
 * Section 4: Atomic operations
 *
 * Unified atomicadd() for float across all GPU vendors.
 * CUDA: uses native atomicAdd (available since sm_20).
 * OpenCL: uses PTX inline asm for NVIDIA, extensions or CAS fallback for others.
 *==========================================================================*/

#ifdef USE_ATOMIC

#ifdef __NVCC__

/* CUDA: native float atomicAdd, wrap as atomicadd for compatibility */
inline __device__ float atomicadd(volatile float* addr, float val) {
    return atomicAdd((float*)addr, val);
}

#else /* OpenCL atomic implementations */

#if defined(USE_NVIDIA_GPU) && !defined(USE_OPENCL_ATOMIC)
/* Tier 1: NVIDIA PTX inline asm (all NVIDIA GPUs) */
inline float atomicadd(volatile __global float* address, const float value) {
    float old;
    asm volatile(
        "atom.global.add.f32 %0, [%1], %2;"
        : "=f"(old) : "l"(address), "f"(value) : "memory"
    );
    return old;
}
#elif defined(USE_INTEL_FLOAT_ATOMIC)
/* Tier 2: Intel cl_intel_global_float_atomics */
#pragma OPENCL EXTENSION cl_intel_global_float_atomics : enable
inline float atomicadd(volatile __global float* address, const float value) {
    return atomic_add((volatile __global float*)address, value);
}
#elif defined(USE_OPENCL_FLOAT_ATOMIC)
/* Tier 3: Cross-vendor cl_ext_float_atomics (AMD RDNA2+, Intel) */
#pragma OPENCL EXTENSION cl_ext_float_atomics : enable
inline float atomicadd(volatile __global float* address, const float value) {
    return atomic_fetch_add_explicit(
               (volatile __global atomic_float*)address,
               value, memory_order_relaxed, memory_scope_device);
}
#else
/* Tier 4: CAS fallback (universal) */
inline float atomicadd(volatile __global float* address, const float value) {
    float old = value, orig;

    while ((old = atomic_xchg(address, (orig = atomic_xchg(address, 0.0f)) + old)) != 0.0f);

    return orig;
}
#endif

#endif /* __NVCC__ vs OpenCL */
#endif /* USE_ATOMIC */

/* atomic_inc compatibility */
#ifdef __NVCC__
    #define atomic_inc(x)    atomicAdd((unsigned int*)(x), 1u)
    #define atomic_dec(x)    atomicSub((unsigned int*)(x), 1u)
#endif

/*============================================================================
 * Section 5: Debug printing
 *==========================================================================*/

#ifdef MCX_GPU_DEBUG
    #define GPUDEBUG(x)  printf x
#else
    #define GPUDEBUG(x)
#endif

/*============================================================================
 * Section 6: OpenCL extension pragmas (no-op under CUDA)
 *==========================================================================*/

#ifndef __NVCC__
    #ifdef MCX_SAVE_DETECTORS
        #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
    #endif
    #if defined(USE_HALF) || MED_TYPE==MEDIA_AS_F2H || MED_TYPE==MEDIA_AS_HALF || MED_TYPE==MEDIA_LABEL_HALF
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #endif
#endif

/*============================================================================
 * Section 7: Half-float support
 *==========================================================================*/

#ifdef USE_HALF
    #ifndef __NVCC__
        #define FLOAT4VEC half4
        #define TOFLOAT4  convert_half4
    #else
        /* CUDA: use float4 even in USE_HALF mode for the kernel;
        half conversion done via __half2float in updateproperty */
        #define FLOAT4VEC float4
        #define TOFLOAT4
    #endif
#else
    #define FLOAT4VEC float4
    #define TOFLOAT4
#endif

/*============================================================================
 * Section 8: RNG type and shared memory declaration
 *==========================================================================*/

#ifdef __NVCC__
    #ifndef _SYS_TYPES_H
        typedef unsigned long long ulong;
    #endif
    typedef unsigned int       uint;
    typedef unsigned short     ushort;
    /* CUDA shared memory declared via extern in kernel launch */
#endif

/*============================================================================
 * Section 9: Barrier / memory fence
 *==========================================================================*/

#ifdef __NVCC__
    #define barrier(x)       __syncthreads()
    #define CLK_LOCAL_MEM_FENCE  0  /* unused in CUDA, barrier is full */
#endif

/*============================================================================
 * Section 10: GPU_PARAM macro for constant memory access
 *
 * When USE_MACRO_CONST is defined (optlevel>=3), parameters are injected
 * as compiler macros (gcfgfoo=value) for both backends.
 * Otherwise, they are read from the gcfg constant-memory struct.
 *==========================================================================*/

#ifndef USE_MACRO_CONST
    #define GPU_PARAM(a,b)  (a->b)
#else
    #define GPU_PARAM(a,b)  (a ## b)
#endif

/*============================================================================
 * Section 11: native_divide compatibility
 *==========================================================================*/

#ifdef __NVCC__
    #define native_divide(a,b)  __fdividef(a,b)
#endif

/*============================================================================
 * Section 12: Float4/Float3 constructor macros
 *
 * OpenCL uses (float4)(a,b,c,d) cast syntax.
 * CUDA uses make_float4(a,b,c,d).
 * These macros unify the construction so the kernel body stays clean.
 *==========================================================================*/

#ifdef __NVCC__
    #define FLOAT4(a,b,c,d) make_float4((a),(b),(c),(d))
    #define FLOAT3(a,b,c)   make_float3((a),(b),(c))
    #define FL4(f)           make_float4(f,f,f,f)
    #define FL3(f)           make_float3(f,f,f)
    #define SHORT4(a,b,c,d) make_short4((a),(b),(c),(d))
#else
    #define FLOAT4(a,b,c,d) ((float4)((a),(b),(c),(d)))
    #define FLOAT3(a,b,c)   ((float3)((a),(b),(c)))
    #define FL4(f)           ((float4)(f))
    #define FL3(f)           ((float3)(f))
    #define SHORT4(a,b,c,d) ((short4)((a),(b),(c),(d)))
#endif

/*============================================================================
 * Section 13: Miscellaneous compatibility
 *==========================================================================*/

#ifdef __NVCC__
    #ifndef NULL
        #define NULL 0
    #endif
    /* CUDA has no built-in rsqrt for different precisions, rsqrtf is available */
    #define rsqrt(x) rsqrtf(x)
#endif

/* fabs for float4 components - OpenCL has fabs() for vectors, CUDA uses fabsf */
#ifdef __NVCC__
    #define FABS(x) fabsf(x)
#else
    #define FABS(x) fabs(x)
#endif


/*--- floor/rint: OpenCL uses floor()/rint(), CUDA uses FLOOR()/rintf() ---*/
#ifdef __NVCC__
    #define FLOOR(x)  floorf(x)
    #define RINT(x)   rintf(x)
#else
    #define FLOOR(x)  floor(x)
    #define RINT(x)   rint(x)
#endif

/*--- acos: OpenCL has no native_acos, only acos. CUDA uses acosf. ---*/
#ifdef __NVCC__
    #define ACOS(x)   acosf(x)
#else
    #define ACOS(x)   acos(x)
#endif

/*--- as_int / address-of-vector-element workaround ---*/
#ifdef __NVCC__
    #define AS_INT(x)   (*((int*)&(x)))
    #define AS_UINT(x)  (*((uint*)&(x)))
#else
    #define AS_INT(x)   as_int(x)
    #define AS_UINT(x)  as_uint(x)
#endif



#ifdef __NVCC__
/* Template parameters are injected by the template<...> declaration.
   These macros convert template constants into the same tests that
   the OpenCL #ifdef pattern uses. The compiler optimizes away dead
   branches since the template params are compile-time constants. */
#define MCX_IS_SRC_PENCIL      (ispencil)
#define MCX_IS_REFLECT         (isreflect)
#define MCX_IS_LABEL           (islabel)
#define MCX_IS_SVMC            (issvmc)
#define MCX_IS_POLARIZED       (ispolarized)

/* Source type dispatch: CUDA uses runtime switch guarded by template.
   The outer if(ispencil) eliminates the entire switch for pencil beams. */
#define MCX_SRC_IS(type)       (gcfg->srctype == (type))

#else /* OpenCL */
/* Under OpenCL, source type is selected by -DMCX_SRC_PENCIL etc.
   We define MCX_IS_* based on whether the macro is defined. */
#ifdef MCX_SRC_PENCIL
    #define MCX_IS_SRC_PENCIL  1
#else
    #define MCX_IS_SRC_PENCIL  0
#endif

#if defined(__NVCC__) || defined(MCX_DO_REFLECTION)
#ifdef __NVCC__

if (isreflect) {
#endif
#define MCX_IS_REFLECT     1
#else
#define MCX_IS_REFLECT     0
#endif

    /* For OpenCL, label/svmc/polarized are determined by MED_TYPE and flags */
#if MED_TYPE <= 4
    #define MCX_IS_LABEL       1
#else
    #define MCX_IS_LABEL       0
#endif

#if MED_TYPE == MEDIA_2LABEL_SPLIT
    #define MCX_IS_SVMC        1
#else
    #define MCX_IS_SVMC        0
#endif

    /* Polarized is not a compile-time flag in OpenCL currently;
       set to 0 by default, can be overridden with -DMCX_IS_POLARIZED=1 */
#ifndef MCX_IS_POLARIZED
    #define MCX_IS_POLARIZED   0
#endif
#endif

#ifdef __NVCC__
    #define LAUNCHNEWPHOTON launchnewphoton<ispencil, isreflect, islabel, issvmc, ispolarized>
#else
    #define LAUNCHNEWPHOTON launchnewphoton
#endif

#endif /* MCX_GPU_COMPAT_H */

/*=== Constants ===*/

#define R_PI               0.318309886183791f
#ifndef RAND_MAX
    #define RAND_MAX       4294967295
#endif

#define ONE_PI             3.1415926535897932f
#define TWO_PI             6.28318530717959f
#define JUST_ABOVE_ONE     1.0001f
#define JUST_BELOW_ONE     0.9998f

#define C0                 299792458000.f
#define R_C0               3.335640951981520e-12f

#define SIGN_BIT           0x80000000U

#ifdef __NVCC__
    #undef EPS
    #undef VERY_BIG
#endif
#ifndef EPS
    #define EPS            FLT_EPSILON
#endif
#ifndef VERY_BIG
    #define VERY_BIG       (1.f/FLT_EPSILON)
#endif
#ifdef __NVCC__
    /* kernel needs FLT_EPSILON, not the 1e-10 from mcx_const.h */
    #undef EPS
    #define EPS            1.19209290E-07F
    #undef VERY_BIG
    #define VERY_BIG       (1.f/EPS)
#endif

#define SAME_VOXEL         -9999.f
#define NO_LAUNCH          9999
#define FILL_MAXDETPHOTON  3
#define MAX_PROP           2000
#ifndef OUTSIDE_VOLUME_MIN
    #define OUTSIDE_VOLUME_MIN 0xFFFFFFFFU
#endif
#ifndef OUTSIDE_VOLUME_MAX
    #define OUTSIDE_VOLUME_MAX 0x7FFFFFFFU
#endif
#ifndef BOUNDARY_DET_MASK
    #define BOUNDARY_DET_MASK  0xFFFF0000
#endif
#ifndef SEED_FROM_FILE
    #define SEED_FROM_FILE     -999
#endif

#define ROULETTE_SIZE      10.f

#ifndef DET_MASK
    #define DET_MASK       0x80000000
#endif
#ifndef MED_MASK
    #define MED_MASK       0x7FFFFFFF
#endif
#define MIX_MASK           0x7FFF0000
#define MAX_ACCUM          1000.f

#define MCX_DEBUG_REC_LEN  6
#define NANGLES            181

#ifndef MCX_DEBUG_MOVE
    #define MCX_DEBUG_MOVE         2
#endif
#ifndef MCX_DEBUG_MOVE_ONLY
    #define MCX_DEBUG_MOVE_ONLY    8
#endif
#ifndef MCX_DEBUG_RNG
    #define MCX_DEBUG_RNG          1
#endif
#ifndef MCX_DEBUG_PROGRESS
    #define MCX_DEBUG_PROGRESS     2048
#endif


#define MEDIA_ASGN_F2H     96
#ifndef MIN
    #define MIN(a,b)       ((a)<(b)?(a):(b))
#endif
#define MEDIA_2LABEL_SPLIT 97
#define MEDIA_2LABEL_MIX   98
#define MEDIA_LABEL_HALF   99
#define MEDIA_AS_F2H       100
#define MEDIA_MUA_FLOAT    101
#define MEDIA_AS_HALF      102
#define MEDIA_ASGN_BYTE    103
#define MEDIA_AS_SHORT     104

/*=== Detector save flags ===*/
#define SAVE_DETID(a)      ((a)    & 0x1)
#define SAVE_NSCAT(a)      ((a)>>1 & 0x1)
#define SAVE_PPATH(a)      ((a)>>2 & 0x1)
#define SAVE_MOM(a)        ((a)>>3 & 0x1)
#define SAVE_PEXIT(a)      ((a)>>4 & 0x1)
#define SAVE_VEXIT(a)      ((a)>>5 & 0x1)
#define SAVE_W0(a)         ((a)>>6 & 0x1)
#define SAVE_IQUV(a)       ((a)>>7 & 0x1)

#define SET_SAVE_DETID(a)     ((a) | 0x1   )
#define SET_SAVE_NSCAT(a)     ((a) | 0x1<<1)
#define SET_SAVE_PPATH(a)     ((a) | 0x1<<2)
#define SET_SAVE_MOM(a)       ((a) | 0x1<<3)
#define SET_SAVE_PEXIT(a)     ((a) | 0x1<<4)
#define SET_SAVE_VEXIT(a)     ((a) | 0x1<<5)
#define SET_SAVE_W0(a)        ((a) | 0x1<<6)
#define SET_SAVE_IQUV(a)      ((a) | 0x1<<7)

#define UNSET_SAVE_DETID(a)   ((a) & ~(0x1)   )
#define UNSET_SAVE_NSCAT(a)   ((a) & ~(0x1<<1))
#define UNSET_SAVE_PPATH(a)   ((a) & ~(0x1<<2))
#define UNSET_SAVE_MOM(a)     ((a) & ~(0x1<<3))
#define UNSET_SAVE_PEXIT(a)   ((a) & ~(0x1<<4))
#define UNSET_SAVE_VEXIT(a)   ((a) & ~(0x1<<5))
#define UNSET_SAVE_W0(a)      ((a) & ~(0x1<<6))
#define UNSET_SAVE_IQUV(a)    ((a) & ~(0x1<<7))

/*=== SVMC packed bit-field macros ===*/
#define SV_LOWER(sv)       ((unsigned char)((sv) & 0xFF))
#define SV_UPPER(sv)       ((unsigned char)(((sv) >> 8) & 0xFF))
#define SV_ISSPLIT(sv)     (((sv) >> 16) & 1u)
#define SV_ISUPPER(sv)     (((sv) >> 17) & 1u)
#define SV_SET_LOWER(sv,v)   ((sv) = ((sv) & ~0xFFu) | ((unsigned int)(v) & 0xFF))
#define SV_SET_UPPER(sv,v)   ((sv) = ((sv) & ~0xFF00u) | (((unsigned int)(v) & 0xFF) << 8))
#define SV_SET_ISSPLIT(sv,v) ((sv) = ((sv) & ~0x10000u) | (((unsigned int)(!!(v))) << 16))
#define SV_SET_ISUPPER(sv,v) ((sv) = ((sv) & ~0x20000u) | (((unsigned int)(!!(v))) << 17))
#define SV_FLIP_ISUPPER(sv)  ((sv) ^= 0x20000u)
#define SV_CLEAR(sv)         ((sv) = 0u)
#define SV_CURLABEL(sv)    (SV_ISUPPER(sv) ? SV_UPPER(sv) : SV_LOWER(sv))

#define UPPER_MASK         0x00FF0000
#define LOWER_MASK         0xFF000000

/*=== Structs ===*/

typedef struct MCXSplit {
    float3 nv;
    float  pd;
    unsigned int sv;
} MCXsp;

#ifndef __NVCC__
/* OpenCL-only: these are defined in mcx_utils.h for CUDA */

typedef struct StokesVector {
    float i, q, u, v;
} Stokes;

typedef struct MCXSource {
    float4 pos, dir, param1, param2;
} MCXSrc;

typedef struct KernelParams {
    MCXSrc src;
    uint   extrasrclen;
    int    srcid;
    float4 maxidx;
    uint4  dimlen, cp0, cp1;
    uint2  cachebox;
    float  minstep;
    float  twin0, twin1, tmax;
    float  oneoverc0;
    uint   isrowmajor, save2pt, doreflect, dorefint, savedet;
    float  Rtstep;
    float  minenergy;
    float  skipradius2;
    float  minaccumtime;
    uint   maxdetphoton;
    uint   maxmedia;
    uint   detnum;
    uint   blockphoton;
    uint   blockextra;
    int    voidtime;
    int    srctype;
    uint   maxvoidstep;
    uint   issaveexit;
    uint   issaveseed;
    uint   issaveref;
    uint   isspecular;
    uint   maxgate;
    int    seed;
    uint   outputtype;
    uint   threadphoton;
    int    oddphoton;
    uint   debuglevel;
    uint   savedetflag;
    uint   reclen;
    uint   partialdata;
    uint   w0offset;
    uint   mediaformat;
    uint   maxjumpdebug;
    uint   gscatter;
    uint   is2d;
    int    replaydet;
    uint   srcnum;
    unsigned int nphase;
    unsigned int nphaselen;
    unsigned int nangle;
    unsigned int nanglelen;
    uint   maxpolmedia;
    uint   istrajstokes;
    float4 s0;
    float  omega;
    unsigned char bc[12];
} MCXParam __attribute__ ((aligned (32)));

enum TBoundary {bcUnknown, bcReflect, bcAbsorb, bcMirror, bcCyclic};
enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otWP, otDCS, otRF, otL, otRFmus, otWLTOF, otWPTOF};

#endif /* !__NVCC__ */

/*=== RNG (xorshift128+) ===*/

#define RAND_BUF_LEN       2
#define RAND_SEED_LEN      4
#define LOG_MT_MAX          22.1807097779182f

#ifndef __NVCC__
    typedef ulong  RandType;
#endif

__device__ static float xorshift128p_nextf(__private RandType t[RAND_BUF_LEN]) {
    union {
        ulong  i;
        float f[2];
        uint  u[2];
    } s1;
    const ulong s0 = t[1];
    s1.i = t[0];
    t[0] = s0;
    s1.i ^= s1.i << 23;
    t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5);
    s1.i = t[1] + s0;
    s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);
    return s1.f[0] - 1.0f;
}

__device__ static void copystate(__private RandType* t, __local RandType* tnew) {
    tnew[0] = t[0];
    tnew[1] = t[1];
}

__device__ static float rand_uniform01(__private RandType t[RAND_BUF_LEN]) {
    return xorshift128p_nextf(t);
}

__device__ static void xorshift128p_seed(__global uint* seed, RandType t[RAND_BUF_LEN]) {
    t[0] = (ulong)seed[0] << 32 | seed[1];
    t[1] = (ulong)seed[2] << 32 | seed[3];
}

__device__ static void gpu_rng_init(__private RandType t[RAND_BUF_LEN], __global uint* n_seed, int idx) {
    xorshift128p_seed((n_seed + idx * RAND_SEED_LEN), t);
}

__device__ float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]);

__device__ float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]) {
    return -MCX_MATHFUN(log)(rand_uniform01(t) + EPS);
}

#define rand_next_aangle(t)  rand_uniform01(t)
#define rand_next_zangle(t)  rand_uniform01(t)
#define rand_next_reflect(t) rand_uniform01(t)
#define rand_do_roulette(t)  rand_uniform01(t)

/*=== Function prototypes ===*/

__device__ float dot3(float3 a, float3 b);
__device__ void clearpath(__local float* p, uint maxmediatype);
__device__ float mcx_nextafterf(float a, int dir);
__device__ float hitgrid(float4* p0, float4* v, short4* id);
__device__ void rotatevector(float4* v, float stheta, float ctheta, float sphi, float cphi);
__device__ void transmit(float4* v, float n1, float n2, short flipdir);
__device__ float reflectcoeff(float4* v, float n1, float n2, short flipdir);
__device__ void updateproperty(FLOAT4VEC* prop, unsigned int mediaid, __constant float4* gproperty, __constant MCXParam* gcfg);

#if MED_TYPE == MEDIA_ASGN_F2H || defined(__NVCC__)
__device__ void updateproperty_asgn(FLOAT4VEC* prop, unsigned int mediaid, uint idx1d,
                                    __global const uint* media, __constant float4* gproperty, __constant MCXParam* gcfg);
#endif

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
__device__ void updateproperty_svmc(FLOAT4VEC* prop, unsigned int mediaid, uint idx1d,
                                    __global const uint* media, float3 p, MCXsp* nuvox, short4* flipdir,
                                    __constant float4* gproperty, __constant MCXParam* gcfg);
__device__ int ray_plane_intersect(float3 p0, float4* v, FLOAT4VEC* prop, float* len, float* slen, MCXsp* nuvox, __constant MCXParam* gcfg);
__device__ int reflectray_svmc(float n1, float3* c0, MCXsp* nuvox, FLOAT4VEC* prop, __private RandType t[RAND_BUF_LEN], __constant float4* gproperty);
#endif

#ifndef INTERNAL_SOURCE
    __device__ int skipvoid(float4* p, float4* v, float4* f, short4* flipdir, __global const uint* media, __constant float4* gproperty, __constant MCXParam* gcfg, MCXsp* nuvox);
#endif

__device__ void rotate_perpendicular_vector(float4* photon_dir, float3 axis, float stheta, float ctheta);
__device__ void rotatevector2d(float4* v, float stheta, float ctheta, int is2d);

#ifdef MCX_SAVE_DETECTORS
__device__ uint finddetector(float4* p0, __constant float4* gdetpos, __constant MCXParam* gcfg);
__device__ void savedetphoton(__global float* n_det, __global uint* detectedphoton,
                              __local float* ppath, float4* p0, float4* v, Stokes* s,
                              __local RandType* t, __global RandType* seeddata,
                              __constant float4* gdetpos, __constant MCXParam* gcfg, uint isdet);
__device__ void saveexitppath(__global float* n_det, __local float* ppath, float4* p0, uint* idx1d, __constant MCXParam* gcfg);
#endif

__device__ int launchnewphoton(float4* p, float4* v, Stokes* s, float4* f, short4* flipdir, FLOAT4VEC* prop, uint* idx1d,
                               __global float* field, uint* mediaid, float* w0, float* Lmove, uint isdet,
                               __local float* ppath, __global float* n_det, __global uint* dpnum,
                               __private RandType t[RAND_BUF_LEN], __global RandType* rngseed,
                               __constant float4* gproperty, __global const uint* media, __global float* srcpattern,
                               __constant float4* gdetpos, __constant MCXParam* gcfg, int threadid,
                               __local int* blockphoton, volatile __global uint* gprogress,
                               __local RandType* photonseed, __global RandType* gseeddata,
                               __global uint* gjumpdebug, __global float* gdebugdata, __local RandType* sharedmem, MCXsp* nuvox,
                               __global float* photontof);

__device__ void savedebugdata(float4* p, uint id, __global uint* gjumpdebug, __global float* gdebugdata, __constant MCXParam* gcfg, int srcid);

/*=== Utility functions ===*/

__device__ void clearpath(__local float* p, uint maxmediatype) {
    uint i;

    for (i = 0; i < maxmediatype; i++) {
        p[i] = 0.f;
    }
}

__device__ void rotsphi(Stokes* s, float phi, Stokes* s2) {
    float sin2phi, cos2phi;
    MCX_SINCOS(2.f * phi, sin2phi, cos2phi);
    s2->i = s->i;
    s2->q = s->q * cos2phi + s->u * sin2phi;
    s2->u = -s->q * sin2phi + s->u * cos2phi;
    s2->v = s->v;
}

__device__ void updatestokes(Stokes* s, float theta, float phi, float3* u, float3* u2, uint* mediaid, __global float4* gsmatrix) {
    float costheta = MCX_MATHFUN(cos)(theta);
    Stokes s2;
    rotsphi(s, phi, &s2);

    uint imedia = NANGLES * ((*mediaid & MED_MASK) - 1);
    uint ithedeg = (uint)(theta * NANGLES * (R_PI - EPS));

    s->i = gsmatrix[imedia + ithedeg].x * s2.i + gsmatrix[imedia + ithedeg].y * s2.q;
    s->q = gsmatrix[imedia + ithedeg].y * s2.i + gsmatrix[imedia + ithedeg].x * s2.q;
    s->u = gsmatrix[imedia + ithedeg].z * s2.u + gsmatrix[imedia + ithedeg].w * s2.v;
    s->v = -gsmatrix[imedia + ithedeg].w * s2.u + gsmatrix[imedia + ithedeg].z * s2.v;

    float temp, sini, cosi, sin22, cos22;

    temp = (u2->z > -1.f && u2->z < 1.f) ? rsqrt((1.f - costheta * costheta) * (1.f - u2->z * u2->z)) : 0.f;

    cosi = (temp == 0.f) ? 0.f : (((phi > ONE_PI && phi < TWO_PI) ? 1.f : -1.f) * (u2->z * costheta - u->z) * temp);
    cosi = fmax(-1.f, fmin(cosi, 1.f));

    sini = MCX_MATHFUN(sqrt)(1.f - cosi * cosi);
    cos22 = 2.f * cosi * cosi - 1.f;
    sin22 = 2.f * sini * cosi;

    s2.i = s->i;
    s2.q = s->q * cos22 - s->u * sin22;
    s2.u = s->q * sin22 + s->u * cos22;
    s2.v = s->v;

    temp = native_divide(1.f, s2.i);
    s->q = s2.q * temp;
    s->u = s2.u * temp;
    s->v = s2.v * temp;
    s->i = 1.f;
}

#ifdef MCX_SAVE_DETECTORS
__device__ uint finddetector(float4* p0, __constant float4* gdetpos, __constant MCXParam* gcfg) {
    uint i;

    for (i = 0; i < GPU_PARAM(gcfg, detnum); i++) {
        if ((gdetpos[i].x - p0[0].x) * (gdetpos[i].x - p0[0].x) +
                (gdetpos[i].y - p0[0].y) * (gdetpos[i].y - p0[0].y) +
                (gdetpos[i].z - p0[0].z) * (gdetpos[i].z - p0[0].z) < gdetpos[i].w * gdetpos[i].w) {
            return i + 1;
        }
    }

    return 0;
}

__device__ void saveexitppath(__global float* n_det, __local float* ppath, float4* p0, uint* idx1d, __constant MCXParam* gcfg) {
    if (GPU_PARAM(gcfg, issaveref) > 1) {
        if (*idx1d >= GPU_PARAM(gcfg, maxdetphoton)) {
            return;
        }

        uint baseaddr = (*idx1d) * GPU_PARAM(gcfg, reclen);
        n_det[baseaddr] += p0[0].w;

        for (uint i = 0; i < GPU_PARAM(gcfg, maxmedia); i++) {
            n_det[baseaddr + i] += ppath[i] * p0[0].w;
        }
    }
}

__device__ void savedetphoton(__global float* n_det, __global uint* detectedphoton,
                              __local float* ppath, float4* p0, float4* v, Stokes* s,
                              __local RandType* t, __global RandType* seeddata,
                              __constant float4* gdetpos, __constant MCXParam* gcfg, uint isdet) {
    int detid;
    detid = (isdet == OUTSIDE_VOLUME_MIN) ? -1 : (int)finddetector(p0, gdetpos, gcfg);

    if (detid) {
        uint baseaddr = atomic_inc(detectedphoton);

        if (baseaddr < GPU_PARAM(gcfg, maxdetphoton)) {
            uint i;

            for (i = 0; i < GPU_PARAM(gcfg, issaveseed) * RAND_BUF_LEN; i++) {
                seeddata[baseaddr * RAND_BUF_LEN + i] = t[i];
            }

            baseaddr *= GPU_PARAM(gcfg, reclen);

            if (SAVE_DETID(GPU_PARAM(gcfg, savedetflag))) {
                if (GPU_PARAM(gcfg, extrasrclen) * (GPU_PARAM(gcfg, srcid) <= 0)) {
                    detid |= (((int)ppath[GPU_PARAM(gcfg, w0offset) - 1]) << 16);
                }

                n_det[baseaddr++] = detid;
            }

            for (i = 0; i < GPU_PARAM(gcfg, partialdata); i++) {
                n_det[baseaddr++] = ppath[i];
            }

            if (SAVE_PEXIT(GPU_PARAM(gcfg, savedetflag))) {
                *((__global float*)(n_det + baseaddr)) = p0[0].x;
                *((__global float*)(n_det + baseaddr + 1)) = p0[0].y;
                *((__global float*)(n_det + baseaddr + 2)) = p0[0].z;
                baseaddr += 3;
            }

            if (SAVE_VEXIT(GPU_PARAM(gcfg, savedetflag))) {
                *((__global float*)(n_det + baseaddr)) = v[0].x;
                *((__global float*)(n_det + baseaddr + 1)) = v[0].y;
                *((__global float*)(n_det + baseaddr + 2)) = v[0].z;
                baseaddr += 3;
            }

            if (SAVE_W0(GPU_PARAM(gcfg, savedetflag))) {
                n_det[baseaddr++] = ppath[GPU_PARAM(gcfg, w0offset) - 2];
            }

            if (SAVE_IQUV(GPU_PARAM(gcfg, savedetflag))) {
                n_det[baseaddr++] = s->i;
                n_det[baseaddr++] = s->q;
                n_det[baseaddr++] = s->u;
                n_det[baseaddr++] = s->v;
            }
        } else if (GPU_PARAM(gcfg, savedet) == FILL_MAXDETPHOTON) {
            atomic_dec(detectedphoton);
        }
    }
}
#endif

__device__ uint savedebugdata2(float4* p, uint id, __global uint* gjumpdebug, __global float* gdebugdata, __constant MCXParam* gcfg, int srcid) {
    uint pos = atomic_inc(gjumpdebug);

    if (pos < GPU_PARAM(gcfg, maxjumpdebug)) {
        pos *= MCX_DEBUG_REC_LEN + (GPU_PARAM(gcfg, istrajstokes) << 2);
        ((__global uint*)gdebugdata)[pos++] = id;
        gdebugdata[pos++] = p[0].x;
        gdebugdata[pos++] = p[0].y;
        gdebugdata[pos++] = p[0].z;
        gdebugdata[pos++] = p[0].w;
        gdebugdata[pos++] = srcid;
        return pos;
    }

    return 0;
}

__device__ void savedebugdata(float4* p, uint id, __global uint* gjumpdebug, __global float* gdebugdata, __constant MCXParam* gcfg, int srcid) {
    savedebugdata2(p, id, gjumpdebug, gdebugdata, gcfg, srcid);
}

__device__ void savedebugstokes(float4* p, Stokes* s, uint id, __global uint* gjumpdebug, __global float* gdebugdata, __constant MCXParam* gcfg, int srcid) {
    uint pos = savedebugdata2(p, id, gjumpdebug, gdebugdata, gcfg, srcid);

    if (pos > 0 && GPU_PARAM(gcfg, istrajstokes)) {
        gdebugdata[pos++] = s->i;
        gdebugdata[pos++] = s->q;
        gdebugdata[pos++] = s->u;
        gdebugdata[pos++] = s->v;
    }
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float mcx_nextafterf(float a, int dir) {
    union {
        float f;
        uint  i;
    } num;
    num.f = a + 1000.f;
    num.i += dir ^ (num.i & 0x80000000U);
    return num.f - 1000.f;
}

__device__ float hitgrid(float4* p0, float4* v, short4* id) {
    float dist;
    float4 htime;

#ifdef __NVCC__
    float4 fid = FLOAT4((float)id[0].x, (float)id[0].y, (float)id[0].z, (float)id[0].w);
    float4 gt  = FLOAT4(v[0].x > 0.f ? 1.f : 0.f, v[0].y > 0.f ? 1.f : 0.f, v[0].z > 0.f ? 1.f : 0.f, 0.f);
    htime = FLOAT4(
                FABS((fid.x + gt.x - p0[0].x) * native_divide(1.f, v[0].x + (v[0].x == 0.f) * EPS)),
                FABS((fid.y + gt.y - p0[0].y) * native_divide(1.f, v[0].y + (v[0].y == 0.f) * EPS)),
                FABS((fid.z + gt.z - p0[0].z) * native_divide(1.f, v[0].z + (v[0].z == 0.f) * EPS)),
                0.f);
#else
    htime = fabs(convert_float4_rtp(id[0]) - convert_float4_rtp(isgreater(v[0], ((float4)(0.f)))) - p0[0]);
    htime = fabs(native_divide(htime + (float4)EPS, v[0]));
#endif

    dist = fmin(fmin(htime.x, htime.y), htime.z);
    id->w = (dist == htime.x ? 0 : (dist == htime.y ? 1 : 2));
    return dist;
}

__device__ void rotate_perpendicular_vector(float4* photon_dir, float3 axis, float stheta, float ctheta) {
    float3 cross_v;
    cross_v.x = axis.y * photon_dir->z - axis.z * photon_dir->y;
    cross_v.y = axis.z * photon_dir->x - axis.x * photon_dir->z;
    cross_v.z = axis.x * photon_dir->y - axis.y * photon_dir->x;

    photon_dir->x = photon_dir->x * ctheta + cross_v.x * stheta;
    photon_dir->y = photon_dir->y * ctheta + cross_v.y * stheta;
    photon_dir->z = photon_dir->z * ctheta + cross_v.z * stheta;
}

__device__ void rotatevector2d(float4* v, float stheta, float ctheta, int is2d) {
    if (is2d == 1) {
        *((float4*)v) = FLOAT4(0.f, v[0].y * ctheta - v[0].z * stheta, v[0].y * stheta + v[0].z * ctheta, v[0].w);
    } else if (is2d == 2) {
        *((float4*)v) = FLOAT4(v[0].x * ctheta - v[0].z * stheta, 0.f, v[0].x * stheta + v[0].z * ctheta, v[0].w);
    } else if (is2d == 3) {
        *((float4*)v) = FLOAT4(v[0].x * ctheta - v[0].y * stheta, v[0].x * stheta + v[0].y * ctheta, 0.f, v[0].w);
    }

    GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n", v[0].x, v[0].y, v[0].z));
}

__device__ void rotatevector(float4* v, float stheta, float ctheta, float sphi, float cphi) {
    if ( v[0].z > -1.f + EPS && v[0].z < 1.f - EPS ) {
        float tmp0 = 1.f - v[0].z * v[0].z;
        float tmp1 = stheta * rsqrt(tmp0);
        *((float4*)v) = FLOAT4(
                            tmp1 * (v[0].x * v[0].z * cphi - v[0].y * sphi) + v[0].x * ctheta,
                            tmp1 * (v[0].y * v[0].z * cphi + v[0].x * sphi) + v[0].y * ctheta,
                            -tmp1 * tmp0 * cphi                          + v[0].z * ctheta,
                            v[0].w);
    } else {
        v[0] = FLOAT4(stheta * cphi, stheta * sphi, (v[0].z > 0.f) ? ctheta : -ctheta, v[0].w);
    }

    v[0].x *= rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z);
    v[0].y *= rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z);
    v[0].z *= rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z);
    GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n", v[0].x, v[0].y, v[0].z));
}

__device__ void transmit(float4* v, float n1, float n2, short flipdir) {
    float tmp0 = n1 / n2;
    v[0].x *= tmp0;
    v[0].y *= tmp0;
    v[0].z *= tmp0;

    (flipdir == 0) ?
    (v[0].x = MCX_MATHFUN(sqrt)(1.f - v[0].y * v[0].y - v[0].z * v[0].z) * ((v[0].x > 0.f) - (v[0].x < 0.f))) :
    ((flipdir == 1) ?
     (v[0].y = MCX_MATHFUN(sqrt)(1.f - v[0].x * v[0].x - v[0].z * v[0].z) * ((v[0].y > 0.f) - (v[0].y < 0.f))) :
     (v[0].z = MCX_MATHFUN(sqrt)(1.f - v[0].x * v[0].x - v[0].y * v[0].y) * ((v[0].z > 0.f) - (v[0].z < 0.f))));
}

__device__ float reflectcoeff(float4* v, float n1, float n2, short flipdir) {
    float Icos = FABS((flipdir == 0) ? v[0].x : (flipdir == 1 ? v[0].y : v[0].z));
    float tmp0 = n1 * n1;
    float tmp1 = n2 * n2;
    float tmp2 = 1.f - tmp0 / tmp1 * (1.f - Icos * Icos);

    if (tmp2 > 0.f) {
        float Re, Im, Rtotal;
        Re = tmp0 * Icos * Icos + tmp1 * tmp2;
        tmp2 = MCX_MATHFUN(sqrt)(tmp2);
        Im = 2.f * n1 * n2 * Icos * tmp2;
        Rtotal = (Re - Im) / (Re + Im);
        Re = tmp1 * Icos * Icos + tmp0 * tmp2 * tmp2;
        Rtotal = (Rtotal + (Re - Im) / (Re + Im)) * 0.5f;
        return Rtotal;
    } else {
        return 1.f;
    }
}

/*=== updateproperty: loading optical properties from various media formats ===*/

__device__ void updateproperty(FLOAT4VEC* prop, unsigned int mediaid, __constant float4* gproperty, __constant MCXParam* gcfg) {

#ifdef __NVCC__

    if (gcfg->mediaformat <= 4) {
#endif
#if !defined(__NVCC__) && MED_TYPE <= 4
#endif
        *((FLOAT4VEC*)(prop)) = gproperty[mediaid & MED_MASK];
#ifdef __NVCC__
    } else if (gcfg->mediaformat == MEDIA_MUA_FLOAT) {
#endif
#if !defined(__NVCC__) && MED_TYPE == MEDIA_MUA_FLOAT
#endif
#if defined(__NVCC__) || MED_TYPE == MEDIA_MUA_FLOAT
        prop[0].x = FABS(*((float*)&mediaid));
        prop[0].w = gproperty[(!(mediaid & MED_MASK)) == 0].w;
#endif

#ifdef __NVCC__
    } else if (gcfg->mediaformat == MEDIA_AS_F2H || gcfg->mediaformat == MEDIA_AS_HALF) {
#endif
#if !defined(__NVCC__) && (MED_TYPE == MEDIA_AS_F2H || MED_TYPE == MEDIA_AS_HALF)
#endif
#if defined(__NVCC__) || MED_TYPE == MEDIA_AS_F2H || MED_TYPE == MEDIA_AS_HALF
        {
            union {
                unsigned int i;
#ifdef __NVCC__
                __half_raw h[2];
#else
                half h[2];
#endif
            } val;
            val.i = mediaid & MED_MASK;
#ifdef __NVCC__
            prop[0].x = FABS(__half2float(val.h[0]));
            prop[0].y = FABS(__half2float(val.h[1]));
#else
            prop[0].x = fabs(convert_float(vload_half(0, val.h)));
            prop[0].y = fabs(convert_float(vload_half(1, val.h)));
#endif
            prop[0].w = gproperty[(!(mediaid & MED_MASK)) == 0].w;
        }
#endif

#ifdef __NVCC__
    } else if (gcfg->mediaformat == MEDIA_LABEL_HALF) {
#endif
#if !defined(__NVCC__) && MED_TYPE == MEDIA_LABEL_HALF
#endif
#if defined(__NVCC__) || MED_TYPE == MEDIA_LABEL_HALF
        {
            union {
                unsigned int i;
#ifdef __NVCC__
                __half_raw h[2];
#else
                half h[2];
#endif
                unsigned short s[2];
            } val;
            val.i = mediaid & MED_MASK;
            * ((FLOAT4VEC*)(prop)) = gproperty[val.s[0] & 0x3FFF];
            float* p = (float*)(prop);
#ifdef __NVCC__
            p[(val.s[0] & 0xC000) >> 14] = FABS(__half2float(val.h[1]));
#else
            p[(val.s[0] & 0xC000) >> 14] = fabs(convert_float(vload_half(1, val.h)));
#endif
        }
#endif

#ifdef __NVCC__
    } else if (gcfg->mediaformat == MEDIA_ASGN_BYTE) {
#endif
#if !defined(__NVCC__) && MED_TYPE == MEDIA_ASGN_BYTE
#endif
#if defined(__NVCC__) || MED_TYPE == MEDIA_ASGN_BYTE
        {
            union {
                unsigned int i;
                unsigned char h[4];
            } val;
            val.i = mediaid & MED_MASK;
            prop[0].x = val.h[0] * (1.f / 255.f) * (gproperty[2].x - gproperty[1].x) + gproperty[1].x;
            prop[0].y = val.h[1] * (1.f / 255.f) * (gproperty[2].y - gproperty[1].y) + gproperty[1].y;
            prop[0].z = val.h[2] * (1.f / 255.f) * (gproperty[2].z - gproperty[1].z) + gproperty[1].z;
            prop[0].w = val.h[3] * (1.f / 127.f) * (gproperty[2].w - gproperty[1].w) + gproperty[1].w;
        }
#endif

#ifdef __NVCC__
    } else if (gcfg->mediaformat == MEDIA_AS_SHORT) {
#endif
#if !defined(__NVCC__) && MED_TYPE == MEDIA_AS_SHORT
#endif
#if defined(__NVCC__) || MED_TYPE == MEDIA_AS_SHORT
        {
            union {
                unsigned int i;
                unsigned short h[2];
            } val;
            val.i = mediaid & MED_MASK;
            prop[0].x = val.h[0] * (1.f / 65535.f) * (gproperty[2].x - gproperty[1].x) + gproperty[1].x;
            prop[0].y = val.h[1] * (1.f / 65535.f) * (gproperty[2].y - gproperty[1].y) + gproperty[1].y;
            prop[0].w = gproperty[(!(mediaid & MED_MASK)) == 0].w;
        }
#endif

#ifdef __NVCC__
    }

#endif
}

#if MED_TYPE == MEDIA_ASGN_F2H || defined(__NVCC__)
__device__ void updateproperty_asgn(FLOAT4VEC* prop, unsigned int mediaid, uint idx1d,
                                    __global const uint* media, __constant float4* gproperty,
                                    __constant MCXParam* gcfg) {
    if (idx1d == OUTSIDE_VOLUME_MIN || idx1d == OUTSIDE_VOLUME_MAX) {
        *((FLOAT4VEC*)(prop)) = gproperty[0];
        return;
    }

    union {
        unsigned int i[2];
#ifdef __NVCC__
        __half_raw h[4];
#else
        half h[4];
#endif
    } val;

    val.i[0] = mediaid & MED_MASK;
    val.i[1] = media[idx1d + gcfg->dimlen.z];
#ifdef __NVCC__
    prop[0].x = FABS(__half2float(val.h[0]));
    prop[0].y = FABS(__half2float(val.h[1]));
    prop[0].z = FABS(__half2float(val.h[2]));
    prop[0].w = FABS(__half2float(val.h[3]));
#else
    prop[0].x = fabs(convert_float(vload_half(0, val.h)));
    prop[0].y = fabs(convert_float(vload_half(1, val.h)));
    prop[0].z = fabs(convert_float(vload_half(2, val.h)));
    prop[0].w = fabs(convert_float(vload_half(3, val.h)));
#endif
}
#endif

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)

__device__ void updateproperty_svmc(FLOAT4VEC* prop, unsigned int mediaid, uint idx1d,
                                    __global const uint* media, float3 p, MCXsp* nuvox,
                                    short4* flipdir, __constant float4* gproperty,
                                    __constant MCXParam* gcfg) {
    if (idx1d == OUTSIDE_VOLUME_MIN || idx1d == OUTSIDE_VOLUME_MAX) {
        *((FLOAT4VEC*)(prop)) = gproperty[0];
        return;
    }

    union {
        unsigned char c[8];
        unsigned int  i[2];
    } val;

    val.i[0] = media[idx1d + gcfg->dimlen.z];
    val.i[1] = mediaid & MED_MASK;
    unsigned int svpacked = (unsigned int)val.c[7] | ((unsigned int)val.c[6] << 8);

    if (val.c[6]) {
        float3 rp = FLOAT3(val.c[5] * (1.f / 255.f) + (float)flipdir->x,
                           val.c[4] * (1.f / 255.f) + (float)flipdir->y,
                           val.c[3] * (1.f / 255.f) + (float)flipdir->z);
        nuvox->nv = FLOAT3(val.c[2] * (2.f / 255.f) - 1.f,
                           val.c[1] * (2.f / 255.f) - 1.f,
                           val.c[0] * (2.f / 255.f) - 1.f);
        nuvox->nv = nuvox->nv * rsqrt(dot3(nuvox->nv, nuvox->nv));
        nuvox->pd = dot3(rp, nuvox->nv);

        if (dot3(p, nuvox->nv) > nuvox->pd) {
            *((FLOAT4VEC*)(prop)) = gproperty[SV_UPPER(svpacked)];
            SV_SET_ISUPPER(svpacked, 1);
            nuvox->nv.x = -nuvox->nv.x;
            nuvox->nv.y = -nuvox->nv.y;
            nuvox->nv.z = -nuvox->nv.z;
            nuvox->pd = -nuvox->pd;
        } else {
            *((FLOAT4VEC*)(prop)) = gproperty[SV_LOWER(svpacked)];
            SV_SET_ISUPPER(svpacked, 0);
        }

        SV_SET_ISSPLIT(svpacked, 1);
    } else {
        *((FLOAT4VEC*)(prop)) = gproperty[val.c[7]];
        svpacked &= 0xFFFF;
    }

    nuvox->sv = svpacked;
}

__device__ int ray_plane_intersect(float3 p0, float4* v, FLOAT4VEC* prop, float* len,
                                   float* slen, MCXsp* nuvox, __constant MCXParam* gcfg) {
    float3 vdir = FLOAT3(v->x, v->y, v->z);
    float vdotn = dot3(vdir, nuvox->nv);

    if (vdotn <= 0.f) {
        return 0;
    } else {
        float d0 = dot3(p0, nuvox->nv) - nuvox->pd;
        float d1 = d0 + (*len) * vdotn;

        if (d0 * d1 > 0.f) {
            return 0;
        } else {
            float len0 = native_divide((*len) * d0, d0 - d1);
            *len = (len0 > 0.f) ? len0 : *len;
            *slen = (*len) * prop->y * (v->w + 1.f > GPU_PARAM(gcfg, gscatter) ? (1.f - prop->z) : 1.f);
            return 1;
        }
    }
}

__device__ int reflectray_svmc(float n1, float3* c0, MCXsp* nuvox, FLOAT4VEC* prop,
                               __private RandType t[RAND_BUF_LEN], __constant float4* gproperty) {
    float Icos, Re, Im, Rtotal, tmp0, tmp1, tmp2, n2;
    Icos = FABS(dot3(*c0, nuvox->nv));
    n2 = SV_ISUPPER(nuvox->sv) ? gproperty[SV_UPPER(nuvox->sv)].w : gproperty[SV_LOWER(nuvox->sv)].w;
    tmp0 = n1 * n1;
    tmp1 = n2 * n2;
    tmp2 = 1.f - tmp0 / tmp1 * (1.f - Icos * Icos);

    if (tmp2 > 0.f) {
        Re = tmp0 * Icos * Icos + tmp1 * tmp2;
        tmp2 = MCX_MATHFUN(sqrt)(tmp2);
        Im = 2.f * n1 * n2 * Icos * tmp2;
        Rtotal = (Re - Im) / (Re + Im);
        Re = tmp1 * Icos * Icos + tmp0 * tmp2 * tmp2;
        Rtotal = (Rtotal + (Re - Im) / (Re + Im)) * 0.5f;

        if (rand_next_reflect(t) <= Rtotal) {
            *c0 += FL3(-2.f * Icos) * nuvox->nv;
            SV_FLIP_ISUPPER(nuvox->sv);
        } else {
            *c0 += FL3(-Icos) * nuvox->nv;
            *c0 = FL3(tmp2) * nuvox->nv + FL3(n1 / n2) * (*c0);
            nuvox->nv.x = -nuvox->nv.x;
            nuvox->nv.y = -nuvox->nv.y;
            nuvox->nv.z = -nuvox->nv.z;
            nuvox->pd = -nuvox->pd;

            if (SV_CURLABEL(nuvox->sv) == 0) {
                return 1;
            }

            *((FLOAT4VEC*)prop) = gproperty[SV_CURLABEL(nuvox->sv)];
        }
    } else {
        *c0 += FL3(-2.f * Icos) * nuvox->nv;
        SV_FLIP_ISUPPER(nuvox->sv);
    }

    tmp0 = rsqrt(dot3(*c0, *c0));
    *c0 = (*c0) * FL3(tmp0);
    return 0;
}
#endif

/*=== skipvoid: advance photon through background voxels ===*/
#ifndef INTERNAL_SOURCE

__device__ int skipvoid(float4* p, float4* v, float4* f, short4* flipdir, __global const uint* media, __constant float4* gproperty, __constant MCXParam* gcfg, MCXsp* nuvox) {
    int count = 1, idx1d;

    flipdir->x = (short)FLOOR(p->x);
    flipdir->y = (short)FLOOR(p->y);
    flipdir->z = (short)FLOOR(p->z);
    flipdir->w = -1;

    while (1) {
        if ((ushort)flipdir->x < gcfg->maxidx.x && (ushort)flipdir->y < gcfg->maxidx.y && (ushort)flipdir->z < gcfg->maxidx.z) {
            idx1d = (flipdir->z * gcfg->dimlen.y + flipdir->y * gcfg->dimlen.x + flipdir->x);

            if (media[idx1d] & MED_MASK) {
                GPUDEBUG(("inside volume [%f %f %f] v=<%f %f %f>\n", p[0].x, p[0].y, p[0].z, v[0].x, v[0].y, v[0].z));
                p[0].x -= v[0].x;
                p[0].y -= v[0].y;
                p[0].z -= v[0].z;
                flipdir->x = (short)FLOOR(p->x);
                flipdir->y = (short)FLOOR(p->y);
                flipdir->z = (short)FLOOR(p->z);
                f[0].y -= GPU_PARAM(gcfg, minaccumtime);
                idx1d = (flipdir->z * gcfg->dimlen.y + flipdir->y * gcfg->dimlen.x + flipdir->x);
                count = 0;

                while (!((ushort)flipdir->x < gcfg->maxidx.x && (ushort)flipdir->y < gcfg->maxidx.y && (ushort)flipdir->z < gcfg->maxidx.z) || !(media[idx1d] & MED_MASK)) {
                    float dist = hitgrid(p, v, flipdir);
                    f[0].y += GPU_PARAM(gcfg, minaccumtime) * dist;
                    p[0] = FLOAT4(p->x + dist * v->x, p->y + dist * v->y, p->z + dist * v->z, p[0].w);

                    if (flipdir->w == 0) {
                        flipdir->x += (v->x > 0.f ? 1 : -1);
                    }

                    if (flipdir->w == 1) {
                        flipdir->y += (v->y > 0.f ? 1 : -1);
                    }

                    if (flipdir->w == 2) {
                        flipdir->z += (v->z > 0.f ? 1 : -1);
                    }

                    idx1d = (flipdir->z * gcfg->dimlen.y + flipdir->y * gcfg->dimlen.x + flipdir->x);
                    GPUDEBUG(("entry p=[%f %f %f] flipdir=%d\n", p[0].x, p[0].y, p[0].z, flipdir->w));

                    if (count++ > 3) {
                        GPUDEBUG(("fail to find entry point after 3 iterations, abort!!"));
                        break;
                    }
                }

                FLOAT4VEC htime;
                f[0].y = (GPU_PARAM(gcfg, voidtime)) ? f[0].y : 0.f;
#ifdef __NVCC__

                if (gcfg->mediaformat == MEDIA_ASGN_F2H) {
                    updateproperty_asgn(&htime, media[idx1d], idx1d, media, gproperty, gcfg);
                } else if (gcfg->mediaformat == MEDIA_2LABEL_SPLIT) {
                    updateproperty_svmc(&htime, media[idx1d], idx1d, media,
                                        FLOAT3(p->x, p->y, p->z), nuvox, flipdir, gproperty, gcfg);
                } else {
                    updateproperty(&htime, media[idx1d], gproperty, gcfg);
                }

#else
#if MED_TYPE == MEDIA_ASGN_F2H
                updateproperty_asgn(&htime, media[idx1d], idx1d, media, gproperty, gcfg);
#elif MED_TYPE == MEDIA_2LABEL_SPLIT
                updateproperty_svmc(&htime, media[idx1d], idx1d, media,
                                    FLOAT3(p->x, p->y, p->z), nuvox, flipdir, gproperty, gcfg);
#else
                updateproperty(&htime, media[idx1d], gproperty, gcfg);
#endif
#endif

                if (GPU_PARAM(gcfg, isspecular) && htime.w != gproperty[0].w) {
                    p[0].w *= 1.f - reflectcoeff(v, gproperty[0].w, gproperty[media[idx1d] & MED_MASK].w, flipdir->w);
                    GPUDEBUG(("transmitted intensity w=%e\n", p[0].w));

                    if (p[0].w > EPS) {
                        transmit(v, gproperty[0].w, gproperty[media[idx1d] & MED_MASK].w, flipdir->w);
                        GPUDEBUG(("transmit into volume v=<%f %f %f>\n", v[0].x, v[0].y, v[0].z));
                    }
                }

                return idx1d;
            }
        }

        if ( ((p[0].x < 0.f) && (v[0].x <= 0.f)) || ((p[0].x >= gcfg->maxidx.x) && (v[0].x >= 0.f))
                || ((p[0].y < 0.f) && (v[0].y <= 0.f)) || ((p[0].y >= gcfg->maxidx.y) && (v[0].y >= 0.f))
                || ((p[0].z < 0.f) && (v[0].z <= 0.f)) || ((p[0].z >= gcfg->maxidx.z) && (v[0].z >= 0.f))) {
            return -1;
        }

        p[0] = FLOAT4(p[0].x + v[0].x, p[0].y + v[0].y, p[0].z + v[0].z, p[0].w);
        flipdir->x = (short)FLOOR(p->x);
        flipdir->y = (short)FLOOR(p->y);
        flipdir->z = (short)FLOOR(p->z);
        GPUDEBUG(("inside void [%f %f %f]\n", p[0].x, p[0].y, p[0].z));
        f[0].y += GPU_PARAM(gcfg, minaccumtime);

        if ((uint)count++ > GPU_PARAM(gcfg, maxvoidstep)) {
            return -1;
        }
    }
}
#endif

/**
 * @brief Terminate a photon and launch a new photon according to specified source form
 */

#ifdef __NVCC__
    template <const int ispencil, const int isreflect, const int islabel,
    const int issvmc, const int ispolarized>
#endif
__device__ int launchnewphoton(float4* p, float4* v, Stokes* s, float4* f, short4* flipdir, FLOAT4VEC* prop, uint* idx1d,
                               __global float* field, uint* mediaid, float* w0, float* Lmove, uint isdet,
                               __local float* ppath, __global float* n_det, __global uint* dpnum,
                               __private RandType t[RAND_BUF_LEN], __global RandType* rngseed,
                               __constant float4* gproperty, __global const uint* media, __global float* srcpattern,
                               __constant float4* gdetpos, __constant MCXParam* gcfg, int threadid,
                               __local int* blockphoton, volatile __global uint* gprogress,
                               __local RandType* photonseed, __global RandType* gseeddata,
                               __global uint* gjumpdebug, __global float* gdebugdata, __local RandType* sharedmem,
                               MCXsp* nuvox, __global float* photontof) {

    *w0 = 1.f;
    *Lmove = -1.f;
    __constant MCXSrc* launchsrc = &(gcfg->src);

    /**
     * Early termination when detphoton buffer is filled
     */
    if (GPU_PARAM(gcfg, savedet) == FILL_MAXDETPHOTON) {
        if (*dpnum >= GPU_PARAM(gcfg, maxdetphoton)) {
            gprogress[0] = (gcfg->threadphoton >> 1) * 4.5f;
            return 1;
        }
    }

    /**
     * Terminate current photon, perform detection
     */
    if (FABS(p[0].w) >= 0.f) {
        ppath[GPU_PARAM(gcfg, partialdata)] += p[0].w;

        if (GPU_PARAM(gcfg, debuglevel) & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
            if (GPU_PARAM(gcfg, maxpolmedia) > 0 && GPU_PARAM(gcfg, istrajstokes)) {
                savedebugstokes(p, s, (uint)f[0].w + threadid * gcfg->threadphoton + min(threadid, gcfg->oddphoton), gjumpdebug, gdebugdata, gcfg, (int)ppath[GPU_PARAM(gcfg, w0offset) - 1]);
            } else {
                savedebugdata(p, (uint)f[0].w + threadid * gcfg->threadphoton + min(threadid, gcfg->oddphoton), gjumpdebug, gdebugdata, gcfg, (int)ppath[GPU_PARAM(gcfg, w0offset) - 1]);
            }
        }

        if (*mediaid == 0 && *idx1d != OUTSIDE_VOLUME_MIN && *idx1d != OUTSIDE_VOLUME_MAX && GPU_PARAM(gcfg, issaveref) && p[0].w > 0.f) {
            if (GPU_PARAM(gcfg, issaveref) == 1) {
                int tshift = MIN((int)GPU_PARAM(gcfg, maxgate) - 1, (int)(FLOOR((f[0].y - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep))));

                if (GPU_PARAM(gcfg, extrasrclen) * (GPU_PARAM(gcfg, srcid) < 0)) {
                    tshift += ((int)ppath[GPU_PARAM(gcfg, w0offset) - 1] - 1) * (int)GPU_PARAM(gcfg, maxgate);
                }

#if !defined(MCX_SRC_PATTERN) && !defined(MCX_SRC_PATTERN3D)
#ifdef USE_ATOMIC
                float oldval = atomicadd(field + *idx1d + tshift * gcfg->dimlen.z, -p[0].w);

                if (FABS(oldval) > MAX_ACCUM) {
                    atomicadd(field + *idx1d + tshift * gcfg->dimlen.z, ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                    atomicadd(field + *idx1d + tshift * gcfg->dimlen.z + gcfg->dimlen.w, ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                }

#else
                field[*idx1d + tshift * gcfg->dimlen.z] += -p[0].w;
#endif
#else

                for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                    if (FABS(ppath[GPU_PARAM(gcfg, w0offset) + i]) > 0.f) {
#ifdef USE_ATOMIC
                        float oldval = atomicadd(field + (*idx1d + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i, -((GPU_PARAM(gcfg, srcnum) == 1) ? p[0].w : p[0].w * ppath[GPU_PARAM(gcfg, w0offset) + i]));

                        if (FABS(oldval) > MAX_ACCUM) {
                            atomicadd(field + (*idx1d + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i, ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                            atomicadd(field + (*idx1d + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i + gcfg->dimlen.w, ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                        }

#else
                        field[(*idx1d + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i] += -((GPU_PARAM(gcfg, srcnum) == 1) ? p[0].w : p[0].w * ppath[GPU_PARAM(gcfg, w0offset) + i]);
#endif
                    }
                }

#endif
            }

#ifdef MCX_SAVE_DETECTORS
            else {
                saveexitppath(n_det, ppath, p, idx1d, gcfg);
            }

#endif
        }

#ifdef MCX_SAVE_DETECTORS

        if ((isdet & DET_MASK) == DET_MASK && (*mediaid == 0
#if MED_TYPE == MEDIA_2LABEL_SPLIT
                                               || SV_CURLABEL(nuvox->sv) == 0
#endif
                                              ) && (bool)(GPU_PARAM(gcfg, issaveref) < 2)) {
            savedetphoton(n_det, dpnum, ppath, p, v, s, photonseed, gseeddata, gdetpos, gcfg, isdet);
        }

#endif
    }

#ifdef MCX_SAVE_DETECTORS
    clearpath(ppath, GPU_PARAM(gcfg, partialdata));
#endif

#ifdef GROUP_LOAD_BALANCE
    GPUDEBUG(("block workload [%f] done over [%d]\n", f[0].w, blockphoton[0]));

    if (atomic_dec(blockphoton) < 1) {
        return 1;
    }

#else
    GPUDEBUG(("thread workload [%f] done over [%d]\n", f[0].w, (gcfg->threadphoton + (threadid < gcfg->oddphoton))));

    if (f[0].w >= (gcfg->threadphoton + (threadid < gcfg->oddphoton))) {
        return 1;
    }

#endif

    /**
     * For replay mode, initialize RNG with stored seed
     */
    if (GPU_PARAM(gcfg, seed) == SEED_FROM_FILE) {
        int seedoffset = (threadid * gcfg->threadphoton + min(threadid, gcfg->oddphoton - 1) + max(0, (int)f[0].w)) * RAND_BUF_LEN;

        for (int i = 0; i < RAND_BUF_LEN; i++) {
            t[i] = rngseed[seedoffset + i];
        }
    }

    if (GPU_PARAM(gcfg, issaveseed)) {
        copystate(t, photonseed);
    }

    if (GPU_PARAM(gcfg, extrasrclen) * (GPU_PARAM(gcfg, srcid) != 1)) {
        if (GPU_PARAM(gcfg, srcid) > 1) {
            launchsrc = (__constant MCXSrc*)(gproperty + gcfg->maxmedia + 1 + ((GPU_PARAM(gcfg, srcid) - 2) * 4));
        } else {
            ppath[GPU_PARAM(gcfg, w0offset) - 1] = (int)(rand_uniform01(t) * JUST_BELOW_ONE * (GPU_PARAM(gcfg, extrasrclen) + 1)) + 1;

            if ((int)ppath[GPU_PARAM(gcfg, w0offset) - 1] > 1) {
                launchsrc = (__constant MCXSrc*)(gproperty + gcfg->maxmedia + 1 + ((int)(ppath[GPU_PARAM(gcfg, w0offset) - 1] - 2) * 4));
            }
        }
    }

    if ((GPU_PARAM(gcfg, seed) == SEED_FROM_FILE) * (GPU_PARAM(gcfg, srcid) >= 1)) {
        rand_uniform01(t);
    }

    ppath += GPU_PARAM(gcfg, partialdata);

    /**
     * Attempt to launch a new photon until success
     */
    do {
        p[0] = launchsrc->pos;
        v[0] = launchsrc->dir;
        f[0] = FLOAT4(0.f, 0.f, GPU_PARAM(gcfg, minaccumtime), f[0].w);
        *idx1d = AS_UINT(launchsrc->param2.z);
        *mediaid = AS_UINT(launchsrc->param2.w);

        if (gcfg->maxpolmedia > 0) {
            s->i = gcfg->s0.x;
            s->q = gcfg->s0.y;
            s->u = gcfg->s0.z;
            s->v = gcfg->s0.w;
        }

#if MED_TYPE == MEDIA_2LABEL_SPLIT
        SV_CLEAR(nuvox->sv);
#endif

        *prop = TOFLOAT4(FLOAT4(launchsrc->pos.x, launchsrc->pos.y, launchsrc->pos.z, 0));

        /* --- Pencil beam (no position change needed) --- */
#if defined(__NVCC__) || defined(MCX_SRC_PENCIL)
#ifdef __NVCC__

        if (ispencil) {
#endif
            /* pencil beam: position and direction already set from launchsrc */
#endif

            /* --- Planar / Pattern / Pattern3D / Fourier / PencilArray --- */
#if defined(__NVCC__) || defined(MCX_SRC_PLANAR) || defined(MCX_SRC_PATTERN) || defined(MCX_SRC_PATTERN3D) || defined(MCX_SRC_FOURIER) || defined(MCX_SRC_PENCILARRAY)
#ifdef __NVCC__
        } else if (gcfg->srctype == MCX_SRC_PLANAR || gcfg->srctype == MCX_SRC_PATTERN
                   || gcfg->srctype == MCX_SRC_PATTERN3D || gcfg->srctype == MCX_SRC_FOURIER
                   || gcfg->srctype == MCX_SRC_PENCILARRAY) {
#endif
            {
                float rx = rand_uniform01(t);
                float ry = rand_uniform01(t);
                float rz = 0.f;

#if defined(__NVCC__) || defined(MCX_SRC_PATTERN3D)
#ifdef __NVCC__

                if (gcfg->srctype == MCX_SRC_PATTERN3D) {
#endif
                    rz = rand_uniform01(t);
                    p[0] = FLOAT4(p[0].x + rx * launchsrc->param1.x,
                                  p[0].y + ry * launchsrc->param1.y,
                                  p[0].z + rz * launchsrc->param1.z,
                                  p[0].w);
#ifdef __NVCC__
                } else
#endif
#endif
                {
                    p[0] = FLOAT4(p[0].x + rx * launchsrc->param1.x + ry * launchsrc->param2.x,
                                  p[0].y + rx * launchsrc->param1.y + ry * launchsrc->param2.y,
                                  p[0].z + rx * launchsrc->param1.z + ry * launchsrc->param2.z,
                                  p[0].w);
                }

#if defined(__NVCC__) || defined(MCX_SRC_PATTERN)
#ifdef __NVCC__

                if (gcfg->srctype == MCX_SRC_PATTERN) {
#endif

                    if (GPU_PARAM(gcfg, srcnum) <= 1) {
                        p[0].w = launchsrc->pos.w * srcpattern[(int)(ry * JUST_BELOW_ONE * launchsrc->param2.w) * (int)(launchsrc->param1.w) + (int)(rx * JUST_BELOW_ONE * launchsrc->param1.w)];
                        ppath[4] = p[0].w;
                    } else {
                        *((__local uint*)(ppath + 2)) = ((int)(ry * JUST_BELOW_ONE * launchsrc->param2.w) * (int)(launchsrc->param1.w) + (int)(rx * JUST_BELOW_ONE * launchsrc->param1.w));

                        for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                            ppath[i + 4] = srcpattern[(*((__local uint*)(ppath + 2))) * GPU_PARAM(gcfg, srcnum) + i];
                        }

                        p[0].w = 1.f;
                    }

#endif

#if defined(__NVCC__) || defined(MCX_SRC_PATTERN3D)
#ifdef __NVCC__
                } else if (gcfg->srctype == MCX_SRC_PATTERN3D) {
#endif

                    if (GPU_PARAM(gcfg, srcnum) <= 1) {
                        p[0].w = launchsrc->pos.w * srcpattern[(int)(rz * JUST_BELOW_ONE * launchsrc->param1.z) * (int)(launchsrc->param1.y) * (int)(launchsrc->param1.x) +
                                                               (int)(ry * JUST_BELOW_ONE * launchsrc->param1.y) * (int)(launchsrc->param1.x) + (int)(rx * JUST_BELOW_ONE * launchsrc->param1.x)];
                        ppath[4] = p[0].w;
                    } else {
                        *((__local uint*)(ppath + 2)) = ((int)(rz * JUST_BELOW_ONE * launchsrc->param1.z) * (int)(launchsrc->param1.y) * (int)(launchsrc->param1.x) +
                                                         (int)(ry * JUST_BELOW_ONE * launchsrc->param1.y) * (int)(launchsrc->param1.x) + (int)(rx * JUST_BELOW_ONE * launchsrc->param1.x));

                        for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                            ppath[i + 4] = srcpattern[(*((__local uint*)(ppath + 2))) * GPU_PARAM(gcfg, srcnum) + i];
                        }

                        p[0].w = 1.f;
                    }

#endif

#if defined(__NVCC__) || defined(MCX_SRC_FOURIER)
#ifdef __NVCC__
                } else if (gcfg->srctype == MCX_SRC_FOURIER) {
#endif
                    p[0].w = launchsrc->pos.w * (MCX_MATHFUN(cos)((FLOOR(launchsrc->param1.w) * rx + FLOOR(launchsrc->param2.w) * ry
                                                 + launchsrc->param1.w - FLOOR(launchsrc->param1.w)) * TWO_PI) * (1.f - launchsrc->param2.w + FLOOR(launchsrc->param2.w)) + 1.f) * 0.5f;
#endif

#if defined(__NVCC__) || defined(MCX_SRC_PENCILARRAY)
#ifdef __NVCC__
                } else if (gcfg->srctype == MCX_SRC_PENCILARRAY) {
#endif
                    p[0].x = launchsrc->pos.x + FLOOR(rx * launchsrc->param1.w) * launchsrc->param1.x / (launchsrc->param1.w - 1.f) + FLOOR(ry * launchsrc->param2.w) * launchsrc->param2.x / (launchsrc->param2.w - 1.f);
                    p[0].y = launchsrc->pos.y + FLOOR(rx * launchsrc->param1.w) * launchsrc->param1.y / (launchsrc->param1.w - 1.f) + FLOOR(ry * launchsrc->param2.w) * launchsrc->param2.y / (launchsrc->param2.w - 1.f);
                    p[0].z = launchsrc->pos.z + FLOOR(rx * launchsrc->param1.w) * launchsrc->param1.z / (launchsrc->param1.w - 1.f) + FLOOR(ry * launchsrc->param2.w) * launchsrc->param2.z / (launchsrc->param2.w - 1.f);
#endif

#ifdef __NVCC__
                }

#endif
                * idx1d = ((int)(FLOOR(p[0].z)) * gcfg->dimlen.y + (int)(FLOOR(p[0].y)) * gcfg->dimlen.x + (int)(FLOOR(p[0].x)));

                if (p[0].x < 0.f || p[0].y < 0.f || p[0].z < 0.f || p[0].x >= gcfg->maxidx.x || p[0].y >= gcfg->maxidx.y || p[0].z >= gcfg->maxidx.z) {
                    *mediaid = 0;
                } else {
                    *mediaid = media[*idx1d];
                }

                * prop = TOFLOAT4(FLOAT4(prop[0].x + (launchsrc->param1.x + launchsrc->param2.x) * 0.5f,
                                         prop[0].y + (launchsrc->param1.y + launchsrc->param2.y) * 0.5f,
                                         prop[0].z + (launchsrc->param1.z + launchsrc->param2.z) * 0.5f, 0.f));
            }
#endif /* MCX_SRC_PLANAR family */

            /* --- FourierX / FourierX2D --- */
#if defined(__NVCC__) || defined(MCX_SRC_FOURIERX) || defined(MCX_SRC_FOURIERX2D)
#ifdef __NVCC__
        } else if (gcfg->srctype == MCX_SRC_FOURIERX || gcfg->srctype == MCX_SRC_FOURIERX2D) {
#endif
            {
                float rx = rand_uniform01(t);
                float ry = rand_uniform01(t);
                float4 v2 = launchsrc->param1;
                v2.w *= rsqrt(launchsrc->param1.x * launchsrc->param1.x + launchsrc->param1.y * launchsrc->param1.y + launchsrc->param1.z * launchsrc->param1.z);
                v2.x = v2.w * (launchsrc->dir.y * launchsrc->param1.z - launchsrc->dir.z * launchsrc->param1.y);
                v2.y = v2.w * (launchsrc->dir.z * launchsrc->param1.x - launchsrc->dir.x * launchsrc->param1.z);
                v2.z = v2.w * (launchsrc->dir.x * launchsrc->param1.y - launchsrc->dir.y * launchsrc->param1.x);
                p[0] = FLOAT4(p[0].x + rx * launchsrc->param1.x + ry * v2.x,
                              p[0].y + rx * launchsrc->param1.y + ry * v2.y,
                              p[0].z + rx * launchsrc->param1.z + ry * v2.z,
                              p[0].w);
#if defined(__NVCC__) || defined(MCX_SRC_FOURIERX2D)
#ifdef __NVCC__

                if (gcfg->srctype == MCX_SRC_FOURIERX2D) {
#endif
                    p[0].w = launchsrc->pos.w * (MCX_MATHFUN(sin)((launchsrc->param2.x * rx + launchsrc->param2.z) * TWO_PI) * MCX_MATHFUN(sin)((launchsrc->param2.y * ry + launchsrc->param2.w) * TWO_PI) + 1.f) * 0.5f;
#endif
#if defined(__NVCC__) || defined(MCX_SRC_FOURIERX)
#ifdef __NVCC__
                } else {
#else
#endif
                    p[0].w = launchsrc->pos.w * (MCX_MATHFUN(cos)((launchsrc->param2.x * rx + launchsrc->param2.y * ry + launchsrc->param2.z) * TWO_PI) * (1.f - launchsrc->param2.w) + 1.f) * 0.5f;
#ifdef __NVCC__
                }

#endif
#endif

                * idx1d = ((int)(FLOOR(p[0].z)) * gcfg->dimlen.y + (int)(FLOOR(p[0].y)) * gcfg->dimlen.x + (int)(FLOOR(p[0].x)));

                if (p[0].x < 0.f || p[0].y < 0.f || p[0].z < 0.f || p[0].x >= gcfg->maxidx.x || p[0].y >= gcfg->maxidx.y || p[0].z >= gcfg->maxidx.z) {
                    *mediaid = 0;
                } else {
                    *mediaid = media[*idx1d];
                }

                * prop = TOFLOAT4(FLOAT4(prop[0].x + (launchsrc->param1.x + v2.x) * 0.5f,
                                         prop[0].y + (launchsrc->param1.y + v2.y) * 0.5f,
                                         prop[0].z + (launchsrc->param1.z + v2.z) * 0.5f, 0.f));
            }
#endif /* MCX_SRC_FOURIERX family */

            /* --- Hyperboloid Gaussian --- */
#if defined(__NVCC__) || defined(MCX_SRC_HYPERBOLOID_GAUSSIAN)
#ifdef __NVCC__
        } else if (gcfg->srctype == MCX_SRC_HYPERBOLOID_GAUSSIAN) {
#endif
            {
                float sphi, cphi;
                float r = TWO_PI * rand_uniform01(t);
                MCX_SINCOS(r, sphi, cphi);

                r = MCX_MATHFUN(sqrt)(0.5f * rand_next_scatlen(t)) * launchsrc->param1.x;

                prop[0].x = -launchsrc->param1.y / launchsrc->param1.z;
                prop[0].y = rsqrt(r * r + launchsrc->param1.z * launchsrc->param1.z);

                p[0] = FLOAT4(r * (cphi - prop[0].x * sphi), r * (sphi + prop[0].x * cphi), 0.f, p[0].w);
                * prop = TOFLOAT4(FLOAT4(-r* sphi* prop[0].y, r* cphi* prop[0].y, launchsrc->param1.z* prop[0].y, 0.f));

                if ( v[0].z > -1.f + EPS && v[0].z < 1.f - EPS ) {
                    r = 1.f - v[0].z * v[0].z;
                    float stheta = MCX_MATHFUN(sqrt)(r);
                    r = rsqrt(r);
                    cphi = v[0].x * r;
                    sphi = v[0].y * r;
                    p[0] = FLOAT4(p[0].x * cphi * v[0].z - p[0].y * sphi, p[0].x * sphi * v[0].z + p[0].y * cphi, -p[0].x * stheta, p[0].w);
                    v[0] = FLOAT4(prop[0].x * cphi * v[0].z - prop[0].y * sphi + prop[0].z * cphi * stheta,
                                  prop[0].x * sphi * v[0].z + prop[0].y * cphi + prop[0].z * sphi * stheta,
                                  -prop[0].x * stheta + prop[0].z * v[0].z,
                                  v[0].w);
                } else {
                    *((float4*)v) = FLOAT4(prop[0].x, prop[0].y, (v[0].z > 0.f) ? prop[0].z : -prop[0].z, v[0].w);
                }

                p[0] = FLOAT4(p[0].x + launchsrc->pos.x, p[0].y + launchsrc->pos.y, p[0].z + launchsrc->pos.z, p[0].w);
                * prop = TOFLOAT4(FLOAT4(launchsrc->pos.x, launchsrc->pos.y, launchsrc->pos.z, 0));
                * Lmove = 0.f;
            }
#endif /* MCX_SRC_HYPERBOLOID_GAUSSIAN */

            /* --- Disk / Gaussian / Ring --- */
#if defined(__NVCC__) || defined(MCX_SRC_DISK) || defined(MCX_SRC_GAUSSIAN) || defined(MCX_SRC_RING)
#ifdef __NVCC__
        } else if (gcfg->srctype == MCX_SRC_DISK || gcfg->srctype == MCX_SRC_GAUSSIAN
                   || gcfg->srctype == MCX_SRC_RING) {
#endif
            {
                float sphi, cphi, phi;
#if defined(__NVCC__) || defined(MCX_SRC_DISK) || defined(MCX_SRC_RING)
#ifdef __NVCC__

                if (gcfg->srctype == MCX_SRC_DISK || gcfg->srctype == MCX_SRC_RING) {
#endif

                    if (launchsrc->param1.z > 0.f || launchsrc->param1.w > 0.f) {
                        phi = FABS(launchsrc->param1.z - launchsrc->param1.w) * rand_uniform01(t) + fmin(launchsrc->param1.z, launchsrc->param1.w);
                    } else {
                        phi = TWO_PI * rand_uniform01(t);
                    }

#endif
#if defined(__NVCC__) || defined(MCX_SRC_GAUSSIAN)
#ifdef __NVCC__
                } else {
#else
#endif
                    phi = TWO_PI * rand_uniform01(t);
#ifdef __NVCC__
                }

#endif
#endif
                MCX_SINCOS(phi, sphi, cphi);
                float r;
#if defined(__NVCC__) || defined(MCX_SRC_DISK) || defined(MCX_SRC_RING)
#ifdef __NVCC__

                if (gcfg->srctype == MCX_SRC_DISK || gcfg->srctype == MCX_SRC_RING) {
#endif
                    r = MCX_MATHFUN(sqrt)(rand_uniform01(t) * FABS(launchsrc->param1.x * launchsrc->param1.x - launchsrc->param1.y * launchsrc->param1.y) + launchsrc->param1.y * launchsrc->param1.y);
#endif
#if defined(__NVCC__) || defined(MCX_SRC_GAUSSIAN)
#ifdef __NVCC__
                } else {
#else
#endif

                    if (FABS(launchsrc->dir.w) < 1e-5f || FABS(launchsrc->param1.y) < 1e-5f) {
                        r = MCX_MATHFUN(sqrt)(-0.5f * MCX_MATHFUN(log)(rand_uniform01(t))) * launchsrc->param1.x;
                    } else {
                        r = launchsrc->param1.x * launchsrc->param1.x * ONE_PI / launchsrc->param1.y;
                        r = MCX_MATHFUN(sqrt)(-0.5f * MCX_MATHFUN(log)(rand_uniform01(t)) * (1.f + (launchsrc->dir.w * launchsrc->dir.w / (r * r)))) * launchsrc->param1.x;
                    }

#ifdef __NVCC__
                }

#endif
#endif

                if ( v[0].z > -1.f + EPS && v[0].z < 1.f - EPS ) {
                    float tmp0 = 1.f - v[0].z * v[0].z;
                    float tmp1 = r * rsqrt(tmp0);
                    p[0] = FLOAT4(
                               p[0].x + tmp1 * (v[0].x * v[0].z * cphi - v[0].y * sphi),
                               p[0].y + tmp1 * (v[0].y * v[0].z * cphi + v[0].x * sphi),
                               p[0].z - tmp1 * tmp0 * cphi,
                               p[0].w);
                } else {
                    p[0].x += r * cphi;
                    p[0].y += r * sphi;
                }

                * idx1d = ((int)(FLOOR(p[0].z)) * gcfg->dimlen.y + (int)(FLOOR(p[0].y)) * gcfg->dimlen.x + (int)(FLOOR(p[0].x)));

                if (p[0].x < 0.f || p[0].y < 0.f || p[0].z < 0.f || p[0].x >= gcfg->maxidx.x || p[0].y >= gcfg->maxidx.y || p[0].z >= gcfg->maxidx.z) {
                    *mediaid = 0;
                } else {
                    *mediaid = media[*idx1d];
                }
            }
#endif /* MCX_SRC_DISK family */

            /* --- Cone / Isotropic / Arcsine --- */
#if defined(__NVCC__) || defined(MCX_SRC_CONE) || defined(MCX_SRC_ISOTROPIC) || defined(MCX_SRC_ARCSINE)
#ifdef __NVCC__
        } else if (gcfg->srctype == MCX_SRC_CONE || gcfg->srctype == MCX_SRC_ISOTROPIC
                   || gcfg->srctype == MCX_SRC_ARCSINE) {
#endif
            {
                float ang, stheta, ctheta, sphi, cphi;
                ang = TWO_PI * rand_uniform01(t);
                MCX_SINCOS(ang, sphi, cphi);
#if defined(__NVCC__) || defined(MCX_SRC_CONE)
#ifdef __NVCC__

                if (gcfg->srctype == MCX_SRC_CONE) {
#endif

                    do {
                        ang = (launchsrc->param1.y > 0) ? TWO_PI * rand_uniform01(t) : ACOS(2.f * rand_uniform01(t) - 1.f);
                    } while (ang > launchsrc->param1.x);

#endif
#if defined(__NVCC__) || defined(MCX_SRC_ISOTROPIC) || defined(MCX_SRC_ARCSINE)
#ifdef __NVCC__
                } else {
#else
#endif

#if defined(__NVCC__) || defined(MCX_SRC_ISOTROPIC)
                    ang = ACOS(2.f * rand_uniform01(t) - 1.f);
#else
                    ang = ONE_PI * rand_uniform01(t);
#endif

#ifdef __NVCC__
                }

#endif
#endif
                MCX_SINCOS(ang, stheta, ctheta);
                rotatevector(v, stheta, ctheta, sphi, cphi);
                * Lmove = 0.f;
            }
#endif /* MCX_SRC_CONE family */

            /* --- ZGaussian --- */
#if defined(__NVCC__) || defined(MCX_SRC_ZGAUSSIAN)
#ifdef __NVCC__
        } else if (gcfg->srctype == MCX_SRC_ZGAUSSIAN) {
#endif
            {
                float ang, stheta, ctheta, sphi, cphi;
                ang = TWO_PI * rand_uniform01(t);
                MCX_SINCOS(ang, sphi, cphi);
                ang = MCX_MATHFUN(sqrt)(-2.f * MCX_MATHFUN(log)(rand_uniform01(t))) * (1.f - 2.f * rand_uniform01(t)) * launchsrc->param1.x;
                MCX_SINCOS(ang, stheta, ctheta);
                rotatevector(v, stheta, ctheta, sphi, cphi);
                * Lmove = 0.f;
            }
#endif /* MCX_SRC_ZGAUSSIAN */

            /* --- Line / Slit --- */
#if defined(__NVCC__) || defined(MCX_SRC_LINE) || defined(MCX_SRC_SLIT)
#ifdef __NVCC__
        } else if (gcfg->srctype == MCX_SRC_LINE || gcfg->srctype == MCX_SRC_SLIT) {
#endif
            {
                float r_l = rand_uniform01(t);
                p[0] = FLOAT4(p[0].x + r_l * launchsrc->param1.x, p[0].y + r_l * launchsrc->param1.y,
                              p[0].z + r_l * launchsrc->param1.z, p[0].w);

#if defined(__NVCC__) || defined(MCX_SRC_LINE)
#ifdef __NVCC__

                if (gcfg->srctype == MCX_SRC_LINE) {
#endif
                    {
                        float sphi_l, cphi_l;
                        r_l = rsqrt(launchsrc->param1.x * launchsrc->param1.x + launchsrc->param1.y * launchsrc->param1.y + launchsrc->param1.z * launchsrc->param1.z);

                        if (launchsrc->param2.x > 0.f) {
                            float3 lineaxis = FLOAT3(launchsrc->param1.x * r_l, launchsrc->param1.y * r_l, launchsrc->param1.z * r_l);
                            float vdotaxis = v[0].x * lineaxis.x + v[0].y * lineaxis.y + v[0].z * lineaxis.z;
                            v[0] = FLOAT4(v[0].x - vdotaxis * lineaxis.x, v[0].y - vdotaxis * lineaxis.y, v[0].z - vdotaxis * lineaxis.z, v[0].w);
                            float vnorm = rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z);
                            v[0] = FLOAT4(v[0].x * vnorm, v[0].y * vnorm, v[0].z * vnorm, v[0].w);
                            r_l = launchsrc->param2.x * (2.f * rand_uniform01(t) - 1.f);
                            MCX_SINCOS(r_l, sphi_l, cphi_l);
                            rotate_perpendicular_vector(v, lineaxis, sphi_l, cphi_l);
                        } else {
                            v[0] = FLOAT4(launchsrc->param1.x * r_l, launchsrc->param1.y * r_l, launchsrc->param1.z * r_l, v[0].w);
                            r_l = TWO_PI * rand_uniform01(t);
                            MCX_SINCOS(r_l, sphi_l, cphi_l);
                            rotatevector(v, 1.f, 0.f, sphi_l, cphi_l);
                        }
                    }
                    *Lmove = 0.f;

#endif /* MCX_SRC_LINE */

#if defined(__NVCC__) || defined(MCX_SRC_SLIT)
#ifdef __NVCC__
                } else {
#else
#endif

                    if (launchsrc->param2.x > 0.f || launchsrc->param2.y > 0.f) {
                        float sphi_s, cphi_s;
                        r_l = TWO_PI * rand_uniform01(t);
                        MCX_SINCOS(r_l, sphi_s, cphi_s);
                        r_l = MCX_MATHFUN(sqrt)(2.f * rand_next_scatlen(t));
                        cphi_s *= launchsrc->param2.x * r_l;
                        sphi_s *= launchsrc->param2.y * r_l;
                        sphi_s *= rsqrt(launchsrc->param1.x * launchsrc->param1.x + launchsrc->param1.y * launchsrc->param1.y + launchsrc->param1.z * launchsrc->param1.z);
                        *prop = TOFLOAT4(FLOAT4(launchsrc->param1.y * v[0].z - launchsrc->param1.z * v[0].y,
                                                launchsrc->param1.z * v[0].x - launchsrc->param1.x * v[0].z,
                                                launchsrc->param1.x * v[0].y - launchsrc->param1.y * v[0].x, 0));
                        cphi_s *= rsqrt(prop[0].x * prop[0].x + prop[0].y * prop[0].y + prop[0].z * prop[0].z);
                        v[0].x += cphi_s * prop[0].x + sphi_s * launchsrc->param1.x;
                        v[0].y += cphi_s * prop[0].y + sphi_s * launchsrc->param1.y;
                        v[0].z += cphi_s * prop[0].z + sphi_s * launchsrc->param1.z;
                        r_l = rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z);
                        v[0].x *= r_l;
                        v[0].y *= r_l;
                        v[0].z *= r_l;
                    }

                    *Lmove = -1.f;
#ifdef __NVCC__
                }

#endif
#endif /* MCX_SRC_SLIT */

                *idx1d = ((int)(FLOOR(p[0].z)) * gcfg->dimlen.y + (int)(FLOOR(p[0].y)) * gcfg->dimlen.x + (int)(FLOOR(p[0].x)));

                if (p[0].x < 0.f || p[0].y < 0.f || p[0].z < 0.f || p[0].x >= gcfg->maxidx.x || p[0].y >= gcfg->maxidx.y || p[0].z >= gcfg->maxidx.z) {
                    *mediaid = 0;
                } else {
                    *mediaid = media[*idx1d];
                }

                * prop = TOFLOAT4(FLOAT4(launchsrc->pos.x + (launchsrc->param1.x) * 0.5f,
                                         launchsrc->pos.y + (launchsrc->param1.y) * 0.5f,
                                         launchsrc->pos.z + (launchsrc->param1.z) * 0.5f, 0.f));
                * Lmove = -1.f;
            }
#endif /* MCX_SRC_LINE family */

            /* --- Close the if/else chain for CUDA --- */
#ifdef __NVCC__
        }

#endif

        /**
         * If weight too low, skip
         */
        if (FABS(p[0].w) <= GPU_PARAM(gcfg, minenergy)) {
            continue;
        }

        /**
         * Angular distribution / beam focus
         */
        if (GPU_PARAM(gcfg, nangle)) {
            float ang, stheta, ctheta, sphi, cphi;

            if ( launchsrc->dir.w > 0.f) {
                ang = fmin(rand_uniform01(t) * GPU_PARAM(gcfg, nangle), GPU_PARAM(gcfg, nangle) - EPS);
                cphi = ((__local float*)(sharedmem))[(int)(ang) + GPU_PARAM(gcfg, nphaselen)];
            } else {
                ang = fmin(rand_uniform01(t) * (GPU_PARAM(gcfg, nangle) - 1), GPU_PARAM(gcfg, nangle) - 1 - EPS);
                sphi = ang - ((int)ang);
                cphi = ((1.f - sphi) * (((__local float*)(sharedmem))[((uint)ang >= GPU_PARAM(gcfg, nangle) - 1 ? GPU_PARAM(gcfg, nangle) - 1 : (int)(ang)) + GPU_PARAM(gcfg, nphaselen)]) +
                        sphi * (((__local float*)(sharedmem))[((uint)ang + 1 >= GPU_PARAM(gcfg, nangle) - 1 ? GPU_PARAM(gcfg, nangle) - 1 : (int)(ang) + 1) + GPU_PARAM(gcfg, nphaselen)]));
            }

            cphi *= ONE_PI;
            MCX_SINCOS(cphi, stheta, ctheta);
            ang = TWO_PI * rand_uniform01(t);
            MCX_SINCOS(ang, sphi, cphi);

            if ( launchsrc->dir.w < 1.5f && launchsrc->dir.w >= 0.f) {
                *((float4*)v) = launchsrc->dir;
            }

            rotatevector(v, stheta, ctheta, sphi, cphi);
        } else if (*Lmove < 0.f) {
            if (isnan(launchsrc->dir.w)) {
                float ang, stheta, ctheta, sphi, cphi;
                ang = TWO_PI * rand_uniform01(t);
                MCX_SINCOS(ang, sphi, cphi);
                ang = ACOS(2.f * rand_uniform01(t) - 1.f);
                MCX_SINCOS(ang, stheta, ctheta);
                rotatevector(v, stheta, ctheta, sphi, cphi);
            } else if (launchsrc->dir.w < 0.f && isinf(launchsrc->dir.w)) {
                float ang, stheta, ctheta, sphi, cphi;
                ang = TWO_PI * rand_uniform01(t);
                MCX_SINCOS(ang, sphi, cphi);
                stheta = MCX_MATHFUN(sqrt)(rand_uniform01(t));
                ctheta = MCX_MATHFUN(sqrt)(1.f - stheta * stheta);
                rotatevector(v, stheta, ctheta, sphi, cphi);
            } else if (launchsrc->dir.w != 0.f) {
                float Rn2 = (launchsrc->dir.w > 0.f) - (launchsrc->dir.w < 0.f);
                prop[0].x += launchsrc->dir.w * v[0].x;
                prop[0].y += launchsrc->dir.w * v[0].y;
                prop[0].z += launchsrc->dir.w * v[0].z;
                v[0].x = Rn2 * (prop[0].x - p[0].x);
                v[0].y = Rn2 * (prop[0].y - p[0].y);
                v[0].z = Rn2 * (prop[0].z - p[0].z);
                Rn2 = rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z);
                v[0].x *= Rn2;
                v[0].y *= Rn2;
                v[0].z *= Rn2;
            }
        }

#ifndef INTERNAL_SOURCE

        if ((*mediaid & MED_MASK) == 0) {
            int idx = skipvoid(p, v, f, flipdir, media, gproperty, gcfg, nuvox);

            if (idx >= 0) {
                *idx1d = idx;
                *mediaid = media[*idx1d];
            }
        }

#endif
        flipdir->x = (short)FLOOR(p->x);
        flipdir->y = (short)FLOOR(p->y);
        flipdir->z = (short)FLOOR(p->z);

        *w0 += 1.f;

        if (*w0 > GPU_PARAM(gcfg, maxvoidstep)) {
            return -1;
        }
    } while ((*mediaid & MED_MASK) == 0 || FABS(p[0].w) <= GPU_PARAM(gcfg, minenergy));

    /**
     * Photon successfully launched — initialize for new trajectory
     */
    f[0].w += 1.f;
    *prop = TOFLOAT4(gproperty[1]);
#if MED_TYPE == MEDIA_ASGN_F2H || defined(__NVCC__)
    updateproperty_asgn(prop, *mediaid, *idx1d, media, gproperty, gcfg);
#elif MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
    updateproperty_svmc(prop, *mediaid, *idx1d, media, FLOAT3(p->x, p->y, p->z), nuvox, flipdir, gproperty, gcfg);
#else
    updateproperty(prop, *mediaid, gproperty, gcfg);
#endif

#ifdef __NVCC__

    if (gcfg->mediaformat == MEDIA_ASGN_F2H) {
        updateproperty_asgn(prop, *mediaid, *idx1d, media, gproperty, gcfg);
    } else if (gcfg->mediaformat == MEDIA_2LABEL_SPLIT) {
        updateproperty_svmc(prop, *mediaid, *idx1d, media, FLOAT3(p->x, p->y, p->z), nuvox, flipdir, gproperty, gcfg);

    } else {
        updateproperty(prop, *mediaid, gproperty, gcfg);
    }

#else
#if MED_TYPE == MEDIA_ASGN_F2H
    updateproperty_asgn(prop, *mediaid, *idx1d, media, gproperty, gcfg);
#elif MED_TYPE == MEDIA_2LABEL_SPLIT
    updateproperty_svmc(prop, *mediaid, *idx1d, media, FLOAT3(p->x, p->y, p->z), nuvox, flipdir, gproperty, gcfg);
#else
    updateproperty(prop, *mediaid, gproperty, gcfg);
#endif
#endif

    if (GPU_PARAM(gcfg, debuglevel) & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
        if (GPU_PARAM(gcfg, maxpolmedia) > 0 && GPU_PARAM(gcfg, istrajstokes)) {
            savedebugstokes(p, s, (uint)f[0].w + threadid * gcfg->threadphoton + min(threadid, (threadid < gcfg->oddphoton) * threadid), gjumpdebug, gdebugdata, gcfg, (int)ppath[3]);
        } else {
            savedebugdata(p, (uint)f[0].w + threadid * gcfg->threadphoton + min(threadid, (threadid < gcfg->oddphoton) * threadid), gjumpdebug, gdebugdata, gcfg, (int)ppath[3]);
        }
    }

    ppath[1] += p[0].w;
    *w0 = p[0].w;
    ppath[2] = ((GPU_PARAM(gcfg, srcnum) > 1) ? ppath[2] : p[0].w);
    v[0].w = EPS;
    *Lmove = 0.f;

    if ((GPU_PARAM(gcfg, outputtype) == otRF) | (GPU_PARAM(gcfg, outputtype) == otRFmus)) {
        float rf_tof = photontof[(threadid * gcfg->threadphoton + min(threadid, gcfg->oddphoton - 1) + (int)f[0].w)];
        float rf_sin, rf_cos;
        MCX_SINCOS(GPU_PARAM(gcfg, omega) * rf_tof, rf_sin, rf_cos);
        ppath[GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum)] = rf_cos;
        ppath[GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum) + 1] = rf_sin;
    }

    if (GPU_PARAM(gcfg, debuglevel) & MCX_DEBUG_PROGRESS) {

        if (((int)(f[0].w) & 1) && (threadid == 0 || threadid == (int)(get_global_size(0) - 1) || threadid == (int)(get_global_size(0) >> 2)
                                    || threadid == (int)(get_global_size(0) >> 1) || threadid == (int)(get_global_size(0) >> 2) + (int)(get_global_size(0) >> 1))) {
            gprogress[0]++;
        }

    }

    return 0;
}

/*==========================================================================
 * Main kernel entry point
 *
 * The main simulation loop body is identical between OpenCL and CUDA,
 * since all API differences are abstracted by the macros above.
 * The only structural difference is the kernel signature and shared
 * memory declaration.
 *
 * Under CUDA (__NVCC__), the kernel is a C++ template function for
 * compile-time specialization. This is not shown here to avoid
 * duplicating the entire ~500 line loop body; instead, the CUDA host
 * code (mcx_cu_host.cu) would #include this file and instantiate the
 * template with the appropriate parameters.
 *
 * For the actual main loop body, it is identical to the original
 * mcx_core.cl with the following systematic substitutions already
 * applied throughout this file:
 *   (float4)(...) -> FLOAT4(...)
 *   (float3)(...) -> FLOAT3(...)
 *   fabs(scalar)  -> FABS()
 *   convert_short3_rtn(p->xyz) -> manual (short)FLOOR() per component
 *   convert_float_rte() -> RINT()
 *   native_divide() -> native_divide() (shimmed in compat header)
 *   barrier(CLK_LOCAL_MEM_FENCE) -> barrier() (shimmed to __syncthreads())
 *==========================================================================*/

#ifdef __NVCC__
    template <const int ispencil, const int isreflect, const int islabel,
    const int issvmc, const int ispolarized>
#endif
__kernel void mcx_main_loop(
    __global const uint* media,
    __global float* field, __global float* genergy, __global uint* n_seed,
    __global float* n_det, __constant float4* gproperty, __global float* srcpattern,
    __constant float4* gdetpos, volatile __global uint* gprogress, __global uint* detectedphoton,
    __global float* replayweight, __global float* photontof, __global int* photondetid,
    __global RandType* gseeddata, __global uint* gjumpdebug, __global float* gdebugdata,
    __global float* ginvcdf, __global float* gangleinvcdf, __local RandType* sharedmem,
    __global float4* gsmatrix
#ifndef __NVCC__
    , __constant MCXParam* gcfg
#endif
) {

    int idx = get_global_id(0);

#ifdef __NVCC__
    extern __shared__ char sharedmem_raw[];
    sharedmem = (__local RandType*)sharedmem_raw;
    const MCXParam* gcfg = gcfg_const;
#endif

    float4 p = FLOAT4(0.f, 0.f, 0.f, NAN);
    float4 v = FLOAT4(0.f, 0.f, 0.f, 0.f);
    float4 f = FLOAT4(0.f, 0.f, 0.f, 0.f);

    uint idx1d, idx1dold;
    uint   mediaid = AS_UINT(gcfg->src.param2.w);
    uint   mediaidold = 0;
    uint   isdet = 0;
    float  w0, Lmove, pathlen = 0.f;
    MCXsp nuvox;
    nuvox.sv = 0;
    Stokes s = {1.f, 0.f, 0.f, 0.f};
    unsigned char testint = 0;
    unsigned char hitintf = 0;
    float  n1;
    short4 flipdir = SHORT4(0, 0, 0, -1);

    RandType t[RAND_BUF_LEN];
    FLOAT4VEC prop;

    __local float* ppath = (__local float*)sharedmem;
    __local int   blockphoton[1];

    /*--- Load phase function invcdf to shared memory ---*/
    if (GPU_PARAM(gcfg, nphase)) {
        idx1d = GPU_PARAM(gcfg, nphase) / get_local_size(0);

        for (idx1dold = 0; idx1dold < idx1d; idx1dold++) {
            ppath[get_local_id(0) * idx1d + idx1dold] = ginvcdf[get_local_id(0) * idx1d + idx1dold];
        }

        if (GPU_PARAM(gcfg, nphase) - (idx1d * get_local_size(0)) > 0 && get_local_id(0) == 0)
            for (idx1dold = 0; idx1dold < GPU_PARAM(gcfg, nphase) - (idx1d * get_local_size(0)); idx1dold++) {
                ppath[get_local_size(0) * idx1d + idx1dold] = ginvcdf[get_local_size(0) * idx1d + idx1dold];
            }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /*--- Load angle invcdf to shared memory ---*/
    if (GPU_PARAM(gcfg, nangle)) {
        idx1d = GPU_PARAM(gcfg, nangle) / get_local_size(0);

        for (idx1dold = 0; idx1dold < idx1d; idx1dold++) {
            ppath[get_local_id(0) * idx1d + idx1dold + GPU_PARAM(gcfg, nphaselen)] = gangleinvcdf[get_local_id(0) * idx1d + idx1dold];
        }

        if (GPU_PARAM(gcfg, nangle) - (idx1d * get_local_size(0)) > 0 && get_local_id(0) == 0)
            for (idx1dold = 0; idx1dold < GPU_PARAM(gcfg, nangle) - (idx1d * get_local_size(0)); idx1dold++) {
                ppath[get_local_size(0) * idx1d + idx1dold + GPU_PARAM(gcfg, nphaselen)] = gangleinvcdf[get_local_size(0) * idx1d + idx1dold];
            }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ((uint)idx >= gcfg->threadphoton * (get_local_size(0) * get_num_groups(0)) + gcfg->oddphoton) {
        return;
    }

    ppath = (__local float*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen)) + get_local_size(0) * (GPU_PARAM(gcfg, issaveseed) * RAND_BUF_LEN * sizeof(RandType)));

#ifdef GROUP_LOAD_BALANCE

    if (get_local_id(0) == 0) {
        blockphoton[0] = gcfg->blockphoton + ((int)get_group_id(0) < gcfg->blockextra);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    GPUDEBUG(("set block load to %d\n", blockphoton[0]));
#endif

    ppath += get_local_id(0) * (GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum) + (((GPU_PARAM(gcfg, outputtype) == otRF) | (GPU_PARAM(gcfg, outputtype) == otRFmus)) << 1));
    clearpath(ppath, GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum) + (((GPU_PARAM(gcfg, outputtype) == otRF) | (GPU_PARAM(gcfg, outputtype) == otRFmus)) << 1));
    ppath[GPU_PARAM(gcfg, partialdata)]  = genergy[idx << 1];
    ppath[GPU_PARAM(gcfg, partialdata) + 1] = genergy[(idx << 1) + 1];

    gpu_rng_init(t, n_seed, idx);

    if (GPU_PARAM(gcfg, debuglevel) & MCX_DEBUG_RNG) {
        for (int i = idx; i < gcfg->dimlen.w; i += get_global_size(0)) {
            field[i] = rand_uniform01(t);
        }

        return;
    }

    if (LAUNCHNEWPHOTON(&p, &v, &s, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove, 0, ppath,
                        n_det, detectedphoton, t, (__global RandType*)n_seed, gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton,
                        gprogress, (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen)) + get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)),
                        gseeddata, gjumpdebug, gdebugdata, sharedmem, &nuvox, photontof)) {
        n_seed[idx] = NO_LAUNCH;
        return;
    }

    isdet = mediaid & DET_MASK;
    mediaid &= MED_MASK;

    /*==========================================================================
     * Main photon propagation loop
     *==========================================================================*/

#ifdef GROUP_LOAD_BALANCE

    while (1) {
        GPUDEBUG(("block workload [%d] left\n", blockphoton[0]));
#else

    while (f.w <= gcfg->threadphoton + (idx < gcfg->oddphoton)) {
#endif
        GPUDEBUG(("photonid [%d] L=%f w=%e medium=%d\n", (int)f.w, f.x, p.w, mediaid));

        if (f.x <= 0.f) {
            f.x = rand_next_scatlen(t);
            GPUDEBUG(("scat L=%f RNG=[%e %e %e] \n", f.x, rand_next_aangle(t), rand_next_zangle(t), rand_uniform01(t)));

            if (v.w != EPS) {
                float cphi = 1.f, sphi = 0.f, theta, stheta, ctheta;
                float tmp0 = 0.f;

                if ((GPU_PARAM(gcfg, maxpolmedia) > 0) & (!GPU_PARAM(gcfg, is2d))) {
                    uint ipol = (uint)NANGLES * ((mediaid & MED_MASK) - 1);
                    float I0, I, sin2phi, cos2phi;

                    do {
                        theta = ACOS(2.f * rand_next_zangle(t) - 1.f);
                        tmp0 = TWO_PI * rand_next_aangle(t);
                        MCX_SINCOS(2.f * tmp0, sin2phi, cos2phi);
                        I0 = gsmatrix[ipol].x * s.i + gsmatrix[ipol].y * (s.q * cos2phi + s.u * sin2phi);
                        uint ithedeg = (uint)(theta * NANGLES * (R_PI - EPS));
                        I = gsmatrix[ipol + ithedeg].x * s.i + gsmatrix[ipol + ithedeg].y * (s.q * cos2phi + s.u * sin2phi);
                    } while (rand_uniform01(t) * I0 >= I);

                    MCX_SINCOS(tmp0, sphi, cphi);
                    MCX_SINCOS(theta, stheta, ctheta);
                } else {
                    if (!GPU_PARAM(gcfg, is2d)) {
                        tmp0 = TWO_PI * rand_next_aangle(t);
                        MCX_SINCOS(tmp0, sphi, cphi);
                    }

                    if (GPU_PARAM(gcfg, nphase) > 2) {
                        tmp0 = rand_uniform01(t) * (GPU_PARAM(gcfg, nphase) - 1);
                        theta = tmp0 - ((int)tmp0);
                        tmp0 = (1.f - theta) * ((__local float*)(sharedmem))[(uint)tmp0 >= GPU_PARAM(gcfg, nphase) ? GPU_PARAM(gcfg, nphase) - 1 : (int)(tmp0)] +
                               theta * ((__local float*)(sharedmem))[(uint)tmp0 + 1 >= GPU_PARAM(gcfg, nphase) ? GPU_PARAM(gcfg, nphase) - 1 : (int)(tmp0) + 1];
                        theta = ACOS(tmp0);
                        stheta = MCX_MATHFUN(sin)(theta);
                        ctheta = tmp0;
                    } else {
                        tmp0 = (v.w > GPU_PARAM(gcfg, gscatter)) ? 0.f : prop.z;

                        if (FABS(tmp0) > EPS) {
                            tmp0 = (1.f - prop.z * prop.z) / (1.f - prop.z + 2.f * prop.z * rand_next_zangle(t));
                            tmp0 *= tmp0;
                            tmp0 = (1.f + prop.z * prop.z - tmp0) / (2.f * prop.z);
                            tmp0 = fmax(-1.f, fmin(1.f, tmp0));
                            theta = ACOS(tmp0);
                            stheta = MCX_MATHFUN(sin)(theta);
                            ctheta = tmp0;
                        } else {
                            theta = ACOS(2.f * rand_next_zangle(t) - 1.f);
                            MCX_SINCOS(theta, stheta, ctheta);
                        }
                    }
                }

#ifdef MCX_SAVE_DETECTORS

                if (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag))) {
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

                    if (issvmc) {
#endif

                        if (SV_CURLABEL(nuvox.sv) > 0) {
                            ppath[SV_CURLABEL(nuvox.sv) - 1]++;
                        }

#ifdef __NVCC__
                    } else {
                        ppath[(mediaid & MED_MASK) - 1]++;
                    }

#endif
#else
                    ppath[(mediaid & MED_MASK) - 1]++;
#endif
                }

                if (SAVE_MOM(GPU_PARAM(gcfg, savedetflag))) {
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

                    if (issvmc) {
#endif

                        if (SV_CURLABEL(nuvox.sv) > 0) {
                            ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag)) + SAVE_PPATH(GPU_PARAM(gcfg, savedetflag))) + SV_CURLABEL(nuvox.sv) - 1] += 1.f - ctheta;
                        }

#ifdef __NVCC__
                    } else {
                        ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag)) + SAVE_PPATH(GPU_PARAM(gcfg, savedetflag))) + (mediaid & MED_MASK) - 1] += 1.f - ctheta;
                    }

#endif
#else
                    ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag)) + SAVE_PPATH(GPU_PARAM(gcfg, savedetflag))) + (mediaid & MED_MASK) - 1] += 1.f - ctheta;
#endif
                }

#endif
                float3 olddir;

                if (GPU_PARAM(gcfg, maxpolmedia) > 0) {
                    olddir = FLOAT3(v.x, v.y, v.z);
                }

                if (GPU_PARAM(gcfg, is2d)) {
                    rotatevector2d(&v, (rand_next_aangle(t) > 0.5f ? stheta : -stheta), ctheta, GPU_PARAM(gcfg, is2d));
                } else {
                    rotatevector(&v, stheta, ctheta, sphi, cphi);
                }

                v.w += 1.f;

                if (GPU_PARAM(gcfg, maxpolmedia) > 0) {
                    float3 newdir = FLOAT3(v.x, v.y, v.z);
                    updatestokes(&s, theta, tmp0, &olddir, &newdir, &mediaid, gsmatrix);
                }

                /* WP/DCS/RF replay energy deposit at scattering site */
                if ((bool)(GPU_PARAM(gcfg, outputtype) == otWP) | (bool)(GPU_PARAM(gcfg, outputtype) == otDCS) | (bool)(GPU_PARAM(gcfg, outputtype) == otWPTOF) | ((GPU_PARAM(gcfg, seed) == SEED_FROM_FILE) & (GPU_PARAM(gcfg, outputtype) == otRFmus))) {
                    int tshift = (idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w);
                    tmp0 = (GPU_PARAM(gcfg, outputtype) == otDCS) ? (1.f - ctheta) : 1.f;

                    if (GPU_PARAM(gcfg, outputtype) == otRFmus) {
                        float rf_cos_val = ppath[GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum)];
                        float rf_sin_val = ppath[GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum) + 1];
                        float rw = replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)];
                        tmp0 = rw * rf_cos_val;
                        sphi = rw * rf_sin_val;
                    } else if (GPU_PARAM(gcfg, outputtype) == otWPTOF) {
                        tmp0 = photontof[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)];
                        tmp0 *= replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)];
                    } else {
                        tmp0 *= replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)];
                    }

                    tshift = (int)(FLOOR((photontof[tshift] - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep))) +
                             ( (GPU_PARAM(gcfg, replaydet) == -1) ? (((photondetid[tshift] & 0xFFFF) - 1) * GPU_PARAM(gcfg, maxgate)) : 0);

                    if (GPU_PARAM(gcfg, extrasrclen) * (GPU_PARAM(gcfg, srcid) < 0)) {
                        tshift += ((int)ppath[GPU_PARAM(gcfg, w0offset) - 1] - 1) * ((GPU_PARAM(gcfg, replaydet) == -1) ? GPU_PARAM(gcfg, detnum) : 1) * GPU_PARAM(gcfg, maxgate);
                    }

                    tshift = MIN(GPU_PARAM(gcfg, maxgate) - 1, tshift);

#ifndef USE_ATOMIC
                    field[idx1d + tshift * gcfg->dimlen.z] += tmp0 * replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)];
#else
                    float oldval = atomicadd(field + idx1d + tshift * gcfg->dimlen.z, tmp0 * replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)]);

                    if (FABS(oldval) > MAX_ACCUM) {
                        if (atomicadd(field + idx1d + tshift * gcfg->dimlen.z, -oldval) < 0.f) {
                            atomicadd(field + idx1d + tshift * gcfg->dimlen.z, oldval);
                        } else {
                            atomicadd(field + idx1d + tshift * gcfg->dimlen.z + gcfg->dimlen.w, oldval);
                        }
                    }

                    GPUDEBUG(("atomic write to [%d] %e, w=%f\n", idx1d, tmp0 * replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)], p.w));
#endif
                }

                if (GPU_PARAM(gcfg, debuglevel) & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {

                    if (GPU_PARAM(gcfg, maxpolmedia) > 0 && GPU_PARAM(gcfg, istrajstokes)) {
                        savedebugstokes(&p, &s, (uint)f.w + idx * gcfg->threadphoton + min(idx, (idx < gcfg->oddphoton)*idx), gjumpdebug, gdebugdata, gcfg, (int)ppath[GPU_PARAM(gcfg, w0offset) - 1]);
                    } else {
                        savedebugdata(&p, (uint)f.w + idx * gcfg->threadphoton + min(idx, (idx < gcfg->oddphoton)*idx), gjumpdebug, gdebugdata, gcfg, (int)ppath[GPU_PARAM(gcfg, w0offset) - 1]);
                    }

                }
            }

            v.w = (int)v.w;

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

            if (issvmc) {
#endif
                testint = 1;
#ifdef __NVCC__
            }

#endif
#endif
        }

        /*--- Read optical properties of current voxel ---*/
        n1 = prop.w;

#ifdef __NVCC__

        if (gcfg->mediaformat == MEDIA_ASGN_F2H) {
            updateproperty_asgn(&prop, mediaid, idx1d, media, gproperty, gcfg);
        } else if (gcfg->mediaformat == MEDIA_2LABEL_SPLIT) {
            updateproperty_svmc(&prop, mediaid, idx1d, media,
                                FLOAT3(p.x, p.y, p.z), &nuvox, &flipdir, gproperty, gcfg);
        } else {
            updateproperty(&prop, mediaid, gproperty, gcfg);
        }

#else
#if MED_TYPE == MEDIA_ASGN_F2H
        updateproperty_asgn(&prop, mediaid, idx1d, media, gproperty, gcfg);
#elif MED_TYPE == MEDIA_2LABEL_SPLIT
        updateproperty_svmc(&prop, mediaid, idx1d, media,
                            FLOAT3(p.x, p.y, p.z), &nuvox, &flipdir, gproperty, gcfg);
#else
        updateproperty(&prop, mediaid, gproperty, gcfg);
#endif
#endif

        /*--- Advance photon to next voxel boundary ---*/
        f.z = hitgrid(&p, &v, &flipdir);
        float slen = f.z * prop.y * (v.w + 1.f > GPU_PARAM(gcfg, gscatter) ? (1.f - prop.z) : 1.f);
        slen = fmin(slen, f.x);
        f.z = native_divide(slen, prop.y * (v.w + 1.f > GPU_PARAM(gcfg, gscatter) ? (1.f - prop.z) : 1.f));

        GPUDEBUG(("p=[%f %f %f] -> <%f %f %f>*%f -> flip=%d\n", p.x, p.y, p.z, v.x, v.y, v.z, f.z, flipdir.w));

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

        if (issvmc) {
#endif

            if (SV_ISSPLIT(nuvox.sv) && testint) {
                float tmplen = f.z;
                hitintf = ray_plane_intersect(FLOAT3(p.x, p.y, p.z), &v, &prop, &tmplen, &slen, &nuvox, gcfg);
                f.z = tmplen;
            } else {
                hitintf = 0;
            }

#ifdef __NVCC__
        } else {
            hitintf = 0;
        }

#endif
#endif

        pathlen += f.z;

        p.x = p.x + f.z * v.x;
        p.y = p.y + f.z * v.y;
        p.z = p.z + f.z * v.z;

        /* FIX: flipdir updates — use logical expression instead of if-statement inside ternary */
        if (flipdir.w == 0) {
            flipdir.x += (slen == f.x
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__
                          || (issvmc && hitintf)
#else
                          || hitintf
#endif
#endif
                         ) ? 0 : (v.x > 0.f ? 1 : -1);
        }

        if (flipdir.w == 1) {
            flipdir.y += (slen == f.x
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__
                          || (issvmc && hitintf)
#else
                          || hitintf
#endif
#endif
                         ) ? 0 : (v.y > 0.f ? 1 : -1);
        }

        if (flipdir.w == 2) {
            flipdir.z += (slen == f.x
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__
                          || (issvmc && hitintf)
#else
                          || hitintf
#endif
#endif
                         ) ? 0 : (v.z > 0.f ? 1 : -1);
        }

        /*--- Beer-Lambert attenuation ---*/
        p.w *= MCX_MATHFUN(exp)(-prop.x * f.z);
        f.x -= slen;
        f.y += f.z * prop.w * GPU_PARAM(gcfg, oneoverc0);

        GPUDEBUG(("update p=[%f %f %f] -> f.z=%f\n", p.x, p.y, p.z, f.z));

#ifdef MCX_SAVE_DETECTORS

        if (GPU_PARAM(gcfg, savedet) && SAVE_PPATH(GPU_PARAM(gcfg, savedetflag))) {
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

            if (issvmc) {
#endif

                if (SV_CURLABEL(nuvox.sv) > 0) {
                    ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag))) + SV_CURLABEL(nuvox.sv) - 1] += f.z;
                }

#ifdef __NVCC__
            } else {
                ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag))) + (mediaid & MED_MASK) - 1] += f.z;
            }

#endif
#else
            ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag))) + (mediaid & MED_MASK) - 1] += f.z;
#endif
        }

#endif

        /*--- Read new voxel media index ---*/
        mediaidold = mediaid | isdet;
        idx1dold = idx1d;
        idx1d = (flipdir.z * gcfg->dimlen.y + flipdir.y * gcfg->dimlen.x + flipdir.x);
        GPUDEBUG(("idx1d [%d]->[%d]\n", idx1dold, idx1d));

        if ((ushort)flipdir.x >= gcfg->maxidx.x || (ushort)flipdir.y >= gcfg->maxidx.y || (ushort)flipdir.z >= gcfg->maxidx.z) {
            mediaid = 0;
            idx1d = (flipdir.x < 0 || flipdir.y < 0 || flipdir.z < 0) ? OUTSIDE_VOLUME_MIN : OUTSIDE_VOLUME_MAX;
            isdet = gcfg->bc[(idx1d == OUTSIDE_VOLUME_MAX) * 3 + flipdir.w];
            isdet = ((isdet & 0xF) == bcUnknown) ? (GPU_PARAM(gcfg, doreflect) ? bcReflect : bcAbsorb) : isdet;
            GPUDEBUG(("moving outside: [%f %f %f], idx1d [%d]->[out], bcflag %d\n", p.x, p.y, p.z, idx1d, isdet));
        } else {
            mediaid = media[idx1d];
            isdet = mediaid & DET_MASK;
            mediaid &= MED_MASK;
        }

        GPUDEBUG(("medium [%d]->[%d]\n", mediaidold, mediaid));

        /*--- Deposit energy when crossing voxel boundary ---*/
        if ((idx1d != idx1dold
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__
                || (issvmc && hitintf)
#else
                || hitintf
#endif
#endif
            ) && idx1dold < gcfg->dimlen.z && mediaidold) {

            GPUDEBUG(("field add to %d->%f(%d)\n", idx1dold, w0 - p.w, (int)f.w));

            if (GPU_PARAM(gcfg, save2pt) && f.y >= gcfg->twin0 && f.y < gcfg->twin1) {
                float weight = 0.f;
                int tshift = (int)(FLOOR((f.y - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep)));

                if (GPU_PARAM(gcfg, outputtype) == otEnergy) {
                    weight = w0 - p.w;
                } else if ((bool)(GPU_PARAM(gcfg, outputtype) == otFluence) || (bool)(GPU_PARAM(gcfg, outputtype) == otFlux)) {
                    weight = (prop.x < EPS) ? (w0 * pathlen) : ((w0 - p.w) / (prop.x));
                } else if (GPU_PARAM(gcfg, seed) == SEED_FROM_FILE) {
                    if (GPU_PARAM(gcfg, outputtype) == otJacobian | GPU_PARAM(gcfg, outputtype) == otRF | GPU_PARAM(gcfg, outputtype) == otWLTOF) {
                        weight = replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w) - 1] * pathlen;

                        tshift = (idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w - 1);
                        tshift = (int)(FLOOR((photontof[tshift] - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep))) +
                                 ( (GPU_PARAM(gcfg, replaydet) == -1) ? (((photondetid[tshift] & 0xFFFF) - 1) * GPU_PARAM(gcfg, maxgate)) : 0);

                        if (GPU_PARAM(gcfg, outputtype) == otRF) {
                            weight = -weight * ppath[GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum)];
                        } else if (GPU_PARAM(gcfg, outputtype) == otWLTOF) {
                            weight = weight * photontof[idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w - 1];
                        }
                    }
                } else if (GPU_PARAM(gcfg, outputtype) == otL) {
                    weight = w0 * pathlen;
                }

                if (GPU_PARAM(gcfg, extrasrclen) * (GPU_PARAM(gcfg, srcid) < 0)) {
                    tshift += ((int)ppath[GPU_PARAM(gcfg, w0offset) - 1] - 1) * ((GPU_PARAM(gcfg, replaydet) == -1) ? GPU_PARAM(gcfg, detnum) : 1) * GPU_PARAM(gcfg, maxgate);
                }

                GPUDEBUG(("deposit to [%d] %e, w=%f\n", idx1dold, weight, p.w));

                if (FABS(weight) > 0.f) {
#ifndef USE_ATOMIC
                    field[idx1dold + tshift * gcfg->dimlen.z] += weight;
#else
#if !defined(MCX_SRC_PATTERN) && !defined(MCX_SRC_PATTERN3D)
                    float oldval = atomicadd(field + idx1dold + tshift * gcfg->dimlen.z, weight);

                    if (FABS(oldval) > MAX_ACCUM) {
                        atomicadd(field + idx1dold + tshift * gcfg->dimlen.z, ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                        atomicadd(field + idx1dold + tshift * gcfg->dimlen.z + gcfg->dimlen.w, ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                    }

                    if ((GPU_PARAM(gcfg, outputtype) == otRF) & (GPU_PARAM(gcfg, omega) > 0.f)) {
                        float rf_imag = -replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w - 1)] * pathlen * ppath[GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum) + 1];
                        atomicadd(field + idx1dold + tshift * gcfg->dimlen.z + gcfg->dimlen.w, rf_imag);
                    }

#else

                    for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                        if (FABS(ppath[GPU_PARAM(gcfg, w0offset) + i]) > 0.f) {
                            float oldval = atomicadd(field + (idx1dold + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i, ((GPU_PARAM(gcfg, srcnum) == 1) ? weight : weight * ppath[GPU_PARAM(gcfg, w0offset) + i]));

                            if (FABS(oldval) > MAX_ACCUM) {
                                atomicadd(field + (idx1dold + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i, ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                                atomicadd(field + (idx1dold + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i + gcfg->dimlen.w, ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                            }
                        }
                    }

#endif
                    GPUDEBUG(("atomic write to [%d] %e, w=%f\n", idx1dold, w0, p.w));
#endif
                }
            }

            w0 = p.w;
            pathlen = 0.f;
        } else {
            mediaid = mediaidold;
        }

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

        if (issvmc) {
#endif

            /*--- SVMC: update tissue type on voxel or interface crossing ---*/
            if (idx1d != idx1dold) {
                updateproperty_svmc(&prop, mediaid, idx1d, media,
                                    FLOAT3(p.x, p.y, p.z), &nuvox, &flipdir, gproperty, gcfg);
                testint = 1;
            } else if (hitintf) {
                nuvox.nv.x = -nuvox.nv.x;
                nuvox.nv.y = -nuvox.nv.y;
                nuvox.nv.z = -nuvox.nv.z;
                nuvox.pd = -nuvox.pd;
                SV_FLIP_ISUPPER(nuvox.sv);
                testint = 0;
            }

#ifdef __NVCC__
        }

#endif
#endif

        /*--- Launch new photon: exit volume / time window / cyclic BC ---*/
        if ((mediaid == 0 && ((isdet & 0xF) == bcAbsorb || (isdet & 0xF) == bcCyclic || ((isdet & 0xF) == bcReflect && n1 == gproperty[0].w)))
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__
                || (issvmc && ((idx1d != idx1dold || hitintf) && !SV_ISUPPER(nuvox.sv) && !SV_LOWER(nuvox.sv)
                               && (!GPU_PARAM(gcfg, doreflect) || n1 == gproperty[0].w)))
#else
                || ((idx1d != idx1dold || hitintf) && !SV_ISUPPER(nuvox.sv) && !SV_LOWER(nuvox.sv)
                    && (!GPU_PARAM(gcfg, doreflect) || n1 == gproperty[0].w))
#endif
#endif
                || f.y > gcfg->twin1) {

            if (isdet == bcCyclic) {
                if (flipdir.w == 0) {
                    p.x = mcx_nextafterf(RINT(((idx1d == OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.x : 0)), (v.x > 0.f) - (v.x < 0.f));
                    flipdir.x = (short)FLOOR(p.x);
                }

                if (flipdir.w == 1) {
                    p.y = mcx_nextafterf(RINT(((idx1d == OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.y : 0)), (v.y > 0.f) - (v.y < 0.f));
                    flipdir.y = (short)FLOOR(p.y);
                }

                if (flipdir.w == 2) {
                    p.z = mcx_nextafterf(RINT(((idx1d == OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.z : 0)), (v.z > 0.f) - (v.z < 0.f));
                    flipdir.z = (short)FLOOR(p.z);
                }

                GPUDEBUG(("cyclic: p=[%f %f %f] -> voxel =[%d %d %d] %d %d\n", p.x, p.y, p.z, flipdir.x, flipdir.y, flipdir.z, isdet, bcCyclic));

                if ((ushort)flipdir.x < gcfg->maxidx.x && (ushort)flipdir.y < gcfg->maxidx.y && (ushort)flipdir.z < gcfg->maxidx.z) {
                    idx1d = (flipdir.z * gcfg->dimlen.y + flipdir.y * gcfg->dimlen.x + flipdir.x);
                    mediaid = media[idx1d];
                    isdet = mediaid & DET_MASK;
                    mediaid &= MED_MASK;
                    GPUDEBUG(("Cyclic BC, dir %d flag %d, new pos=[%f %f %f]\n", flipdir.w, isdet, p.x, p.y, p.z));
                    continue;
                }
            }

            GPUDEBUG(("direct relaunch at idx=[%d] mediaid=[%d] bc=[%d] ref=[%d]\n", idx1d, mediaid, isdet, GPU_PARAM(gcfg, doreflect)));

            if (LAUNCHNEWPHOTON(&p, &v, &s, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove,
                                (((idx1d == OUTSIDE_VOLUME_MAX && gcfg->bc[9 + flipdir.w]) || (idx1d == OUTSIDE_VOLUME_MIN && gcfg->bc[6 + flipdir.w])) ? OUTSIDE_VOLUME_MIN : (mediaidold & DET_MASK)),
                                ppath, n_det, detectedphoton, t, (__global RandType*)n_seed, gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton, gprogress,
                                (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen))
                                                    + get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)), gseeddata, gjumpdebug, gdebugdata, sharedmem, &nuvox, photontof)) {
                break;
            }

            isdet = mediaid & DET_MASK;
            mediaid &= MED_MASK;
#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

            if (issvmc) {
#endif
                testint = 1;
#ifdef __NVCC__
            }

#endif
#endif
            continue;
        }

        /*--- Russian Roulette ---*/
        if (FABS(p.w) < gcfg->minenergy) {
            if (rand_do_roulette(t) * ROULETTE_SIZE <= 1.f) {
                p.w *= ROULETTE_SIZE;
            } else {
                GPUDEBUG(("relaunch after Russian roulette at idx=[%d] mediaid=[%d], ref=[%d]\n", idx1d, mediaid, GPU_PARAM(gcfg, doreflect)));

                if (LAUNCHNEWPHOTON(&p, &v, &s, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove,
                                    (mediaidold & DET_MASK), ppath, n_det, detectedphoton, t, (__global RandType*)n_seed, gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton, gprogress,
                                    (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen))
                                                        + get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)), gseeddata, gjumpdebug, gdebugdata, sharedmem, &nuvox, photontof)) {
                    break;
                }

                isdet = mediaid & DET_MASK;
                mediaid &= MED_MASK;
                continue;
            }
        }

#if defined(__NVCC__) || defined(MCX_DO_REFLECTION)
#ifdef __NVCC__

        if (isreflect) {
#endif

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

            if (issvmc) {
#endif

                /*--- SVMC intra-voxel reflection/refraction ---*/
                if (hitintf) {
                    if (gproperty[SV_LOWER(nuvox.sv)].w != gproperty[SV_UPPER(nuvox.sv)].w) {
                        nuvox.nv.x = -nuvox.nv.x;
                        nuvox.nv.y = -nuvox.nv.y;
                        nuvox.nv.z = -nuvox.nv.z;
                        nuvox.pd = -nuvox.pd;
                        float3 c0 = FLOAT3(v.x, v.y, v.z);

                        if (reflectray_svmc(n1, &c0, &nuvox, &prop, t, gproperty)) {
                            if (LAUNCHNEWPHOTON(&p, &v, &s, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove,
                                                (mediaidold & DET_MASK), ppath, n_det, detectedphoton, t, (__global RandType*)n_seed,
                                                gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton, gprogress,
                                                (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen)) +
                                                                    get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)),
                                                gseeddata, gjumpdebug, gdebugdata, sharedmem, &nuvox, photontof)) {
                                break;
                            }

                            isdet = mediaid & DET_MASK;
                            mediaid &= MED_MASK;
                            testint = 1;
                            continue;
                        }

                        v.x = c0.x;
                        v.y = c0.y;
                        v.z = c0.z;
                    } else {
                        *((FLOAT4VEC*)(&prop)) = gproperty[SV_CURLABEL(nuvox.sv)];
                    }
                } else
#ifdef __NVCC__
                {
                    /* empty else block for SVMC when !hitintf, fall through to voxel-boundary reflection */
                }
            } else {
                /* non-SVMC path under CUDA */
            }

            /* voxel-boundary reflection for both SVMC and non-SVMC */
            if (!issvmc || !hitintf) {
#endif
#endif

#ifdef __NVCC__

                if (gcfg->mediaformat == MEDIA_ASGN_F2H) {
                    updateproperty_asgn(&prop, mediaid, idx1d, media, gproperty, gcfg);
                } else {
                    updateproperty(&prop, mediaid, gproperty, gcfg);
                }

#else
#if MED_TYPE == MEDIA_ASGN_F2H
                updateproperty_asgn(&prop, mediaid, idx1d, media, gproperty, gcfg);
#elif MED_TYPE != MEDIA_2LABEL_SPLIT
                updateproperty(&prop, mediaid, gproperty, gcfg);
#endif
#endif

                /*--- Reflection/transmission at voxel boundary ---*/
                if (((mediaid && GPU_PARAM(gcfg, doreflect))
                        || (mediaid == 0 &&
                            (((isdet & 0xF) == bcUnknown && GPU_PARAM(gcfg, doreflect))
                             || ((isdet & 0xF) == bcReflect || (isdet & 0xF) == bcMirror))))
                        && (((isdet & 0xF) == bcMirror) || n1 != ((GPU_PARAM(gcfg, mediaformat) < 100) ? (prop.w) : (gproperty[(mediaid > 0 && (bool)(GPU_PARAM(gcfg, mediaformat) >= 100)) ? 1 : mediaid].w)))) {

                    float Rtotal = 1.f;
                    float cphi, sphi, stheta, ctheta;

                    updateproperty(&prop, mediaid, gproperty, gcfg);

                    float tmp0 = n1 * n1;
                    float tmp1 = prop.w * prop.w;
                    cphi = FABS( (flipdir.w == 0) ? v.x : (flipdir.w == 1 ? v.y : v.z));
                    sphi = 1.f - cphi * cphi;

                    f.z = 1.f - tmp0 / tmp1 * sphi;
                    GPUDEBUG(("ref total ref=%f\n", f.z));

                    if (f.z > 0.f && (isdet & 0xF) != bcMirror) {
                        ctheta = tmp0 * cphi * cphi + tmp1 * f.z;
                        stheta = 2.f * n1 * prop.w * cphi * MCX_MATHFUN(sqrt)(f.z);
                        Rtotal = (ctheta - stheta) / (ctheta + stheta);
                        ctheta = tmp1 * cphi * cphi + tmp0 * f.z;
                        Rtotal = (Rtotal + (ctheta - stheta) / (ctheta + stheta)) * 0.5f;
                        GPUDEBUG(("Rtotal=%f\n", Rtotal));
                    }

                    if (Rtotal < 1.f
                            && (!(mediaid == 0 && ((isdet & 0xF) == bcMirror)))
                            && rand_next_reflect(t) > Rtotal) {

                        transmit(&v, n1, prop.w, flipdir.w);

                        if (mediaid == 0) {
                            GPUDEBUG(("transmit to air, relaunch, idx=[%d] mediaid=[%d] bc=[%d] ref=[%d]\n", idx1d, mediaid, isdet, GPU_PARAM(gcfg, doreflect)));

                            if (LAUNCHNEWPHOTON(&p, &v, &s, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove,
                                                (((idx1d == OUTSIDE_VOLUME_MAX && gcfg->bc[9 + flipdir.w]) || (idx1d == OUTSIDE_VOLUME_MIN && gcfg->bc[6 + flipdir.w])) ? OUTSIDE_VOLUME_MIN : (mediaidold & DET_MASK)),
                                                ppath, n_det, detectedphoton, t, (__global RandType*)n_seed, gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton, gprogress,
                                                (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen))
                                                                    + get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)), gseeddata, gjumpdebug, gdebugdata, sharedmem, &nuvox, photontof)) {
                                break;
                            }

                            isdet = mediaid & DET_MASK;
                            mediaid &= MED_MASK;
                            continue;
                        }

                        GPUDEBUG(("do transmission\n"));
                    } else {
                        /*--- Reflection ---*/
                        GPUDEBUG(("do reflection\n"));
                        GPUDEBUG(("ref faceid=%d p=[%f %f %f] v_old=[%f %f %f]\n", flipdir.w, p.x, p.y, p.z, v.x, v.y, v.z));
                        (flipdir.w == 0) ? (v.x = -v.x) : ((flipdir.w == 1) ? (v.y = -v.y) : (v.z = -v.z)) ;
                        (flipdir.w == 0) ?
                        (p.x = mcx_nextafterf(RINT(p.x), (v.x > 0.f) - 0.5f)) :
                        ((flipdir.w == 1) ?
                         (p.y = mcx_nextafterf(RINT(p.y), (v.y > 0.f) - 0.5f)) :
                         (p.z = mcx_nextafterf(RINT(p.z), (v.z > 0.f) - 0.5f)) );
                        (flipdir.w == 0) ? (flipdir.x = (short)RINT(p.x)) : ((flipdir.w == 1) ? (flipdir.y = (short)RINT(p.y)) : (flipdir.z = (short)RINT(p.z))) ;
                        GPUDEBUG(("ref p_new=[%f %f %f] v_new=[%f %f %f]\n", p.x, p.y, p.z, v.x, v.y, v.z));
                        idx1d = idx1dold;
                        mediaid = (media[idx1d] & MED_MASK);

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__

                        if (issvmc) {
#endif
                            updateproperty_svmc(&prop, mediaid, idx1d, media,
                                                FLOAT3(p.x, p.y, p.z), &nuvox, &flipdir, gproperty, gcfg);

                            if (SV_CURLABEL(nuvox.sv) == 0) {
                                if (LAUNCHNEWPHOTON(&p, &v, &s, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove,
                                                    (mediaidold & DET_MASK), ppath, n_det, detectedphoton, t, (__global RandType*)n_seed,
                                                    gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton, gprogress,
                                                    (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen)) +
                                                                        get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)),
                                                    gseeddata, gjumpdebug, gdebugdata, sharedmem, &nuvox, photontof)) {
                                    break;
                                }

                                isdet = mediaid & DET_MASK;
                                mediaid &= MED_MASK;
#ifdef __NVCC__
                                testint = 1;
#endif
                                continue;
                            }

#ifdef __NVCC__
                        } else {
#else
                            /* OpenCL: fall through to generic updateproperty */
#endif
#endif

#ifdef __NVCC__

                            if (gcfg->mediaformat == MEDIA_ASGN_F2H) {
                                updateproperty_asgn(&prop, mediaid, idx1d, media, gproperty, gcfg);
                            } else {
                                updateproperty(&prop, mediaid, gproperty, gcfg);
                            }

#else
#if MED_TYPE == MEDIA_ASGN_F2H
                            updateproperty_asgn(&prop, mediaid, idx1d, media, gproperty, gcfg);
#elif MED_TYPE != MEDIA_2LABEL_SPLIT
                            updateproperty(&prop, mediaid, gproperty, gcfg);
#endif
#endif

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__
                        }

#endif
#endif
                        n1 = prop.w;
                    }
                }

#if MED_TYPE == MEDIA_2LABEL_SPLIT || defined(__NVCC__)
#ifdef __NVCC__
            } /* close: if (!issvmc || !hitintf) */

#endif
#endif

            if ((GPU_PARAM(gcfg, debuglevel) & MCX_DEBUG_MOVE_ONLY) &&
                    (mediaid == 0 || idx1d == OUTSIDE_VOLUME_MIN || idx1d == OUTSIDE_VOLUME_MAX)) {
                GPUDEBUG(("ERROR: should never happen! mediaid=%d idx1d=%X\n", mediaid, idx1d));
                return;
            }

#ifdef __NVCC__
        } /* close: if (isreflect) */

#endif
#endif  /* MCX_DO_REFLECTION */
    }   /* end of main while loop */

    /*--- Write back accumulated energy ---*/
    genergy[idx << 1]    = ppath[GPU_PARAM(gcfg, partialdata)];
    genergy[(idx << 1) + 1] = ppath[GPU_PARAM(gcfg, partialdata) + 1];

    if (GPU_PARAM(gcfg, issaveref) > 1) {
        *detectedphoton = GPU_PARAM(gcfg, maxdetphoton);
    }
}