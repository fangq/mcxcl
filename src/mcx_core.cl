/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration \
**      -- OpenCL edition
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2024
**
**  \section sref Reference:
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**         "Scalable and massively parallel Monte Carlo photon transport simulations
**         for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504 (2018)
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

#ifdef MCX_SAVE_DETECTORS
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#if defined(USE_HALF) || MED_TYPE==MEDIA_AS_F2H || MED_TYPE==MEDIA_AS_HALF || MED_TYPE==MEDIA_LABEL_HALF
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef USE_HALF
    #define FLOAT4VEC half4
    #define TOFLOAT4  convert_half4
#else
    #define FLOAT4VEC float4
    #define TOFLOAT4
#endif

#ifdef MCX_USE_NATIVE
    #define MCX_MATHFUN(fun)              native_##fun
    #define MCX_SINCOS(theta,osin,ocos)   {(osin)=native_sin(theta);(ocos)=native_cos(theta);}
#else
    #define MCX_MATHFUN(fun)              fun
    #define MCX_SINCOS(theta,osin,ocos)   (ocos)=sincos((theta),&(osin))
#endif

#ifdef MCX_GPU_DEBUG
    #define GPUDEBUG(x)        printf x             // enable debugging in CPU mode
#else
    #define GPUDEBUG(x)
#endif

#define R_PI               0.318309886183791f
#define RAND_MAX           4294967295

#define ONE_PI             3.1415926535897932f     //pi
#define TWO_PI             6.28318530717959f       //2*pi
#define JUST_ABOVE_ONE     1.0001f                 /**< test for boundary */
#define JUST_BELOW_ONE     0.9998f                 /**< test for boundary */

#define C0                 299792458000.f          //speed of light in mm/s
#define R_C0               3.335640951981520e-12f  //1/C0 in s/mm

#define EPS                FLT_EPSILON             //round-off limit
#define VERY_BIG           (1.f/FLT_EPSILON)       //a big number
#define JUST_ABOVE_ONE     1.0001f                 //test for boundary
#define SAME_VOXEL         -9999.f                 //scatter within a voxel
#define NO_LAUNCH          9999                    //when fail to launch, for debug
#define FILL_MAXDETPHOTON  3                       /**< when the detector photon buffer is filled, terminate simulation*/
#define MAX_PROP           2000                     /*maximum property number*/
#define OUTSIDE_VOLUME_MIN 0xFFFFFFFF              /**< flag indicating the index is outside of the volume from x=xmax,y=ymax,z=zmax*/
#define OUTSIDE_VOLUME_MAX 0x7FFFFFFF              /**< flag indicating the index is outside of the volume from x=0/y=0/z=0*/
#define BOUNDARY_DET_MASK  0xFFFF0000              /**< flag indicating a boundary face is used as a detector*/
#define SEED_FROM_FILE      -999                   /**< special flag indicating to read seeds from an mch file for replay */

#define DET_MASK           0x80000000              /**< mask of the sign bit to get the detector */
#define MED_MASK           0x7FFFFFFF              /**< mask of the remaining bits to get the medium index */
#define MIX_MASK           0x7FFF0000              /**< mask of the upper 16bit to get the volume mix ratio */
#define NULL               0
#define MAX_ACCUM          1000.f

#define MCX_DEBUG_REC_LEN   6  /**<  number of floating points per position saved when -D M is used for trajectory */

#define MIN(a,b)           ((a)<(b)?(a):(b))

#define MEDIA_2LABEL_SPLIT    97   /**<  svmc media format (not supported): 64bit:{[byte: lower label][byte: upper label][byte*3: reference point][byte*3: normal vector]} */
#define MEDIA_2LABEL_MIX      98   /**<  mixlabel media format: {[int: label1][int: label2][float32: label1 %]} -> 32bit:{[short label1 % scaled to 65535],[byte: label2],[byte: label1]} */
#define MEDIA_LABEL_HALF      99   /**<  labelplus media format: {[float32: 1/2/3/4][float32: type][float32: mua/mus/g/n]} -> 32bit:{[half: mua/mus/g/n][int16: [B15-B16: 0/1/2/3][B1-B14: tissue type]} */
#define MEDIA_AS_F2H          100  /**<  muamus_float media format: {[float32: mua][float32: mus]} -> 32bit:{[half: mua],{half: mus}} */
#define MEDIA_MUA_FLOAT       101  /**<  mua_float media format: 32bit:{[float32: mua]} */
#define MEDIA_AS_HALF         102  /**<  muamus_half media format: 32bit:{[half: mua],[half: mus]} */
#define MEDIA_ASGN_BYTE       103  /**<  asgn_byte media format: 32bit:{[byte: mua],[byte: mus],[byte: g],[byte: n]} */
#define MEDIA_AS_SHORT        104  /**<  muamus_short media format: 32bit:{[short: mua],[short: mus]} */

#define SAVE_DETID(a)         ((a)    & 0x1)   /**<  mask to save detector ID*/
#define SAVE_NSCAT(a)         ((a)>>1 & 0x1)   /**<  output partial scattering counts */
#define SAVE_PPATH(a)         ((a)>>2 & 0x1)   /**<  output partial path */
#define SAVE_MOM(a)           ((a)>>3 & 0x1)   /**<  output momentum transfer */
#define SAVE_PEXIT(a)         ((a)>>4 & 0x1)   /**<  save exit positions */
#define SAVE_VEXIT(a)         ((a)>>5 & 0x1)   /**<  save exit vector/directions */
#define SAVE_W0(a)            ((a)>>6 & 0x1)   /**<  save initial weight */

#define SET_SAVE_DETID(a)     ((a) | 0x1   )   /**<  mask to save detector ID*/
#define SET_SAVE_NSCAT(a)     ((a) | 0x1<<1)   /**<  output partial scattering counts */
#define SET_SAVE_PPATH(a)     ((a) | 0x1<<2)   /**<  output partial path */
#define SET_SAVE_MOM(a)       ((a) | 0x1<<3)   /**<  output momentum transfer */
#define SET_SAVE_PEXIT(a)     ((a) | 0x1<<4)   /**<  save exit positions */
#define SET_SAVE_VEXIT(a)     ((a) | 0x1<<5)   /**<  save exit vector/directions */
#define SET_SAVE_W0(a)        ((a) | 0x1<<6)   /**<  save initial weight */

#define UNSET_SAVE_DETID(a)     ((a) & ~(0x1)   )   /**<  mask to save detector ID*/
#define UNSET_SAVE_NSCAT(a)     ((a) & ~(0x1<<1))   /**<  output partial scattering counts */
#define UNSET_SAVE_PPATH(a)     ((a) & ~(0x1<<2))   /**<  output partial path */
#define UNSET_SAVE_MOM(a)       ((a) & ~(0x1<<3))   /**<  output momentum transfer */
#define UNSET_SAVE_PEXIT(a)     ((a) & ~(0x1<<4))   /**<  save exit positions */
#define UNSET_SAVE_VEXIT(a)     ((a) & ~(0x1<<5))   /**<  save exit vector/directions */
#define UNSET_SAVE_W0(a)        ((a) & ~(0x1<<6))   /**<  save initial weight */

typedef struct KernelParams {
    float4 ps, c0;
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
    uint   idx1dorig;
    uint   mediaidorig;
    uint   blockphoton;
    uint   blockextra;
    int    voidtime;
    int    srctype;                    /**< type of the source */
    float4 srcparam1;                  /**< source parameters set 1 */
    float4 srcparam2;                  /**< source parameters set 2 */
    uint   maxvoidstep;
    uint   issaveexit;    /**<1 save the exit position and dir of a detected photon, 0 do not save*/
    uint   issaveseed;     /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    uint   issaveref;     /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    uint   isspecular;           /**< 0 do not perform specular reflection at launch, 1 do specular reflection */
    uint   maxgate;
    int    seed;                          /**< RNG seed passted from the host */
    uint   outputtype;           /**< Type of output to be accummulated */
    uint   threadphoton;                  /**< how many photons to be simulated in a thread */
    int    oddphoton;                    /**< how many threads need to simulate 1 more photon above the basic load (threadphoton) */
    uint   debuglevel;           /**< debug flags */
    uint   savedetflag;          /**< detected photon save flags */
    uint   reclen;               /**< length of buffer per detected photon */
    uint   partialdata;          /**< per-medium detected photon data length */
    uint   w0offset;             /**< photon-sharing buffer offset */
    uint   mediaformat;          /**< format of the media buffer */
    uint   maxjumpdebug;         /**< max number of positions to be saved to save photon trajectory when -D M is used */
    uint   gscatter;             /**< how many scattering events after which mus/g can be approximated by mus' */
    uint   is2d;                 /**< is the domain a 2D slice? */
    int    replaydet;                     /**< select which detector to replay, 0 for all, -1 save all separately */
    uint   srcnum;               /**< total number of source patterns */
    unsigned int nphase;               /**< number of samples for inverse-cdf, will be added by 2 to include -1 and 1 on the two ends */
    unsigned int nphaselen;            /**< even-rounded nphase so that shared memory buffer won't give an error */
    unsigned int nangle;               /**< number of samples for launch angle inverse-cdf, will be added by 2 to include 0 and 1 on the two ends */
    unsigned int nanglelen;            /**< even-rounded nangle so that shared memory buffer won't give an error */
    unsigned char bc[12];               /**< boundary conditions */
} MCXParam __attribute__ ((aligned (32)));

enum TBoundary {bcUnknown, bcReflect, bcAbsorb, bcMirror, bcCyclic};            /**< boundary conditions */
enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otWP, otDCS, otL};   /**< types of output */

#ifndef USE_MACRO_CONST
    #define GPU_PARAM(a,b) (a->b)
#else
    #define GPU_PARAM(a,b) (a ## b)
#endif

//#ifndef USE_XORSHIFT128P_RAND     // xorshift128+ is the default RNG
#ifdef USE_LL5_RAND                 //enable the legacy Logistic Lattic RNG

#define RAND_BUF_LEN       5        //register arrays
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define INIT_LOGISTIC      100

typedef float RandType;

#define FUN(x)               (4.f*(x)*(1.f-(x)))
#define NU                   1e-7f
#define NU2                  (1.f-2.f*NU)
#define MIN_INVERSE_LIMIT    1e-7f
#define logistic_uniform(v)  (acos(1.f-2.f*(v))*R_PI)
#define R_MAX_C_RAND         (1.f/RAND_MAX)
#define LOG_MT_MAX           22.1807097779182f

#define RING_FUN(x,y,z)      (NU2*(x)+NU*((y)+(z)))

void logistic_step(__private RandType* t, __private RandType* tnew, int len_1) {
    t[0] = FUN(t[0]);
    t[1] = FUN(t[1]);
    t[2] = FUN(t[2]);
    t[3] = FUN(t[3]);
    t[4] = FUN(t[4]);
    tnew[4] = RING_FUN(t[0], t[4], t[1]); /* shuffle the results by separation of 2*/
    tnew[0] = RING_FUN(t[1], t[0], t[2]);
    tnew[1] = RING_FUN(t[2], t[1], t[3]);
    tnew[2] = RING_FUN(t[3], t[2], t[4]);
    tnew[3] = RING_FUN(t[4], t[3], t[0]);
}
// generate random number for the next zenith angle
void rand_need_more(__private RandType t[RAND_BUF_LEN]) {
    RandType tnew[RAND_BUF_LEN] = {0.f};
    logistic_step(t, tnew, RAND_BUF_LEN - 1);
    logistic_step(tnew, t, RAND_BUF_LEN - 1);
}

void logistic_init(__private RandType* t, __global uint* seed, uint idx) {
    int i;

    for (i = 0; i < RAND_BUF_LEN; i++) {
        t[i] = (RandType)seed[idx * RAND_BUF_LEN + i] * R_MAX_C_RAND;
    }

    for (i = 0; i < INIT_LOGISTIC; i++) { /*initial randomization*/
        rand_need_more(t);
    }
}
// transform into [0,1] random number
RandType rand_uniform01(__private RandType t[RAND_BUF_LEN]) {
    rand_need_more(t);
    return logistic_uniform(t[0]);
}
void gpu_rng_init(__private RandType t[RAND_BUF_LEN], __global uint* n_seed, int idx) {
    logistic_init(t, n_seed, idx);
}

#else

#define RAND_BUF_LEN       2        //register arrays
#define RAND_SEED_LEN      4        //48 bit packed with 64bit length
#define LOG_MT_MAX         22.1807097779182f
#define IEEE754_DOUBLE_BIAS     0x3FF0000000000000ul /* Added to exponent.  */

typedef ulong  RandType;

static float xorshift128p_nextf (__private RandType t[RAND_BUF_LEN]) {
    union {
        ulong  i;
        float f[2];
        uint  u[2];
    } s1;
    const ulong s0 = t[1];
    s1.i = t[0];
    t[0] = s0;
    s1.i ^= s1.i << 23; // a
    t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5); // b, c
    s1.i = t[1] + s0;
    s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);

    return s1.f[0] - 1.0f;
}

static void copystate(__private RandType* t, __local RandType* tnew) {
    tnew[0] = t[0];
    tnew[1] = t[1];
}

static float rand_uniform01(__private RandType t[RAND_BUF_LEN]) {
    return xorshift128p_nextf(t);
}

static void xorshift128p_seed (__global uint* seed, __private RandType t[RAND_BUF_LEN]) {
    t[0] = (ulong)seed[0] << 32 | seed[1] ;
    t[1] = (ulong)seed[2] << 32 | seed[3];
}

static void gpu_rng_init(__private RandType t[RAND_BUF_LEN], __global uint* n_seed, int idx) {
    xorshift128p_seed((n_seed + idx * RAND_SEED_LEN), t);
}

#endif

float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]);

float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]) {
    return -MCX_MATHFUN(log)(rand_uniform01(t) + EPS);
}

#define rand_next_aangle(t)  rand_uniform01(t)
#define rand_next_zangle(t)  rand_uniform01(t)
#define rand_next_reflect(t) rand_uniform01(t)
#define rand_do_roulette(t)  rand_uniform01(t)


/* function prototypes */

void clearpath(__local float* p, uint maxmediatype);
float mcx_nextafterf(float a, int dir);
float hitgrid(float4* p0, float4* v, short4* id);
void rotatevector(float4* v, float stheta, float ctheta, float sphi, float cphi);
void transmit(float4* v, float n1, float n2, short flipdir);
float reflectcoeff(float4* v, float n1, float n2, short flipdir);
int skipvoid(float4* p, float4* v, float4* f, short4* flipdir, __global const uint* media, __constant float4* gproperty, __constant MCXParam* gcfg);
void rotatevector2d(float4* v, float stheta, float ctheta, int is2d);
void updateproperty(FLOAT4VEC* prop, unsigned int mediaid, __constant float4* gproperty, __constant MCXParam* gcfg);

#ifdef MCX_SAVE_DETECTORS
uint finddetector(float4* p0, __constant float4* gdetpos, __constant MCXParam* gcfg);
void savedetphoton(__global float* n_det, __global uint* detectedphoton,
                   __local float* ppath, float4* p0, float4* v,
                   __local RandType* t, __global RandType* seeddata,
                   __constant float4* gdetpos, __constant MCXParam* gcfg, uint isdet);
void saveexitppath(__global float* n_det, __local float* ppath, float4* p0, uint* idx1d, __constant MCXParam* gcfg);
#endif
int launchnewphoton(float4* p, float4* v, float4* f, short4* flipdir, FLOAT4VEC* prop, uint* idx1d,
                    __global float* field, uint* mediaid, float* w0, float* Lmove, uint isdet,
                    __local float* ppath, __global float* n_det, __global uint* dpnum,
                    __private RandType t[RAND_BUF_LEN], __global RandType* rngseed,
                    __constant float4* gproperty, __global const uint* media, __global float* srcpattern,
                    __constant float4* gdetpos, __constant MCXParam* gcfg, int threadid,
                    __local int* blockphoton, volatile __global uint* gprogress,
                    __local RandType* photonseed, __global RandType* gseeddata,
                    __global uint* gjumpdebug, __global float* gdebugdata, __local RandType* sharedmem);

#if defined(MCX_DEBUG_MOVE) || defined(MCX_DEBUG_MOVE_ONLY)
    void savedebugdata(float4* p, uint id, __global uint* gjumpdebug, __global float* gdebugdata, __constant MCXParam* gcfg);
#endif

#ifdef USE_ATOMIC

#if defined(USE_NVIDIA_GPU) && !defined(USE_OPENCL_ATOMIC)
// float atomicadd on NVIDIA GPU via PTX
// https://stackoverflow.com/a/72049624/4271392

inline float atomicadd(volatile __global float* address, const float value) {
    float old;
    asm volatile(
        "atom.global.add.f32 %0, [%1], %2;"
        : "=f"(old)
        : "l"(address), "f"(value)
        : "memory"
    );
    return old;
}
#else
// OpenCL float atomicadd hack:
// http://suhorukov.blogspot.co.uk/2011/12/opencl-11-atomic-operations-on-floating.html
// https://devtalk.nvidia.com/default/topic/458062/atomicadd-float-float-atomicmul-float-float-/

inline float atomicadd(volatile __global float* address, const float value) {
    float old = value, orig;

    while ((old = atomic_xchg(address, (orig = atomic_xchg(address, 0.0f)) + old)) != 0.0f);

    return orig;
}
#endif

#endif

void clearpath(__local float* p, uint maxmediatype) {
    uint i;

    for (i = 0; i < maxmediatype; i++) {
        p[i] = 0.f;
    }
}

#ifdef MCX_SAVE_DETECTORS
uint finddetector(float4* p0, __constant float4* gdetpos, __constant MCXParam* gcfg) {
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

void saveexitppath(__global float* n_det, __local float* ppath, float4* p0, uint* idx1d, __constant MCXParam* gcfg) {
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

void savedetphoton(__global float* n_det, __global uint* detectedphoton,
                   __local float* ppath, float4* p0, float4* v,
                   __local RandType* t, __global RandType* seeddata,
                   __constant float4* gdetpos, __constant MCXParam* gcfg, uint isdet) {
    int detid;
    detid = (isdet == OUTSIDE_VOLUME_MIN) ? -1 : (int)finddetector(p0, gdetpos, gcfg);

    if (detid) {
        uint baseaddr = atomic_inc(detectedphoton);

        if (baseaddr < GPU_PARAM(gcfg, maxdetphoton)) {
            uint i;

            for (i = 0; i < GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN; i++) {
                seeddata[baseaddr * RAND_BUF_LEN + i] = t[i];    ///< save photon seed for replay
            }

            baseaddr *= GPU_PARAM(gcfg, reclen);

            if (SAVE_DETID(GPU_PARAM(gcfg, savedetflag))) {
                n_det[baseaddr++] = detid;
            }

            for (i = 0; i < GPU_PARAM(gcfg, partialdata); i++) {
                n_det[baseaddr++] = ppath[i];    ///< save partial pathlength to the memory
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
                n_det[baseaddr++] = ppath[GPU_PARAM(gcfg, w0offset) - 1];
            }
        } else if (GPU_PARAM(gcfg, savedet) == FILL_MAXDETPHOTON) {
            atomic_dec(detectedphoton);
        }
    }
}
#endif

#if defined(MCX_DEBUG_MOVE) || defined(MCX_DEBUG_MOVE_ONLY)
/**
 * @brief Saving photon trajectory data for debugging purposes
 * @param[in] p: the position/weight of the current photon packet
 * @param[in] id: the global index of the photon
 * @param[in] gdebugdata: pointer to the global-memory buffer to store the trajectory info
 */

void savedebugdata(float4* p, uint id, __global uint* gjumpdebug, __global float* gdebugdata, __constant MCXParam* gcfg) {
    uint pos = atomic_inc(gjumpdebug);

    if (pos < GPU_PARAM(gcfg, maxjumpdebug)) {
        pos *= MCX_DEBUG_REC_LEN;
        ((__global uint*)gdebugdata)[pos++] = id;
        gdebugdata[pos++] = p[0].x;
        gdebugdata[pos++] = p[0].y;
        gdebugdata[pos++] = p[0].z;
        gdebugdata[pos++] = p[0].w;
        gdebugdata[pos++] = 0;
    }
}
#endif

float mcx_nextafterf(float a, int dir) {
    union {
        float f;
        uint  i;
    } num;
    num.f = a + 1000.f;
    num.i += dir ^ (num.i & 0x80000000U);
    return num.f - 1000.f;
}

float hitgrid(float4* p0, float4* v, short4* id) {
    float dist;
    float4 htime;

    //time-of-flight to hit the wall in each direction

    htime = fabs(convert_float4(id[0]) - convert_float4(isgreater(v[0], ((float4)(0.f)))) - p0[0]);
    htime = fabs(native_divide(htime + (float4)EPS, v[0]));

    //get the direction with the smallest time-of-flight
    dist = fmin(fmin(htime.x, htime.y), htime.z);
    id->w = (dist == htime.x ? 0 : (dist == htime.y ? 1 : 2));

    return dist;
}

/**
 * @brief Compute 2D-scattering if the domain has a dimension of 1 in x/y or z
 *
 * This function performs 2D scattering calculation if the domain is only a sheet of voxels
 *
 * @param[in,out] v: the direction vector of the photon
 * @param[in] stheta: the sine of the rotation angle
 * @param[in] ctheta: the cosine of the rotation angle
 */

void rotatevector2d(float4* v, float stheta, float ctheta, int is2d) {
    if (is2d == 1)
        *((float4*)v) = (float4)(
                            0.f,
                            v[0].y * ctheta - v[0].z * stheta,
                            v[0].y * stheta + v[0].z * ctheta,
                            v[0].w
                        );
    else if (is2d == 2)
        *((float4*)v) = (float4)(
                            v[0].x * ctheta - v[0].z * stheta,
                            0.f,
                            v[0].x * stheta + v[0].z * ctheta,
                            v[0].w
                        );
    else if (is2d == 3)
        *((float4*)v) = (float4)(
                            v[0].x * ctheta - v[0].y * stheta,
                            v[0].x * stheta + v[0].y * ctheta,
                            0.f,
                            v[0].w
                        );

    GPUDEBUG(((__constant char*)"new dir: %10.5e %10.5e %10.5e\n", v[0].x, v[0].y, v[0].z));
}

/**
 * @brief Compute 3D-scattering direction vector
 *
 * This function updates the direction vector after a 3D scattering event
 *
 * @param[in,out] v: the direction vector of the photon
 * @param[in] stheta: the sine of the azimuthal angle
 * @param[in] ctheta: the cosine of the azimuthal angle
 * @param[in] sphi: the sine of the zenith angle
 * @param[in] cphi: the cosine of the zenith angle
 */

void rotatevector(float4* v, float stheta, float ctheta, float sphi, float cphi) {
    if ( v[0].z > -1.f + EPS && v[0].z < 1.f - EPS ) {
        float tmp0 = 1.f - v[0].z * v[0].z;
        float tmp1 = stheta * rsqrt(tmp0);
        *((float4*)v) = (float4)(
                            tmp1 * (v[0].x * v[0].z * cphi - v[0].y * sphi) + v[0].x * ctheta,
                            tmp1 * (v[0].y * v[0].z * cphi + v[0].x * sphi) + v[0].y * ctheta,
                            -tmp1 * tmp0 * cphi                          + v[0].z * ctheta,
                            v[0].w
                        );
    } else {
        v[0] = (float4)(stheta * cphi, stheta * sphi, (v[0].z > 0.f) ? ctheta : -ctheta, v[0].w);
    }

    v[0].xyz = v[0].xyz * rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z);
    GPUDEBUG(((__constant char*)"new dir: %10.5e %10.5e %10.5e\n", v[0].x, v[0].y, v[0].z));
}

void transmit(float4* v, float n1, float n2, short flipdir) {
    float tmp0 = n1 / n2;
    v[0].xyz *= tmp0;

    (flipdir == 0) ?
    (v[0].x = sqrt(1.f - v[0].y * v[0].y - v[0].z * v[0].z) * ((v[0].x > 0.f) - (v[0].x < 0.f))) :
    ((flipdir == 1) ?
     (v[0].y = sqrt(1.f - v[0].x * v[0].x - v[0].z * v[0].z) * ((v[0].y > 0.f) - (v[0].y < 0.f))) :
     (v[0].z = sqrt(1.f - v[0].x * v[0].x - v[0].y * v[0].y) * ((v[0].z > 0.f) - (v[0].z < 0.f))));
}

/**
 * @brief Calculating the reflection coefficient at an interface
 *
 * This function calculates the reflection coefficient at
 * an interface of different refrective indicex (n1/n2)
 *
 * @param[in] v: the direction vector of the photon
 * @param[in] n1: the refrective index of the voxel the photon leaves
 * @param[in] n2: the refrective index of the voxel the photon enters
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 * @return the reflection coefficient R=(Rs+Rp)/2, Rs: R of the perpendicularly polarized light, Rp: parallelly polarized light
 */

float reflectcoeff(float4* v, float n1, float n2, short flipdir) {
    float Icos = fabs((flipdir == 0) ? v[0].x : (flipdir == 1 ? v[0].y : v[0].z));
    float tmp0 = n1 * n1;
    float tmp1 = n2 * n2;
    float tmp2 = 1.f - tmp0 / tmp1 * (1.f - Icos * Icos); /** 1-[n1/n2*sin(si)]^2 = cos(ti)^2*/

    if (tmp2 > 0.f) { ///< partial reflection
        float Re, Im, Rtotal;
        Re = tmp0 * Icos * Icos + tmp1 * tmp2;
        tmp2 = sqrt(tmp2); /** to save one sqrt*/
        Im = 2.f * n1 * n2 * Icos * tmp2;
        Rtotal = (Re - Im) / (Re + Im); /** Rp*/
        Re = tmp1 * Icos * Icos + tmp0 * tmp2 * tmp2;
        Rtotal = (Rtotal + (Re - Im) / (Re + Im)) * 0.5f; /** (Rp+Rs)/2*/
        return Rtotal;
    } else { ///< total reflection
        return 1.f;
    }
}

/**
 * @brief Loading optical properties from constant memory
 *
 * This function parses the media input and load optical properties
 * from GPU memory
 *
 * @param[out] prop: pointer to the current optical properties {mua, mus, g, n}
 * @param[in] mediaid: the media ID (32 bit) of the current voxel, format is specified in GPU_PARAM(gcfg,mediaformat) or cfg->mediabyte
 */

void updateproperty(FLOAT4VEC* prop, unsigned int mediaid, __constant float4* gproperty, __constant MCXParam* gcfg) {
#if MED_TYPE <=4
    *((FLOAT4VEC*)(prop)) = gproperty[mediaid & MED_MASK];
#elif MED_TYPE==MEDIA_MUA_FLOAT
    prop[0].x = fabs(*((float*)&mediaid));
    prop[0].w = gproperty[(!(mediaid & MED_MASK)) == 0].w;
#elif MED_TYPE==MEDIA_AS_F2H || MED_TYPE==MEDIA_AS_HALF //< [h1][h0]: h1/h0: single-prec mua/mus for every voxel; g/n uses those in cfg.prop(2,:)
    union {
        unsigned int i;
        half h[2];
    } val;
    val.i = mediaid & MED_MASK;
    prop[0].x = fabs(convert_float(vload_half(0, val.h)));
    prop[0].y = fabs(convert_float(vload_half(1, val.h)));
    prop[0].w = gproperty[(!(mediaid & MED_MASK)) == 0].w;
#elif MED_TYPE==MEDIA_2LABEL_MIX //< [s1][c1][c0]: s1: (volume fraction of tissue 1)*(2^16-1), c1: tissue 1 label, c0: tissue 0 label
    union {
        unsigned int   i;
        unsigned short h[2];
        unsigned char  c[4];
    } val;
    val.i = mediaid & MED_MASK;

    if (val.h[1] > 0) {
        if ((rand_uniform01(t) * 32767.f) < val.h[1]) {
            *((FLOAT4VEC*)(prop)) = gproperty[val.c[1]];
            mediaid >>= 8;
        } else {
            *((FLOAT4VEC*)(prop)) = gproperty[val.c[0]];
        }

        mediaid &= 0xFFFF;
    } else {
        *((FLOAT4VEC*)(prop)) = gproperty[val.c[0]];
    }

#elif MED_TYPE==MEDIA_LABEL_HALF //< [h1][s0]: h1: half-prec property value; highest 2bit in s0: index 0-3, low 14bit: tissue label
    union {
        unsigned int i;
        half h[2];
        unsigned short s[2]; /**s[1]: half-prec property; s[0]: high 2bits: idx 0-3, low 14bits: tissue label*/
    } val;
    val.i = mediaid & MED_MASK;
    *((FLOAT4VEC*)(prop)) = gproperty[val.s[0] & 0x3FFF];
    float* p = (float*)(prop);
    p[(val.s[0] & 0xC000) >> 14] = fabs(convert_float(vload_half(1, val.h)));
#elif MED_TYPE==MEDIA_ASGN_BYTE
    union {
        unsigned int i;
        unsigned char h[4];
    } val;
    val.i = mediaid & MED_MASK;
    prop[0].x = val.h[0] * (1.f / 255.f) * (gproperty[2].x - gproperty[1].x) + gproperty[1].x;
    prop[0].y = val.h[1] * (1.f / 255.f) * (gproperty[2].y - gproperty[1].y) + gproperty[1].y;
    prop[0].z = val.h[2] * (1.f / 255.f) * (gproperty[2].z - gproperty[1].z) + gproperty[1].z;
    prop[0].w = val.h[3] * (1.f / 127.f) * (gproperty[2].w - gproperty[1].w) + gproperty[1].w;
#elif MED_TYPE==MEDIA_AS_SHORT
    union {
        unsigned int i;
        unsigned short h[2];
    } val;
    val.i = mediaid & MED_MASK;
    prop[0].x = val.h[0] * (1.f / 65535.f) * (gproperty[2].x - gproperty[1].x) + gproperty[1].x;
    prop[0].y = val.h[1] * (1.f / 65535.f) * (gproperty[2].y - gproperty[1].y) + gproperty[1].y;
    prop[0].w = gproperty[(!(mediaid & MED_MASK)) == 0].w;
#endif
}

#ifndef INTERNAL_SOURCE

/**
 * @brief Advance photon to the 1st non-zero voxel if launched in the backgruond
 *
 * This function advances the photon to the 1st non-zero voxel along the direction
 * of v if the photon is launched outside of the cubic domain or in a zero-voxel.
 * To avoid large overhead, photon can only advance GPU_PARAM(gcfg,minaccumtime) steps, which
 * can be set using the --maxvoidstep flag; by default, this limit is 1000.
 *
 * @param[in] v: the direction vector of the photon
 * @param[in] n1: the refrective index of the voxel the photon leaves
 * @param[in] n2: the refrective index of the voxel the photon enters
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 * @return the reflection coefficient R=(Rs+Rp)/2, Rs: R of the perpendicularly polarized light, Rp: parallelly polarized light
 */

int skipvoid(float4* p, float4* v, float4* f, short4* flipdir, __global const uint* media, __constant float4* gproperty, __constant MCXParam* gcfg) {
    int count = 1, idx1d;

    flipdir->xyz = convert_short3_rtn(p->xyz);
    flipdir->w = -1;

    while (1) {
        if ((ushort)flipdir->x < gcfg->maxidx.x && (ushort)flipdir->y < gcfg->maxidx.y && (ushort)flipdir->z < gcfg->maxidx.z) {
            idx1d = (flipdir->z * gcfg->dimlen.y + flipdir->y * gcfg->dimlen.x + flipdir->x);

            if (media[idx1d] & MED_MASK) { ///< if enters a non-zero voxel
                GPUDEBUG(("inside volume [%f %f %f] v=<%f %f %f>\n", p[0].x, p[0].y, p[0].z, v[0].x, v[0].y, v[0].z));
                p[0].xyz -= v[0].xyz;
                flipdir->xyz = convert_short3_rtn(p->xyz);
                f[0].y -= GPU_PARAM(gcfg, minaccumtime);
                idx1d = (flipdir->z * gcfg->dimlen.y + flipdir->y * gcfg->dimlen.x + flipdir->x);

                //GPUDEBUG(("look for entry p0=[%f %f %f] rv=[%f %f %f]\n",p[0].x,p[0].y,p[0].z,rv[0].x,rv[0].y,rv[0].z));
                count = 0;

                while (!((ushort)flipdir->x < gcfg->maxidx.x && (ushort)flipdir->y < gcfg->maxidx.y && (ushort)flipdir->z < gcfg->maxidx.z) || !(media[idx1d] & MED_MASK)) { // at most 3 times
                    float dist = hitgrid(p, v, flipdir);
                    f[0].y += GPU_PARAM(gcfg, minaccumtime) * dist;
                    p[0] = (float4)(p->x + dist * v->x, p->y + dist * v->y, p->z + dist * v->z, p[0].w);

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
                        GPUDEBUG(("fail to find entry point after 3 iterations, something is wrong, abort!!"));
                        break;
                    }
                }

                FLOAT4VEC htime;
                f[0].y = (GPU_PARAM(gcfg, voidtime)) ? f[0].y : 0.f;
                updateproperty(&htime, media[idx1d], gproperty, gcfg);

                if (GPU_PARAM(gcfg, isspecular) &&  htime.w != gproperty[0].w) {
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

        p[0] = (float4)(p[0].x + v[0].x, p[0].y + v[0].y, p[0].z + v[0].z, p[0].w);
        flipdir->xyz = convert_short3_rtn(p->xyz);
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
 *
 * This function terminates the current photon and launches a new photon according
 * to the source type selected. Currently, over a dozen source types are supported,
 * including pencil, isotropic, planar, fourierx, gaussian, zgaussian etc.
 *
 * @param[in,out] p: the 3D position and weight of the photon
 * @param[in,out] v: the direction vector of the photon
 * @param[in,out] f: the parameter vector of the photon
 * @param[in,out] rv: the reciprocal direction vector of the photon (rv[i]=1/v[i])
 * @param[out] prop: the optical properties of the voxel the photon is launched into
 * @param[in,out] idx1d: the linear index of the voxel containing the photon at launch
 * @param[in] field: the 3D array to store photon weights
 * @param[in,out] mediaid: the medium index at the voxel at launch
 * @param[in,out] w0: initial weight, reset here after launch
 * @param[in,out] Lmove: photon movement length variable, reset to 0 after launch
 * @param[in] isdet: whether the previous photon being terminated lands at a detector
 * @param[in,out] ppath: pointer to the shared-mem buffer to store photon partial-path data
 * @param[in,out] n_det: array in the constant memory where detector positions are stored
 * @param[in,out] dpnum: global-mem variable where the count of detected photons are stored
 * @param[in] t: RNG state
 * @param[in,out] photonseed: RNG state stored at photon's launch time if replay is needed
 * @param[in] media: domain medium index array, read-only
 * @param[in] srcpattern: user-specified source pattern array if pattern source is used
 * @param[in] threadid: the global index of the current thread
 * @param[in] rngseed: in the replay mode, pointer to the saved detected photon seed data
 * @param[in,out] gseeddata: pointer to the buffer to save detected photon seeds
 * @param[in,out] gdebugdata: pointer to the buffer to save photon trajectory positions
 * @param[in,out] gprogress: pointer to the host variable to update progress bar
 */

int launchnewphoton(float4* p, float4* v, float4* f, short4* flipdir, FLOAT4VEC* prop, uint* idx1d,
                    __global float* field, uint* mediaid, float* w0, float* Lmove, uint isdet,
                    __local float* ppath, __global float* n_det, __global uint* dpnum,
                    __private RandType t[RAND_BUF_LEN], __global RandType* rngseed,
                    __constant float4* gproperty, __global const uint* media, __global float* srcpattern,
                    __constant float4* gdetpos, __constant MCXParam* gcfg, int threadid,
                    __local int* blockphoton, volatile __global uint* gprogress,
                    __local RandType* photonseed, __global RandType* gseeddata,
                    __global uint* gjumpdebug, __global float* gdebugdata, __local RandType* sharedmem) {

    *w0 = 1.f;   ///< reuse to count for launchattempt
    *Lmove = -1.f; ///< reuse as "canfocus" flag for each source: non-zero: focusable, zero: not focusable


    /**
     * Early termination of simulation when the detphoton buffer is filled if issavedet is set to 3
     */
    if (GPU_PARAM(gcfg, savedet) == FILL_MAXDETPHOTON) {
        if (*dpnum >= GPU_PARAM(gcfg, maxdetphoton)) {
            gprogress[0] = (gcfg->threadphoton >> 1) * 4.5f;
            return 1;
        }
    }

    /**
     * First, let's terminate the current photon and perform detection calculations
     */
    if (fabs(p[0].w) >= 0.f) {
        ppath[GPU_PARAM(gcfg, partialdata)] += p[0].w; ///< sum all the remaining energy

#if defined(MCX_DEBUG_MOVE) || defined(MCX_DEBUG_MOVE_ONLY)
        savedebugdata(p, (uint)f[0].w + threadid * gcfg->threadphoton + min(threadid, gcfg->oddphoton), gjumpdebug, gdebugdata, gcfg);
#endif

        if (*mediaid == 0 && *idx1d != OUTSIDE_VOLUME_MIN && *idx1d != OUTSIDE_VOLUME_MAX && GPU_PARAM(gcfg, issaveref) && p[0].w > 0.f) {
            if (GPU_PARAM(gcfg, issaveref) == 1) {
                int tshift = MIN((int)GPU_PARAM(gcfg, maxgate) - 1, (int)(floor((f[0].y - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep))));
#if !defined(MCX_SRC_PATTERN) && !defined(MCX_SRC_PATTERN3D)
#ifdef USE_ATOMIC
                float oldval = atomicadd(& field[*idx1d + tshift * gcfg->dimlen.z], -p[0].w);

                if (fabs(oldval) > MAX_ACCUM) {
                    atomicadd(& field[*idx1d + tshift * gcfg->dimlen.z], ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                    atomicadd(& field[*idx1d + tshift * gcfg->dimlen.z + gcfg->dimlen.w], ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                }

#else
                field[*idx1d + tshift * gcfg->dimlen.z] += -p[0].w;
#endif
#else

                for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                    if (fabs(ppath[GPU_PARAM(gcfg, w0offset) + i]) > 0.f) {
#ifdef USE_ATOMIC
                        float oldval = atomicadd(& field[(*idx1d + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i], -((GPU_PARAM(gcfg, srcnum) == 1) ? p[0].w : p[0].w * ppath[GPU_PARAM(gcfg, w0offset) + i]));

                        if (fabs(oldval) > MAX_ACCUM) {
                            atomicadd(& field[(*idx1d + tshift * gcfg->dimlen.z)*GPU_PARAM(gcfg, srcnum) + i], ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                            atomicadd(& field[(*idx1d + tshift * gcfg->dimlen.z)*GPU_PARAM(gcfg, srcnum) + i + gcfg->dimlen.w], ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                        }

#else
                        field[(*idx1d + tshift * gcfg->dimlen.z)*GPU_PARAM(gcfg, srcnum) + i] += -((GPU_PARAM(gcfg, srcnum) == 1) ? p[0].w : p[0].w * ppath[GPU_PARAM(gcfg, w0offset) + i]);
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

        // let's handle detectors here
        if ((isdet & DET_MASK) == DET_MASK && *mediaid == 0 && (bool)(GPU_PARAM(gcfg, issaveref) < 2)) {
            savedetphoton(n_det, dpnum, ppath, p, v, photonseed, gseeddata, gdetpos, gcfg, isdet);
        }

#endif
    }

#ifdef MCX_SAVE_DETECTORS
    clearpath(ppath, GPU_PARAM(gcfg, partialdata));
#endif

#ifdef GROUP_LOAD_BALANCE
    /**
     * checking out a photon from the block's total workload, terminate if nothing left to run
     */
    GPUDEBUG(("block workload [%f] done over [%d]\n", f[0].w, blockphoton[0]));

    if (atomic_dec(blockphoton) < 1) {
        return 1;  // all photons assigned to the block are done
    }

#else
    /**
     * If the thread completes all assigned photons, terminate this thread.
     */
    GPUDEBUG(("thread workload [%f] done over [%d]\n", f[0].w, (gcfg->threadphoton + (threadid < gcfg->oddphoton))));

    if (f[0].w >= (gcfg->threadphoton + (threadid < gcfg->oddphoton))) {
        return 1;    // all photons complete
    }

#endif

    ppath += GPU_PARAM(gcfg, partialdata);

    /**
     * If this is a replay of a detected photon, initilize the RNG with the stored seed here.
     */

    if (GPU_PARAM(gcfg, seed) == SEED_FROM_FILE) {
        int seedoffset = (threadid * gcfg->threadphoton + min(threadid, gcfg->oddphoton - 1) + max(0, (int)f[0].w)) * RAND_BUF_LEN;

        for (int i = 0; i < RAND_BUF_LEN; i++) {
            t[i] = rngseed[seedoffset + i];
        }
    }

    /**
     * Attempt to launch a new photon until success
     */
    do {
        p[0] = gcfg->ps;
        v[0] = gcfg->c0;
        f[0] = (float4)(0.f, 0.f, GPU_PARAM(gcfg, minaccumtime), f[0].w);
        *idx1d = GPU_PARAM(gcfg, idx1dorig);
        *mediaid = GPU_PARAM(gcfg, mediaidorig);
        *prop = TOFLOAT4((float4)(gcfg->ps.x, gcfg->ps.y, gcfg->ps.z, 0)); ///< reuse as the origin of the src, needed for focusable sources

        if (GPU_PARAM(gcfg, issaveseed)) {
            copystate(t, photonseed);
        }

        /**
         * Only one branch is taken because of template, this can reduce thread divergence
         */

#if defined(MCX_SRC_PLANAR) || defined(MCX_SRC_PATTERN) || defined(MCX_SRC_PATTERN3D) || defined(MCX_SRC_FOURIER) || defined(MCX_SRC_PENCILARRAY) /*a rectangular grid over a plane*/
        float rx = rand_uniform01(t);
        float ry = rand_uniform01(t);
        float rz;
#if defined(MCX_SRC_PATTERN3D)
        rz = rand_uniform01(t);
        p[0] = (float4)(p[0].x + rx * gcfg->srcparam1.x,
                        p[0].y + ry * gcfg->srcparam1.y,
                        p[0].z + rz * gcfg->srcparam1.z,
                        p[0].w);
#else
        p[0] = (float4)(p[0].x + rx * gcfg->srcparam1.x + ry * gcfg->srcparam2.x,
                        p[0].y + rx * gcfg->srcparam1.y + ry * gcfg->srcparam2.y,
                        p[0].z + rx * gcfg->srcparam1.z + ry * gcfg->srcparam2.z,
                        p[0].w);
#endif
#if defined(MCX_SRC_PATTERN)  // need to prevent rx/ry=1 here

        if (GPU_PARAM(gcfg, srcnum) <= 1) {
            p[0].w = gcfg->ps.w * srcpattern[(int)(ry * JUST_BELOW_ONE * gcfg->srcparam2.w) * (int)(gcfg->srcparam1.w) + (int)(rx * JUST_BELOW_ONE * gcfg->srcparam1.w)];
            ppath[3] = p[0].w;
        } else {
            *((__local uint*)(ppath + 2)) = ((int)(ry * JUST_BELOW_ONE * gcfg->srcparam2.w) * (int)(gcfg->srcparam1.w) + (int)(rx * JUST_BELOW_ONE * gcfg->srcparam1.w));

            for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                ppath[i + 3] = srcpattern[(*((__local uint*)(ppath + 2))) * GPU_PARAM(gcfg, srcnum) + i];
            }

            p[0].w = 1.f;
        }

#elif defined(MCX_SRC_PATTERN3D)  // need to prevent rx/ry=1 here

        if (GPU_PARAM(gcfg, srcnum) <= 1) {
            p[0].w = gcfg->ps.w * srcpattern[(int)(rz * JUST_BELOW_ONE * gcfg->srcparam1.z) * (int)(gcfg->srcparam1.y) * (int)(gcfg->srcparam1.x) +
                                             (int)(ry * JUST_BELOW_ONE * gcfg->srcparam1.y) * (int)(gcfg->srcparam1.x) + (int)(rx * JUST_BELOW_ONE * gcfg->srcparam1.x)];
            ppath[3] = p[0].w;
        } else {
            *((__local uint*)(ppath + 2)) = ((int)(rz * JUST_BELOW_ONE * gcfg->srcparam1.z) * (int)(gcfg->srcparam1.y) * (int)(gcfg->srcparam1.x) +
                                             (int)(ry * JUST_BELOW_ONE * gcfg->srcparam1.y) * (int)(gcfg->srcparam1.x) + (int)(rx * JUST_BELOW_ONE * gcfg->srcparam1.x));

            for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                ppath[i + 3] = srcpattern[(*((__local uint*)(ppath + 2))) * GPU_PARAM(gcfg, srcnum) + i];
            }

            p[0].w = 1.f;
        }

#elif defined(MCX_SRC_FOURIER)  // need to prevent rx/ry=1 here
        p[0].w = gcfg->ps.w * (MCX_MATHFUN(cos)((floor(gcfg->srcparam1.w) * rx + floor(gcfg->srcparam2.w) * ry
                                                + gcfg->srcparam1.w - floor(gcfg->srcparam1.w)) * TWO_PI) * (1.f - gcfg->srcparam2.w + floor(gcfg->srcparam2.w)) + 1.f) * 0.5f; //between 0 and 1
#elif defined(MCX_SRC_PENCILARRAY)  // need to prevent rx/ry=1 here
        p[0].x = gcfg->ps.x + floor(rx * gcfg->srcparam1.w) * gcfg->srcparam1.x / (gcfg->srcparam1.w - 1.f) + floor(ry * gcfg->srcparam2.w) * gcfg->srcparam2.x / (gcfg->srcparam2.w - 1.f);
        p[0].y = gcfg->ps.y + floor(rx * gcfg->srcparam1.w) * gcfg->srcparam1.y / (gcfg->srcparam1.w - 1.f) + floor(ry * gcfg->srcparam2.w) * gcfg->srcparam2.y / (gcfg->srcparam2.w - 1.f);
        p[0].z = gcfg->ps.z + floor(rx * gcfg->srcparam1.w) * gcfg->srcparam1.z / (gcfg->srcparam1.w - 1.f) + floor(ry * gcfg->srcparam2.w) * gcfg->srcparam2.z / (gcfg->srcparam2.w - 1.f);
#endif
        *idx1d = ((int)(floor(p[0].z)) * gcfg->dimlen.y + (int)(floor(p[0].y)) * gcfg->dimlen.x + (int)(floor(p[0].x)));

        if (p[0].x < 0.f || p[0].y < 0.f || p[0].z < 0.f || p[0].x >= gcfg->maxidx.x || p[0].y >= gcfg->maxidx.y || p[0].z >= gcfg->maxidx.z) {
            *mediaid = 0;
        } else {
            *mediaid = media[*idx1d];
        }

        *prop = TOFLOAT4((float4)(prop[0].x + (gcfg->srcparam1.x + gcfg->srcparam2.x) * 0.5f,
                                  prop[0].y + (gcfg->srcparam1.y + gcfg->srcparam2.y) * 0.5f,
                                  prop[0].z + (gcfg->srcparam1.z + gcfg->srcparam2.z) * 0.5f, 0.f));
#elif defined(MCX_SRC_FOURIERX) || defined(MCX_SRC_FOURIERX2D) // [v1x][v1y][v1z][|v2|]; [kx][ky][phi0][M], unit(v0) x unit(v1)=unit(v2)
        float rx = rand_uniform01(t);
        float ry = rand_uniform01(t);
        float4 v2 = gcfg->srcparam1;
        // calculate v2 based on v2=|v2| * unit(v0) x unit(v1)
        v2.w *= rsqrt(gcfg->srcparam1.x * gcfg->srcparam1.x + gcfg->srcparam1.y * gcfg->srcparam1.y + gcfg->srcparam1.z * gcfg->srcparam1.z);
        v2.x = v2.w * (gcfg->c0.y * gcfg->srcparam1.z - gcfg->c0.z * gcfg->srcparam1.y);
        v2.y = v2.w * (gcfg->c0.z * gcfg->srcparam1.x - gcfg->c0.x * gcfg->srcparam1.z);
        v2.z = v2.w * (gcfg->c0.x * gcfg->srcparam1.y - gcfg->c0.y * gcfg->srcparam1.x);
        p[0] = (float4)(p[0].x + rx * gcfg->srcparam1.x + ry * v2.x,
                        p[0].y + rx * gcfg->srcparam1.y + ry * v2.y,
                        p[0].z + rx * gcfg->srcparam1.z + ry * v2.z,
                        p[0].w);
#if defined(MCX_SRC_FOURIERX2D)
        p[0].w = gcfg->ps.w * (MCX_MATHFUN(sin)((gcfg->srcparam2.x * rx + gcfg->srcparam2.z) * TWO_PI) * MCX_MATHFUN(sin)((gcfg->srcparam2.y * ry + gcfg->srcparam2.w) * TWO_PI) + 1.f) * 0.5f; //between 0 and 1
#else
        p[0].w = gcfg->ps.w * (MCX_MATHFUN(cos)((gcfg->srcparam2.x * rx + gcfg->srcparam2.y * ry + gcfg->srcparam2.z) * TWO_PI) * (1.f - gcfg->srcparam2.w) + 1.f) * 0.5f; //between 0 and 1
#endif

        *idx1d = ((int)(floor(p[0].z)) * gcfg->dimlen.y + (int)(floor(p[0].y)) * gcfg->dimlen.x + (int)(floor(p[0].x)));

        if (p[0].x < 0.f || p[0].y < 0.f || p[0].z < 0.f || p[0].x >= gcfg->maxidx.x || p[0].y >= gcfg->maxidx.y || p[0].z >= gcfg->maxidx.z) {
            *mediaid = 0;
        } else {
            *mediaid = media[*idx1d];
        }

        *prop = TOFLOAT4((float4)(prop[0].x + (gcfg->srcparam1.x + v2.x) * 0.5f,
                                  prop[0].y + (gcfg->srcparam1.y + v2.y) * 0.5f,
                                  prop[0].z + (gcfg->srcparam1.z + v2.z) * 0.5f, 0.f));
#elif defined(MCX_SRC_DISK) || defined(MCX_SRC_GAUSSIAN) // uniform disk distribution or Gaussian-beam
        // Uniform disk point picking
        // http://mathworld.wolfram.com/DiskPointPicking.html
        float sphi, cphi;
        float phi = TWO_PI * rand_uniform01(t);
        MCX_SINCOS(phi, sphi, cphi);
        float r;
#if defined(MCX_SRC_DISK)
        r = sqrt(rand_uniform01(t)) * gcfg->srcparam1.x;
#else

        if (fabs(gcfg->c0.w) < 1e-5f || fabs(gcfg->srcparam1.y) < 1e-5f) {
            r = sqrt(-0.5f * log(rand_uniform01(t))) * gcfg->srcparam1.x;
        } else {
            r = gcfg->srcparam1.x * gcfg->srcparam1.x * M_PI / gcfg->srcparam1.y; //Rayleigh range
            r = sqrt(-0.5f * log(rand_uniform01(t)) * (1.f + (gcfg->c0.w * gcfg->c0.w / (r * r)))) * gcfg->srcparam1.x;
        }

#endif

        if ( v[0].z > -1.f + EPS && v[0].z < 1.f - EPS ) {
            float tmp0 = 1.f - v[0].z * v[0].z;
            float tmp1 = r * rsqrt(tmp0);
            p[0] = (float4)(
                       p[0].x + tmp1 * (v[0].x * v[0].z * cphi - v[0].y * sphi),
                       p[0].y + tmp1 * (v[0].y * v[0].z * cphi + v[0].x * sphi),
                       p[0].z - tmp1 * tmp0 * cphi,
                       p[0].w
                   );
            GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n", v[0].x, v[0].y, v[0].z));
        } else {
            p[0].x += r * cphi;
            p[0].y += r * sphi;
            GPUDEBUG(("new dir-z: %10.5e %10.5e %10.5e\n", v[0].x, v[0].y, v[0].z));
        }

        *idx1d = ((int)(floor(p[0].z)) * gcfg->dimlen.y + (int)(floor(p[0].y)) * gcfg->dimlen.x + (int)(floor(p[0].x)));

        if (p[0].x < 0.f || p[0].y < 0.f || p[0].z < 0.f || p[0].x >= gcfg->maxidx.x || p[0].y >= gcfg->maxidx.y || p[0].z >= gcfg->maxidx.z) {
            *mediaid = 0;
        } else {
            *mediaid = media[*idx1d];
        }

#elif defined(MCX_SRC_CONE) || defined(MCX_SRC_ISOTROPIC) || defined(MCX_SRC_ARCSINE)
        // Uniform point picking on a sphere
        // http://mathworld.wolfram.com/SpherePointPicking.html
        float ang, stheta, ctheta, sphi, cphi;
        ang = TWO_PI * rand_uniform01(t); //next arimuth angle
        MCX_SINCOS(ang, sphi, cphi);
#if defined(MCX_SRC_CONE) // a solid-angle section of a uniform sphere

        do {
            ang = (gcfg->srcparam1.y > 0) ? TWO_PI * rand_uniform01(t) : acos(2.f * rand_uniform01(t) - 1.f); //sine distribution
        } while (ang > gcfg->srcparam1.x);

#else
#if defined(MCX_SRC_ISOTROPIC) // a solid-angle section of a uniform sphere
        ang = acos(2.f * rand_uniform01(t) - 1.f); //sine distribution
#else
        ang = ONE_PI * rand_uniform01(t); //uniform distribution in zenith angle, arcsine
#endif
#endif
        MCX_SINCOS(ang, stheta, ctheta);
        rotatevector(v, stheta, ctheta, sphi, cphi);
        *Lmove = 0.f;
#elif defined(MCX_SRC_ZGAUSSIAN)
        float ang, stheta, ctheta, sphi, cphi;
        ang = TWO_PI * rand_uniform01(t); //next arimuth angle
        MCX_SINCOS(ang, sphi, cphi);
        ang = sqrt(-2.f * log(rand_uniform01(t))) * (1.f - 2.f * rand_uniform01(t)) * gcfg->srcparam1.x;
        MCX_SINCOS(ang, stheta, ctheta);
        rotatevector(v, stheta, ctheta, sphi, cphi);
        *Lmove = 0.f;
#elif defined(MCX_SRC_LINE) || defined(MCX_SRC_SLIT)
        float r = rand_uniform01(t);
        p[0] = (float4)(p[0].x + r * gcfg->srcparam1.x,
                        p[0].y + r * gcfg->srcparam1.y,
                        p[0].z + r * gcfg->srcparam1.z,
                        p[0].w);
#if defined(MCX_SRC_LINE)
        float s, q;
        r = 1.f - 2.f * rand_uniform01(t);
        s = 1.f - 2.f * rand_uniform01(t);
        q = sqrt(1.f - v[0].x * v[0].x - v[0].y * v[0].y) * (rand_uniform01(t) > 0.5f ? 1.f : -1.f);
        v[0] = (float4)(v[0].y * q - v[0].z * s, v[0].z * r - v[0].x * q, v[0].x * s - v[0].y * r, v[0].w);
        *Lmove = 0.f;
#else
        *Lmove = -1.f;
#endif
        *prop = TOFLOAT4((float4)(prop[0].x + (gcfg->srcparam1.x) * 0.5f,
                                  prop[0].y + (gcfg->srcparam1.y) * 0.5f,
                                  prop[0].z + (gcfg->srcparam1.z) * 0.5f, 0.f));
#endif
        /**
         * If beam focus is set, determine the incident angle
         */

        if (fabs(p[0].w) <= GPU_PARAM(gcfg, minenergy)) {
            continue;
        }

        if (GPU_PARAM(gcfg, nangle)) {
            /**
             * If angleinvcdf is defined, use user defined launch zenith angle distribution
             */

            float ang, stheta, ctheta, sphi, cphi;

            if ( gcfg->c0.w > 0.f) { // if focal-length > 0, no interpolation, just read the angleinvcdf value
                ang = fmin(rand_uniform01(t) * GPU_PARAM(gcfg, nangle), GPU_PARAM(gcfg, nangle) - EPS);
                cphi = ((__local float*)(sharedmem))[(int)(ang) + GPU_PARAM(gcfg, nphaselen)];
            } else { // odd number length, interpolate between neigboring values
                ang = fmin(rand_uniform01(t) * (GPU_PARAM(gcfg, nangle) - 1), GPU_PARAM(gcfg, nangle) - 1 - EPS);
                sphi = ang - ((int)ang);
                cphi = ((1.f - sphi) * (((__local float*)(sharedmem))[((int)ang >= GPU_PARAM(gcfg, nangle) - 1 ? GPU_PARAM(gcfg, nangle) - 1 : (int)(ang)) + GPU_PARAM(gcfg, nphaselen)]) +
                        sphi * (((__local float*)(sharedmem))[((int)ang + 1 >= GPU_PARAM(gcfg, nangle) - 1 ? GPU_PARAM(gcfg, nangle) - 1 : (int)(ang) + 1) + GPU_PARAM(gcfg, nphaselen)]));
            }

            cphi *= ONE_PI; // next zenith angle computed based on angleinvcdf
            MCX_SINCOS(cphi, stheta, ctheta);
            ang = TWO_PI * rand_uniform01(t); //next arimuth angle
            MCX_SINCOS(ang, sphi, cphi);

            if ( gcfg->c0.w < 1.5f &&  gcfg->c0.w >= 0.f) {
                *((float4*)v) =  gcfg->c0;
            }

            rotatevector(v, stheta, ctheta, sphi, cphi);
        } else if (*Lmove < 0.f) {
            /**
             * If beam focus is set, determine the incident angle
             */
            if (isnan(gcfg->c0.w)) { // isotropic if focal length is -0.f
                float ang, stheta, ctheta, sphi, cphi;
                ang = TWO_PI * rand_uniform01(t); //next arimuth angle
                MCX_SINCOS(ang, sphi, cphi);
                ang = acos(2.f * rand_uniform01(t) - 1.f); //sine distribution
                MCX_SINCOS(ang, stheta, ctheta);
                rotatevector(v, stheta, ctheta, sphi, cphi);
            } else if (gcfg->c0.w != 0.f) {
                float Rn2 = (gcfg->c0.w > 0.f) - (gcfg->c0.w < 0.f);
                prop[0].x += gcfg->c0.w * v[0].x;
                prop[0].y += gcfg->c0.w * v[0].y;
                prop[0].z += gcfg->c0.w * v[0].z;
                v[0].x = Rn2 * (prop[0].x - p[0].x);
                v[0].y = Rn2 * (prop[0].y - p[0].y);
                v[0].z = Rn2 * (prop[0].z - p[0].z);
                Rn2 = rsqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[0].z * v[0].z); // normalize
                v[0].x *= Rn2;
                v[0].y *= Rn2;
                v[0].z *= Rn2;
            }
        }

        /**
         * Compute the reciprocal of the velocity vector
         */
        //          *rv=float3(native_divide(1.f,v[0].x),native_divide(1.f,v[0].y),native_divide(1.f,v[0].z));

#ifndef INTERNAL_SOURCE

        /**
         * If a photon is launched outside of the box, or inside a zero-voxel, move it until it hits a non-zero voxel
         */
        if ((*mediaid & MED_MASK) == 0) {
            int idx = skipvoid(p, v, f, flipdir, media, gproperty, gcfg); /** specular reflection of the bbx is taken care of here*/

            if (idx >= 0) {
                *idx1d = idx;
                *mediaid = media[*idx1d];
            }
        }

#endif
        flipdir->xyz = convert_short3_rtn(p->xyz);

        *w0 += 1.f;

        /**
             * if launch attempted for over 1000 times, stop trying and return
             */
        if (*w0 > GPU_PARAM(gcfg, maxvoidstep)) {
            return -1;    // launch failed
        }
    } while ((*mediaid & MED_MASK) == 0 || fabs(p[0].w) <= GPU_PARAM(gcfg, minenergy));

    /**
     * Now a photon is successfully launched, perform necssary initialization for a new trajectory
     */
    f[0].w += 1.f;
    *prop = TOFLOAT4(gproperty[1]);
    updateproperty(prop, *mediaid, gproperty, gcfg);

#if defined(MCX_DEBUG_MOVE) || defined(MCX_DEBUG_MOVE_ONLY)
    savedebugdata(p, (uint)f[0].w + threadid * gcfg->threadphoton + min(threadid, (threadid < gcfg->oddphoton)*threadid), gjumpdebug, gdebugdata, gcfg);
#endif

    /**
     * total energy enters the volume. for diverging/converting
     * beams, this is less than nphoton due to specular reflection
     * loss. This is different from the wide-field MMC, where the
     * total launched energy includes the specular reflection loss
     */
    ppath[1] += p[0].w;
    *w0 = p[0].w;
    ppath[2] = ((GPU_PARAM(gcfg, srcnum) > 1) ? ppath[2] : p[0].w); // store initial weight
    v[0].w = EPS;
    *Lmove = 0.f;

    /**
     * If a progress bar is needed, only sum completed photons from the 1st, last and middle threads to determine progress bar
     */
#ifdef MCX_DEBUG_PROGRESS
#ifndef MCX_USE_CPU

    if (((int)(f[0].w) & 1) && (threadid == 0 || threadid == (int)(get_global_size(0) - 1) || threadid == (int)(get_global_size(0) >> 2)
                                || threadid == (int)(get_global_size(0) >> 1) || threadid == (int)(get_global_size(0) >> 2) + (int)(get_global_size(0) >> 1))) {
        ///< use the 1st, 1/4, 1/2, 3/4 and last thread for progress report
        gprogress[0]++;
    }

#else
    gprogress[0]++;
#endif
#endif
    return 0;
}

/*
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
*/
__kernel void mcx_main_loop(__global const uint* media,
                            __global float* field, __global float* genergy, __global uint* n_seed,
                            __global float* n_det, __constant float4* gproperty, __global float* srcpattern,
                            __constant float4* gdetpos, volatile __global uint* gprogress, __global uint* detectedphoton,
                            __global float* replayweight, __global float* photontof, __global int* photondetid,
                            __global RandType* gseeddata, __global uint* gjumpdebug, __global float* gdebugdata,
                            __global float* ginvcdf, __global float* gangleinvcdf, __local RandType* sharedmem, __constant MCXParam* gcfg) {

    int idx = get_global_id(0);

    float4 p = {0.f, 0.f, 0.f, NAN}; //< {x,y,z}: x,y,z coordinates,{w}:packet weight
    float4 v = gcfg->c0; //< {x,y,z}: ix,iy,iz unitary direction vector, {w}:total scat event
    float4 f = {0.f, 0.f, 0.f, 0.f}; //< Photon parameter state: x/pscat: remaining scattering probability, y/t: photon elapse time, z/pathlen: total pathlen in one voxel, w/ndone: completed photons

    uint idx1d, idx1dold;   //idx1dold is related to reflection

    uint   mediaid = GPU_PARAM(gcfg, mediaidorig), mediaidold = 0, isdet = 0;
    float  w0, Lmove, pathlen = 0.f;
    float  n1;   //reflection var
    short4 flipdir = {0, 0, 0, -1};

    RandType t[RAND_BUF_LEN];
    FLOAT4VEC prop;    //can become float2 if no reflection

    __local float* ppath = (__local float*)sharedmem;
    __local int   blockphoton[1];

    /**
     *  Load use-defined phase function (inversion of CDF) to the shared memory (first GPU_PARAM(gcfg, nphase) floats)
     */
    if (GPU_PARAM(gcfg, nphase)) {
        idx1d = GPU_PARAM(gcfg, nphase) / get_local_size(0);

        for (idx1dold = 0; idx1dold < idx1d; idx1dold++) {
            ppath[get_local_id(0) * idx1d + idx1dold] = ginvcdf[get_local_id(0) * idx1d + idx1dold];
        }

        if (GPU_PARAM(gcfg, nphase) - (idx1d * get_local_size(0)) > 0 && get_local_id(0) == 0) {
            for (idx1dold = 0; idx1dold < GPU_PARAM(gcfg, nphase) - (idx1d * get_local_size(0)) ; idx1dold++) {
                ppath[get_local_size(0) * idx1d + idx1dold] = ginvcdf[get_local_size(0) * idx1d + idx1dold];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /**
     *  Load use-defined launch angle function (inversion of CDF) to the shared memory (second GPU_PARAM(gcfg, nangle) floats)
     */
    if (GPU_PARAM(gcfg, nangle)) {
        idx1d = GPU_PARAM(gcfg, nangle) / get_local_size(0);

        for (idx1dold = 0; idx1dold < idx1d; idx1dold++) {
            ppath[get_local_id(0) * idx1d + idx1dold + GPU_PARAM(gcfg, nphaselen)] = gangleinvcdf[get_local_id(0) * idx1d + idx1dold];
        }

        if (GPU_PARAM(gcfg, nangle) - (idx1d * get_local_size(0)) > 0 && get_local_id(0) == 0) {
            for (idx1dold = 0; idx1dold < GPU_PARAM(gcfg, nangle) - (idx1d * get_local_size(0)) ; idx1dold++) {
                ppath[get_local_size(0) * idx1d + idx1dold + GPU_PARAM(gcfg, nphaselen)] = gangleinvcdf[get_local_size(0) * idx1d + idx1dold];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx >= gcfg->threadphoton * (get_local_size(0) * get_num_groups(0)) + gcfg->oddphoton) {
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

    ppath += get_local_id(0) * (GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum)); // block#2: maxmedia*thread number to store the partial
    clearpath(ppath, GPU_PARAM(gcfg, w0offset) + GPU_PARAM(gcfg, srcnum));
    ppath[GPU_PARAM(gcfg, partialdata)]  = genergy[idx << 1];
    ppath[GPU_PARAM(gcfg, partialdata) + 1] = genergy[(idx << 1) + 1];

    gpu_rng_init(t, n_seed, idx);

#if defined(MCX_DEBUG_RNG)

    for (int i = idx; i < gcfg->dimlen.w; i += get_global_size(0)) {
        field[i] = rand_uniform01(t);
    }

    return;
#endif

    if (launchnewphoton(&p, &v, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove, 0, ppath,
                        n_det, detectedphoton, t, (__global RandType*)n_seed, gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton,
                        gprogress, (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen)) + get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)),
                        gseeddata, gjumpdebug, gdebugdata, sharedmem)) {
        n_seed[idx] = NO_LAUNCH;
        return;
    }

    isdet = mediaid & DET_MASK;
    mediaid &= MED_MASK; // keep isdet to 0 to avoid launching photon ina

#ifdef GROUP_LOAD_BALANCE

    while (1) {
        GPUDEBUG(("block workload [%d] left\n", blockphoton[0]));
#else

    while (f.w <= gcfg->threadphoton + (idx < gcfg->oddphoton)) {
#endif
        GPUDEBUG(((__constant char*)"photonid [%d] L=%f w=%e medium=%d\n", (int)f.w, f.x, p.w, mediaid));

        if (f.x <= 0.f) { // if this photon has finished the current jump
            f.x = rand_next_scatlen(t);

            GPUDEBUG(((__constant char*)"scat L=%f RNG=[%e %e %e] \n", f.x, rand_next_aangle(t), rand_next_zangle(t), rand_uniform01(t)));

            if (v.w != EPS) { //weight
                //random arimuthal angle
                float cphi = 1.f, sphi = 0.f, theta, stheta, ctheta;
                float tmp0 = 0.f;

                if (!GPU_PARAM(gcfg, is2d)) {
                    tmp0 = TWO_PI * rand_next_aangle(t); //next arimuth angle
                    MCX_SINCOS(tmp0, sphi, cphi);
                }

                GPUDEBUG(("scat phi=%f\n", tmp0));

                if (GPU_PARAM(gcfg, nphase) > 2) { // after padding the left/right ends, nphase must be 3 or more
                    tmp0 = rand_uniform01(t) * (GPU_PARAM(gcfg, nphase) - 1);
                    theta = tmp0 - ((int)tmp0);
                    tmp0 = (1.f - theta) * ((__local float*)(sharedmem))[(int)tmp0   >= GPU_PARAM(gcfg, nphase) ? GPU_PARAM(gcfg, nphase) - 1 : (int)(tmp0)  ] +
                           theta * ((__local float*)(sharedmem))[(int)tmp0 + 1 >= GPU_PARAM(gcfg, nphase) ? GPU_PARAM(gcfg, nphase) - 1 : (int)(tmp0) + 1];
                    theta = acos(tmp0);
                    stheta = MCX_MATHFUN(sin)(theta);
                    ctheta = tmp0;
                } else {
                    tmp0 = (v.w > GPU_PARAM(gcfg, gscatter)) ? 0.f : prop.z;

                    /** Here we use Henyey-Greenstein Phase Function, "Handbook of Optical Biomedical Diagnostics",2002,Chap3,p234, also see Boas2002 */
                    if (fabs(tmp0) > EPS) { //< if prop.g is too small, the distribution of theta is bad
                        tmp0 = (1.f - prop.z * prop.z) / (1.f - prop.z + 2.f * prop.z * rand_next_zangle(t));
                        tmp0 *= tmp0;
                        tmp0 = (1.f + prop.z * prop.z - tmp0) / (2.f * prop.z);

                        // in early CUDA, when ran=1, CUDA gives 1.000002 for tmp0 which produces nan later
                        // detected by Ocelot,thanks to Greg Diamos,see http://bit.ly/cR2NMP
                        tmp0 = fmax(-1.f, fmin(1.f, tmp0));

                        theta = acos(tmp0);
                        stheta = MCX_MATHFUN(sin)(theta);
                        ctheta = tmp0;
                    } else {
                        theta = acos(2.f * rand_next_zangle(t) - 1.f);
                        MCX_SINCOS(theta, stheta, ctheta);
                    }
                }

                GPUDEBUG(((__constant char*)"scat theta=%f\n", theta));

#ifdef MCX_SAVE_DETECTORS

                if (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag))) {
                    ppath[(mediaid & MED_MASK) - 1]++;
                }

                /** accummulate momentum transfer */
                if (SAVE_MOM(GPU_PARAM(gcfg, savedetflag))) {
                    ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag)) + SAVE_PPATH(GPU_PARAM(gcfg, savedetflag))) + (mediaid & MED_MASK) - 1] += 1.f - ctheta;
                }

#endif

                /** Update direction vector with the two random angles */
                if (GPU_PARAM(gcfg, is2d)) {
                    rotatevector2d(&v, (rand_next_aangle(t) > 0.5f ? stheta : -stheta), ctheta, GPU_PARAM(gcfg, is2d));
                } else {
                    rotatevector(&v, stheta, ctheta, sphi, cphi);
                }

                v.w += 1.f;

                if ((bool)(GPU_PARAM(gcfg, outputtype) == otWP) || (bool)(GPU_PARAM(gcfg, outputtype) == otDCS)) {
                    ///< photontof[] and replayweight[] should be cached using local mem to avoid global read
                    int tshift = (idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w);
                    tmp0 = (GPU_PARAM(gcfg, outputtype) == otDCS) ? (1.f - ctheta) : 1.f;
                    tshift = (int)(floor((photontof[tshift] - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep))) +
                             ( (GPU_PARAM(gcfg, replaydet) == -1) ? ((photondetid[tshift] - 1) * GPU_PARAM(gcfg, maxgate)) : 0);
#ifndef USE_ATOMIC
                    field[idx1d + tshift * gcfg->dimlen.z] += tmp0 * replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)];
#else
                    float oldval = atomicadd(& field[idx1d + tshift * gcfg->dimlen.z], tmp0 * replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)]);

                    if (fabs(oldval) > MAX_ACCUM) {
                        if (atomicadd(& field[idx1d + tshift * gcfg->dimlen.z], -oldval) < 0.f) {
                            atomicadd(& field[idx1d + tshift * gcfg->dimlen.z], oldval);
                        } else {
                            atomicadd(& field[idx1d + tshift * gcfg->dimlen.z + gcfg->dimlen.w], oldval);
                        }
                    }

                    GPUDEBUG(("atomic write to [%d] %e, w=%f\n", idx1d, tmp0 * replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w)], p.w));
#endif
                }

#if defined(MCX_DEBUG_MOVE) || defined(MCX_DEBUG_MOVE_ONLY)
                savedebugdata(&p, (uint)f.w + idx * gcfg->threadphoton + min(idx, (idx < gcfg->oddphoton)*idx), gjumpdebug, gdebugdata, gcfg);
#endif
            }

            v.w = (int)v.w;
        }

        n1 = prop.w;
        updateproperty(&prop, mediaid, gproperty, gcfg);

        f.z = hitgrid(&p, &v, &flipdir);
        float slen = f.z * prop.y * (v.w + 1.f > GPU_PARAM(gcfg, gscatter) ? (1.f - prop.z) : 1.f); //unitless (minstep=grid, mus=1/grid)
        slen = fmin(slen, f.x);
        f.z = native_divide(slen, prop.y * (v.w + 1.f > GPU_PARAM(gcfg, gscatter) ? (1.f - prop.z) : 1.f ));
        pathlen += f.z;

        GPUDEBUG(((__constant char*)"p=[%f %f %f] -> <%f %f %f>*%f -> flip=%d\n", p.x, p.y, p.z, v.x, v.y, v.z, f.z, flipdir.w));

        p.xyz = p.xyz + (float3)(f.z) * v.xyz;

        if (flipdir.w == 0) {
            flipdir.x += (slen == f.x) ? 0 : (v.x > 0.f ? 1 : -1);
        }

        if (flipdir.w == 1) {
            flipdir.y += (slen == f.x) ? 0 : (v.y > 0.f ? 1 : -1);
        }

        if (flipdir.w == 2) {
            flipdir.z += (slen == f.x) ? 0 : (v.z > 0.f ? 1 : -1);
        }

        p.w *= MCX_MATHFUN(exp)(-prop.x * f.z);
        f.x -= slen;
        f.y += f.z * prop.w * GPU_PARAM(gcfg, oneoverc0);

        GPUDEBUG(((__constant char*)"update p=[%f %f %f] -> f.z=%f\n", p.x, p.y, p.z, f.z));

#ifdef MCX_SAVE_DETECTORS

        if (GPU_PARAM(gcfg, savedet) && SAVE_PPATH(GPU_PARAM(gcfg, savedetflag))) {
            ppath[GPU_PARAM(gcfg, maxmedia) * (SAVE_NSCAT(GPU_PARAM(gcfg, savedetflag))) + (mediaid & MED_MASK) - 1] += f.z;    //(unit=grid)
        }

#endif

        mediaidold = mediaid | isdet;
        idx1dold = idx1d;
        idx1d = (flipdir.z * gcfg->dimlen.y + flipdir.y * gcfg->dimlen.x + flipdir.x);
        GPUDEBUG(((__constant char*)"idx1d [%d]->[%d]\n", idx1dold, idx1d));

        if ((ushort)flipdir.x >= gcfg->maxidx.x || (ushort)flipdir.y >= gcfg->maxidx.y || (ushort)flipdir.z >= gcfg->maxidx.z) {
            /** if photon moves outside of the volume, set mediaid to 0 */
            mediaid = 0;
            idx1d = (flipdir.x < 0 || flipdir.y < 0 || flipdir.z < 0) ? OUTSIDE_VOLUME_MIN : OUTSIDE_VOLUME_MAX;
            isdet = gcfg->bc[(idx1d == OUTSIDE_VOLUME_MAX) * 3 + flipdir.w]; /** isdet now stores the boundary condition flag, this will be overwriten before the end of the loop */
            isdet = ((isdet & 0xF) == bcUnknown) ? (GPU_PARAM(gcfg, doreflect) ? bcReflect : bcAbsorb) : isdet;
            GPUDEBUG(("moving outside: [%f %f %f], idx1d [%d]->[out], bcflag %d\n", p.x, p.y, p.z, idx1d, isdet));
        } else {
            mediaid = media[idx1d];
            isdet = mediaid & DET_MASK; /** upper 16bit is the mask of the covered detector */
            mediaid &= MED_MASK;       /** lower 16bit is the medium index */
        }

        GPUDEBUG(((__constant char*)"medium [%d]->[%d]\n", mediaidold, mediaid));

        if (idx1d != idx1dold && idx1dold < gcfg->dimlen.z && mediaidold) {
            GPUDEBUG(((__constant char*)"field add to %d->%f(%d)\n", idx1dold, w0 - p.w, (int)f.w));

            // if t is within the time window, which spans cfg->maxgate*cfg->tstep wide
            if (GPU_PARAM(gcfg, save2pt) && f.y >= gcfg->twin0 && f.y < gcfg->twin1) {
                float weight = 0.f;
                int tshift = (int)(floor((f.y - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep)));

                /** calculate the quality to be accummulated */
                if (GPU_PARAM(gcfg, outputtype) == otEnergy) {
                    weight = w0 - p.w;
                } else if ((bool)(GPU_PARAM(gcfg, outputtype) == otFluence) || (bool)(GPU_PARAM(gcfg, outputtype) == otFlux)) {
                    weight = (prop.x * f.z < 0.001f) ? (w0 * f.z) : ((w0 - p.w) / (prop.x));
                } else if (GPU_PARAM(gcfg, seed) == SEED_FROM_FILE) {
                    if (GPU_PARAM(gcfg, outputtype) == otJacobian) {
                        weight = replayweight[(idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w) - 1] * pathlen;
                        tshift = (idx * gcfg->threadphoton + min(idx, gcfg->oddphoton - 1) + (int)f.w - 1);
                        tshift = (int)(floor((photontof[tshift] - gcfg->twin0) * GPU_PARAM(gcfg, Rtstep))) +
                                 ( (GPU_PARAM(gcfg, replaydet) == -1) ? ((photondetid[tshift] - 1) * GPU_PARAM(gcfg, maxgate)) : 0);
                    }
                } else if (GPU_PARAM(gcfg, outputtype) == otL) {
                    weight = w0 * pathlen;
                }

                GPUDEBUG(((__constant char*)"deposit to [%d] %e, w=%f\n", idx1dold, weight, p.w));

                if (fabs(weight) > 0.f) {
#ifndef USE_ATOMIC
                    field[idx1dold + tshift * gcfg->dimlen.z] += weight;
#else
#if !defined(MCX_SRC_PATTERN) && !defined(MCX_SRC_PATTERN3D)
                    float oldval = atomicadd(& field[idx1dold + tshift * gcfg->dimlen.z], weight);

                    if (fabs(oldval) > MAX_ACCUM) {
                        atomicadd(& field[idx1dold + tshift * gcfg->dimlen.z], ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                        atomicadd(& field[idx1dold + tshift * gcfg->dimlen.z + gcfg->dimlen.w], ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                    }

#else

                    for (int i = 0; i < GPU_PARAM(gcfg, srcnum); i++) {
                        if (fabs(ppath[GPU_PARAM(gcfg, w0offset) + i]) > 0.f) {
                            float oldval = atomicadd(& field[(idx1dold + tshift * gcfg->dimlen.z) * GPU_PARAM(gcfg, srcnum) + i], ((GPU_PARAM(gcfg, srcnum) == 1) ? weight : weight * ppath[GPU_PARAM(gcfg, w0offset) + i]));

                            if (fabs(oldval) > MAX_ACCUM) {
                                atomicadd(& field[(idx1dold + tshift * gcfg->dimlen.z)*gcfg->srcnum + i], ((oldval > 0.f) ? -MAX_ACCUM : MAX_ACCUM));
                                atomicadd(& field[(idx1dold + tshift * gcfg->dimlen.z)*gcfg->srcnum + i + gcfg->dimlen.w], ((oldval > 0.f) ? MAX_ACCUM : -MAX_ACCUM));
                            }
                        }
                    }

#endif
                    GPUDEBUG(((__constant char*)"atomic write to [%d] %e, w=%f\n", idx1dold, w0, p.w));
#endif
                }
            }

            w0 = p.w;
            pathlen = 0.f;
        }

        /** launch new photon when exceed time window or moving from non-zero voxel to zero voxel without reflection */
        if ((mediaid == 0 && ((isdet & 0xF) == bcAbsorb || (isdet & 0xF) == bcCyclic || ((isdet & 0xF) == bcReflect && n1 == gproperty[0].w))) ||  f.y > gcfg->twin1) {
            if (isdet == bcCyclic) {
                if (flipdir.w == 0) {
                    p.x = mcx_nextafterf(convert_float_rte(((idx1d == OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.x : 0)), (v.x > 0.f) - (v.x < 0.f));
                    flipdir.x = convert_short_rtn(p.x);
                }

                if (flipdir.w == 1) {
                    p.y = mcx_nextafterf(convert_float_rte(((idx1d == OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.y : 0)), (v.y > 0.f) - (v.y < 0.f));
                    flipdir.y = convert_short_rtn(p.y);
                }

                if (flipdir.w == 2) {
                    p.z = mcx_nextafterf(convert_float_rte(((idx1d == OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.z : 0)), (v.z > 0.f) - (v.z < 0.f));
                    flipdir.z = convert_short_rtn(p.z);
                }

                GPUDEBUG(((__constant char*)"cyclic: p=[%f %f %f] -> voxel =[%d %d %d] %d %d\n", p.x, p.y, p.z, flipdir.x, flipdir.y, flipdir.z, isdet, bcCyclic));

                if ((ushort)flipdir.x < gcfg->maxidx.x && (ushort)flipdir.y < gcfg->maxidx.y && (ushort)flipdir.z < gcfg->maxidx.z) {
                    idx1d = (flipdir.z * gcfg->dimlen.y + flipdir.y * gcfg->dimlen.x + flipdir.x);
                    mediaid = media[idx1d];
                    isdet = mediaid & DET_MASK; /** upper 16bit is the mask of the covered detector */
                    mediaid &= MED_MASK;       /** lower 16bit is the medium index */
                    GPUDEBUG(("Cyclic boundary condition, moving photon in dir %d at %d flag, new pos=[%f %f %f]\n", flipdir.w, isdet, p.x, p.y, p.z));
                    continue;
                }
            }

            GPUDEBUG(((__constant char*)"direct relaunch at idx=[%d] mediaid=[%d] bc=[%d] ref=[%d]\n", idx1d, mediaid, isdet, GPU_PARAM(gcfg, doreflect)));

            if (launchnewphoton(&p, &v, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove,
                                (((idx1d == OUTSIDE_VOLUME_MAX && gcfg->bc[9 + flipdir.w]) || (idx1d == OUTSIDE_VOLUME_MIN && gcfg->bc[6 + flipdir.w])) ? OUTSIDE_VOLUME_MIN : (mediaidold & DET_MASK)),
                                ppath, n_det, detectedphoton, t, (__global RandType*)n_seed, gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton, gprogress,
                                (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen)) +
                                                    get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)), gseeddata, gjumpdebug, gdebugdata, sharedmem)) {
                break;
            }

            isdet = mediaid & DET_MASK;
            mediaid &= MED_MASK;
            continue;
        }

#ifdef MCX_DO_REFLECTION

        if (gcfg->mediaformat < 100) {
            updateproperty(&prop, mediaid, gproperty, gcfg); ///< optical property across the interface
        }

        //if hit the boundary, exceed the max time window or exit the domain, rebound or launch a new one
        if (((mediaid && GPU_PARAM(gcfg, doreflect)) // if at an internal boundary, check cfg.isreflect flag
                || (mediaid == 0 &&  // or if out of bbx or enters 0-voxel
                    (((isdet & 0xF) == bcUnknown && GPU_PARAM(gcfg, doreflect)) // if cfg.bc is "_", check cfg.isreflect
                     || ((isdet & 0xF) == bcReflect || (isdet & 0xF) == bcMirror))))  // or if cfg.bc is 'r' or 'm'
                && (((isdet & 0xF) == bcMirror) || n1 != ((GPU_PARAM(gcfg, mediaformat) < 100) ? (prop.w) : (gproperty[(mediaid > 0 && (bool)(GPU_PARAM(gcfg, mediaformat) >= 100)) ? 1 : mediaid].w)))) {
            float Rtotal = 1.f;
            float cphi, sphi, stheta, ctheta;

            updateproperty(&prop, mediaid, gproperty, gcfg); ///< optical property across the interface

            float tmp0 = n1 * n1;
            float tmp1 = prop.w * prop.w;
            cphi = fabs( (flipdir.w == 0) ? v.x : (flipdir.w == 1 ? v.y : v.z)); // cos(si)
            sphi = 1.f - cphi * cphi;      // sin(si)^2

            f.z = 1.f - tmp0 / tmp1 * sphi; //1-[n1/n2*sin(si)]^2
            GPUDEBUG(((__constant char*)"ref total ref=%f\n", f.z));

            if (f.z > 0.f && (isdet & 0xF) != bcMirror) { //< if no total internal reflection, or not mirror bc
                ctheta = tmp0 * cphi * cphi + tmp1 * f.z;
                stheta = 2.f * n1 * prop.w * cphi * sqrt(f.z);
                Rtotal = (ctheta - stheta) / (ctheta + stheta);
                ctheta = tmp1 * cphi * cphi + tmp0 * f.z;
                Rtotal = (Rtotal + (ctheta - stheta) / (ctheta + stheta)) * 0.5f;
                GPUDEBUG(((__constant char*)"Rtotal=%f\n", Rtotal));
            }

            if (Rtotal < 1.f // if total internal reflection does not happen
                    && (!(mediaid == 0 && ((isdet & 0xF) == bcMirror))) // if out of bbx and cfg.bc is not 'm'
                    && rand_next_reflect(t) > Rtotal) { // and if photon chooses the transmission path, then do transmission

                transmit(&v, n1, prop.w, flipdir.w);

                if (mediaid == 0) { // transmission to external boundary
                    GPUDEBUG(((__constant char*)"transmit to air, relaunch, idx=[%d] mediaid=[%d] bc=[%d] ref=[%d]\n", idx1d, mediaid, isdet, GPU_PARAM(gcfg, doreflect)));

                    if (launchnewphoton(&p, &v, &f, &flipdir, &prop, &idx1d, field, &mediaid, &w0, &Lmove,
                                        (((idx1d == OUTSIDE_VOLUME_MAX && gcfg->bc[9 + flipdir.w]) || (idx1d == OUTSIDE_VOLUME_MIN && gcfg->bc[6 + flipdir.w])) ? OUTSIDE_VOLUME_MIN : (mediaidold & DET_MASK)),
                                        ppath, n_det, detectedphoton, t, (__global RandType*)n_seed, gproperty, media, srcpattern, gdetpos, gcfg, idx, blockphoton, gprogress,
                                        (__local RandType*)((__local char*)sharedmem + sizeof(float) * (GPU_PARAM(gcfg, nphaselen) + GPU_PARAM(gcfg, nanglelen))
                                                            + get_local_id(0)*GPU_PARAM(gcfg, issaveseed)*RAND_BUF_LEN * sizeof(RandType)), gseeddata, gjumpdebug, gdebugdata, sharedmem)) {
                        break;
                    }

                    isdet = mediaid & DET_MASK;
                    mediaid &= MED_MASK;
                    continue;
                }

                GPUDEBUG(((__constant char*)"do transmission\n"));
            } else { //do reflection
                GPUDEBUG(((__constant char*)"do reflection\n"));
                GPUDEBUG(((__constant char*)"ref faceid=%d p=[%f %f %f] v_old=[%f %f %f]\n", flipdir.w, p.x, p.y, p.z, v.x, v.y, v.z));
                (flipdir.w == 0) ? (v.x = -v.x) : ((flipdir.w == 1) ? (v.y = -v.y) : (v.z = -v.z)) ;
                (flipdir.w == 0) ?
                (p.x = mcx_nextafterf(convert_float_rte(p.x), (v.x > 0.f) - 0.5f)) :
                ((flipdir.w == 1) ?
                 (p.y = mcx_nextafterf(convert_float_rte(p.y), (v.y > 0.f) - 0.5f)) :
                 (p.z = mcx_nextafterf(convert_float_rte(p.z), (v.z > 0.f) - 0.5f)) );
                (flipdir.w == 0) ? (flipdir.x = convert_short_rte(p.x)) : ((flipdir.w == 1) ? (flipdir.y = convert_short_rte(p.y)) : (flipdir.z = convert_short_rte(p.z))) ;
                GPUDEBUG(((__constant char*)"ref p_new=[%f %f %f] v_new=[%f %f %f]\n", p.x, p.y, p.z, v.x, v.y, v.z));
                idx1d = idx1dold;
                mediaid = (media[idx1d] & MED_MASK);
                updateproperty(&prop, mediaid, gproperty, gcfg); ///< optical property across the interface
                n1 = prop.w;
            }
        }

        if (mediaid == 0 || idx1d == OUTSIDE_VOLUME_MIN || idx1d == OUTSIDE_VOLUME_MAX) {
            printf("ERROR: should never happen! mediaid=%d idx1d=%X gcfg->doreflect=%d n1=%f n2=%f isdet=%d flipdir[3]=%d p=(%f %f %f)[%d %d %d]\n", mediaid, idx1d, GPU_PARAM(gcfg, doreflect), n1, prop.w, isdet, flipdir.w, p.x, p.y, p.z, flipdir.x, flipdir.y, flipdir.z);
            return;
        }

#endif
    }

    genergy[idx << 1]    = ppath[GPU_PARAM(gcfg, partialdata)];
    genergy[(idx << 1) + 1] = ppath[GPU_PARAM(gcfg, partialdata) + 1];

    if (GPU_PARAM(gcfg, issaveref) > 1) {
        *detectedphoton = GPU_PARAM(gcfg, maxdetphoton);
    }
}

