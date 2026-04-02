/***************************************************************************//**
**  \file    mcx_cu_host.cu
**  \brief   CUDA host code for MCX dual-backend simulation
**
**  This file mirrors mcx_host.cpp (OpenCL) but uses the CUDA runtime API.
**  It includes the shared kernel mcx_core.cl directly.
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025, GPL v3
*******************************************************************************/

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "mcx_cu_host.h"
#include "mcx_tictoc.h"
#include "mcx_const.h"

#include <cuda.h>
#include "cuda_fp16.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

#define CUDA_ASSERT(a) mcx_cu_assess((a), __FILE__, __LINE__)

#ifndef NANGLES
    #define NANGLES 181
#endif

/*=== Constant memory — must be before #include mcx_core.cl ===*/

__constant__ MCXParam gcfg_const[1];

/*=== Include the shared kernel ===*/

#define MCX_ADJOINT_MODE  /**< always compile adjoint kernels for CUDA (static compilation, no JIT overhead) */
#include "mcx_core.cl"

/*=== CUDA error checking ===*/

void mcx_cu_assess(cudaError_t cuerr, const char* file, const int linenum) {
    if (cuerr != cudaSuccess) {
        mcx_error(-(int)cuerr, (char*)cudaGetErrorString(cuerr), file, linenum);
    }
}

/*=== GPU info helpers ===*/

static int mcx_nv_corecount(int v1, int v2) {
    int v = v1 * 10 + v2;

    if (v < 20) {
        return 8;
    } else if (v < 21) {
        return 32;
    } else if (v < 30) {
        return 48;
    } else if (v < 50) {
        return 192;
    } else if (v < 60 || v == 61 || v >= 89) {
        return 128;
    } else {
        return 64;
    }
}

static int mcx_smxblock(int v1, int v2) {
    int v = v1 * 10 + v2;

    if (v < 30) {
        return 8;
    } else if (v < 50) {
        return 16;
    } else {
        return 32;
    }
}

/*=== GPU enumeration ===*/

int mcx_list_cuda_gpu(Config* cfg, GPUInfo** info) {
    int dev, deviceCount, activedev = 0;
    cudaError_t cuerr = cudaGetDeviceCount(&deviceCount);

    if (cuerr != cudaSuccess) {
        if ((int)cuerr == 30) {
            mcx_error(-(int)cuerr, "No CUDA-capable GPU found", __FILE__, __LINE__);
        }

        CUDA_ASSERT(cuerr);
    }

    if (deviceCount == 0) {
        MCX_FPRINTF(cfg->flog, S_RED "ERROR: No CUDA-capable GPU device found\n" S_RESET);
        return 0;
    }

    *info = (GPUInfo*)calloc(deviceCount, sizeof(GPUInfo));

    for (dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp dp;
        CUDA_ASSERT(cudaGetDeviceProperties(&dp, dev));

        if (cfg->isgpuinfo == 3) {
            activedev++;
        } else if (cfg->deviceid[dev] == '1') {
            cfg->deviceid[dev] = '\0';
            cfg->deviceid[activedev] = dev + 1;
            activedev++;
        }

        strncpy((*info)[dev].name, dp.name, MAX_SESSION_LENGTH);
        (*info)[dev].id = dev + 1;
        (*info)[dev].devcount = deviceCount;
        (*info)[dev].major = dp.major;
        (*info)[dev].minor = dp.minor;
        (*info)[dev].globalmem = dp.totalGlobalMem;
        (*info)[dev].constmem = dp.totalConstMem;
        (*info)[dev].sharedmem = dp.sharedMemPerBlock;
        (*info)[dev].regcount = dp.regsPerBlock;
        (*info)[dev].clock = dp.clockRate;
        (*info)[dev].sm = dp.multiProcessorCount;
        (*info)[dev].core = dp.multiProcessorCount * mcx_nv_corecount(dp.major, dp.minor);
        (*info)[dev].maxmpthread = dp.maxThreadsPerMultiProcessor;
        (*info)[dev].maxgate = cfg->maxgate;
        (*info)[dev].autoblock = MAX((*info)[dev].maxmpthread / mcx_smxblock(dp.major, dp.minor), 64);
        (*info)[dev].autothread = (*info)[dev].autoblock * mcx_smxblock(dp.major, dp.minor) * (*info)[dev].sm;

        if (cfg->isgpuinfo) {
            MCX_FPRINTF(stdout, S_BLUE "=============================   GPU Information  ================================\n" S_RESET);
            MCX_FPRINTF(stdout, "Device %d of %d:\t\t%s\n", (*info)[dev].id, (*info)[dev].devcount, (*info)[dev].name);
            MCX_FPRINTF(stdout, "Compute Capability:\t%u.%u\n", (*info)[dev].major, (*info)[dev].minor);
            MCX_FPRINTF(stdout, "Global Memory:\t\t%.0f B\nShared Memory:\t\t%.0f B\n",
                        (double)(*info)[dev].globalmem, (double)(*info)[dev].sharedmem);
            MCX_FPRINTF(stdout, "Number of SMs:\t\t%u\nNumber of Cores:\t%u\n", (*info)[dev].sm, (*info)[dev].core);
            MCX_FPRINTF(stdout, "Clock Speed:\t\t%.2f GHz\n", (*info)[dev].clock * 1e-6f);
            MCX_FPRINTF(stdout, "Auto-thread:\t\t%zu\nAuto-block:\t\t%zu\n", (*info)[dev].autothread, (*info)[dev].autoblock);
        }
    }

    if (cfg->isgpuinfo == 2 && cfg->parentid == mpStandalone) {
        exit(0);
    }

    if (activedev < MAX_DEVICE) {
        cfg->deviceid[activedev] = '\0';
    }

    return activedev;
}


static void mcx_launch_kernel(
    int ispencil, int isref, int islabel, int issvmc, int ispolarized,
    dim3 mcgrid, dim3 mcblock, uint sharedbuf,
    uint* gmedia, float* gfield, float* genergy, uint* gPseed,
    float* gPdet, float4* gproperty, float* gsrcpattern,
    float4* gdetpos, volatile uint* gprogress, uint* gdetected,
    float* greplayw, float* greplaytof, int* greplaydetid,
    RandType* gseeddata, uint* gjumpdebug, float* gdebugdata,
    float* ginvcdf, float* gangleinvcdf,
    float4* gsmatrix) {
#define MCX_LAUNCH(P,R,L,S,Pol) \
    mcx_main_loop<P,R,L,S,Pol><<<mcgrid,mcblock,sharedbuf>>>( \
            gmedia, gfield, genergy, gPseed, gPdet, gproperty, gsrcpattern, \
            gdetpos, gprogress, gdetected, greplayw, greplaytof, greplaydetid, \
            gseeddata, gjumpdebug, gdebugdata, ginvcdf, gangleinvcdf, \
            (RandType*)NULL, gsmatrix)

    int key = ispencil * 10000 + (isref > 0) * 1000 + islabel * 100 + issvmc * 10 + ispolarized;

    switch (key) {
        case     0:
            MCX_LAUNCH(0, 0, 0, 0, 0);
            break;

        case    10:
            MCX_LAUNCH(0, 0, 0, 1, 0);
            break;

        case   100:
            MCX_LAUNCH(0, 0, 1, 0, 0);
            break;

        case   101:
            MCX_LAUNCH(0, 0, 1, 0, 1);
            break;

        case  1000:
            MCX_LAUNCH(0, 1, 0, 0, 0);
            break;

        case  1010:
            MCX_LAUNCH(0, 1, 0, 1, 0);
            break;

        case  1100:
            MCX_LAUNCH(0, 1, 1, 0, 0);
            break;

        case  1101:
            MCX_LAUNCH(0, 1, 1, 0, 1);
            break;

        case 10000:
            MCX_LAUNCH(1, 0, 0, 0, 0);
            break;

        case 10010:
            MCX_LAUNCH(1, 0, 0, 1, 0);
            break;

        case 10100:
            MCX_LAUNCH(1, 0, 1, 0, 0);
            break;

        case 10101:
            MCX_LAUNCH(1, 0, 1, 0, 1);
            break;

        case 11000:
            MCX_LAUNCH(1, 1, 0, 0, 0);
            break;

        case 11010:
            MCX_LAUNCH(1, 1, 0, 1, 0);
            break;

        case 11100:
            MCX_LAUNCH(1, 1, 1, 0, 0);
            break;

        case 11101:
            MCX_LAUNCH(1, 1, 1, 0, 1);
            break;

        default:
            MCX_LAUNCH(0, 1, 1, 0, 0);
            break;  /* safe fallback */
    }

#undef MCX_LAUNCH
}


/*==========================================================================
 * Main simulation driver
 *==========================================================================*/

void mcx_run_cuda(Config* cfg, float* fluence, float* totalenergy) {

    uint i;
    float minstep = MIN(MIN(cfg->steps.x, cfg->steps.y), cfg->steps.z);
    float fullload = 0.f;
    uint tic;
    uint debuglen = MCX_DEBUG_REC_LEN + (cfg->istrajstokes << 2);
    size_t fieldlen;
    int totalgates;
    int isrfforward = (cfg->omega > 0.f && cfg->seed != SEED_FROM_FILE);

    size_t dimxyz = cfg->dim.x * cfg->dim.y * cfg->dim.z *
                    ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D)
                     ? cfg->srcnum : (cfg->srcid < 0) ? (cfg->extrasrclen + 1) : 1);

    uint* media = (uint*)(cfg->vol);
    GPUInfo* gpu = NULL;
    int activedev = 0;

    unsigned int partialdata = (cfg->medianum - 1) *
                               (SAVE_NSCAT(cfg->savedetflag) + SAVE_PPATH(cfg->savedetflag) + SAVE_MOM(cfg->savedetflag));
    unsigned int w0offset = partialdata + 4;
    unsigned int hostdetreclen = partialdata + SAVE_DETID(cfg->savedetflag)
                                 + 3 * (SAVE_PEXIT(cfg->savedetflag) + SAVE_VEXIT(cfg->savedetflag))
                                 + SAVE_W0(cfg->savedetflag) + 4 * SAVE_IQUV(cfg->savedetflag);
    unsigned int is2d = (cfg->dim.x == 1 ? 1 : (cfg->dim.y == 1 ? 2 : (cfg->dim.z == 1 ? 3 : 0)));

    /* Build MCXParam */
    MCXParam param;
    memset(&param, 0, sizeof(MCXParam));

    param.src.pos.x = cfg->srcpos.x;
    param.src.pos.y = cfg->srcpos.y;
    param.src.pos.z = cfg->srcpos.z;
    param.src.pos.w = cfg->srcpos.w;
    param.src.dir.x = cfg->srcdir.x;
    param.src.dir.y = cfg->srcdir.y;
    param.src.dir.z = cfg->srcdir.z;
    param.src.dir.w = cfg->srcdir.w;
    param.src.param1.x = cfg->srcparam1.x;
    param.src.param1.y = cfg->srcparam1.y;
    param.src.param1.z = cfg->srcparam1.z;
    param.src.param1.w = cfg->srcparam1.w;
    param.src.param2.x = cfg->srcparam2.x;
    param.src.param2.y = cfg->srcparam2.y;
    param.src.param2.z = cfg->srcparam2.z;
    param.src.param2.w = cfg->srcparam2.w;
    param.extrasrclen = cfg->extrasrclen;
    param.srcid       = cfg->srcid;
    param.maxidx.x = (float)cfg->dim.x;
    param.maxidx.y = (float)cfg->dim.y;
    param.maxidx.z = (float)cfg->dim.z;
    param.maxidx.w = 0;
    param.minstep     = minstep;
    param.tmax        = cfg->tend;
    param.oneoverc0   = R_C0 * cfg->unitinmm;
    param.isrowmajor  = (uint)cfg->isrowmajor;
    param.save2pt     = (uint)cfg->issave2pt;
    param.doreflect   = (uint)cfg->isreflect;
    param.dorefint    = (uint)cfg->isrefint;
    param.savedet     = (uint)cfg->issavedet;
    param.Rtstep      = 1.f / cfg->tstep;
    param.minenergy   = cfg->minenergy;
    param.skipradius2  = cfg->sradius * cfg->sradius;
    param.minaccumtime = minstep * R_C0 * cfg->unitinmm;
    param.maxdetphoton = cfg->maxdetphoton;
    param.maxmedia    = cfg->medianum - 1;
    param.detnum      = cfg->detnum;
    param.voidtime    = (uint)cfg->voidtime;
    param.srctype     = (uint)cfg->srctype;
    param.maxvoidstep = (uint)cfg->maxvoidstep;
    param.issaveexit  = cfg->issaveexit > 0;
    param.issaveseed  = cfg->issaveseed > 0;
    param.issaveref   = (uint)cfg->issaveref;
    param.isspecular  = cfg->isspecular > 0;
    param.maxgate     = cfg->maxgate;
    param.seed        = cfg->seed;
    param.outputtype  = (uint)cfg->outputtype;
    param.debuglevel  = (uint)cfg->debuglevel;
    param.savedetflag = cfg->savedetflag;
    param.reclen      = hostdetreclen;
    param.partialdata = partialdata;
    param.w0offset    = w0offset;
    param.mediaformat = (uint)cfg->mediabyte;
    param.maxjumpdebug = (uint)cfg->maxjumpdebug;
    param.gscatter    = cfg->gscatter;

    /* 2D mode: only valid when exactly 2 dimensions have length > 1 */
    if (is2d) {
        is2d = is2d * ((cfg->dim.x > 1) + (cfg->dim.y > 1) + (cfg->dim.z > 1) == 2);
    }

    param.is2d        = is2d;

    if (is2d) {
        float* vec = &(cfg->srcdir.x);

        if (fabsf(vec[is2d - 1]) > EPS) {
            mcx_error(-1, "2D simulation requires source direction to be zero in the singleton dimension", __FILE__, __LINE__);
        }
    }

    param.replaydet   = cfg->replaydet;
    param.srcnum      = cfg->srcnum;
    param.nphase      = cfg->nphase;
    param.nphaselen   = cfg->nphase + (cfg->nphase & 0x1);
    param.nangle      = cfg->nangle;
    param.nanglelen   = cfg->nangle + (cfg->nangle & 0x1);
    param.maxpolmedia = cfg->polmedianum;
    param.istrajstokes = (uint)cfg->istrajstokes;
    param.s0.x = cfg->srciquv.x;
    param.s0.y = cfg->srciquv.y;
    param.s0.z = cfg->srciquv.z;
    param.s0.w = cfg->srciquv.w;
    param.omega       = cfg->omega;
    memcpy(param.bc, cfg->bc, 12);

    /*=== List GPUs ===*/
    activedev = mcx_list_cuda_gpu(cfg, &gpu);

    if (activedev == 0) {
        mcx_error(-99, "No GPU device found", __FILE__, __LINE__);
    }

    totalgates = (int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);

    /*=== Compute per-GPU autopilot parameters for all active devices ===*/
    for (i = 0; (int)cfg->deviceid[i] > 0; i++) {
        int gid = (int)cfg->deviceid[i] - 1;

        if (!cfg->autopilot) {
            gpu[gid].autothread = cfg->nthread;
            gpu[gid].autoblock  = cfg->nblocksize;
            gpu[gid].maxgate    = (cfg->maxgate > 0) ? MIN((size_t)cfg->maxgate, (size_t)totalgates) : (size_t)totalgates;
        }

        if (gpu[gid].autothread % gpu[gid].autoblock) {
            gpu[gid].autothread = (gpu[gid].autothread / gpu[gid].autoblock) * gpu[gid].autoblock;
        }

        if (gpu[gid].maxgate == 0 && dimxyz > 0) {
            size_t needmem = dimxyz + gpu[gid].autothread * sizeof(float4) * 4
                             + sizeof(float) * cfg->maxdetphoton * hostdetreclen + 10 * 1024 * 1024;
            gpu[gid].maxgate = (gpu[gid].globalmem - needmem) / dimxyz;
            gpu[gid].maxgate = MIN((size_t)totalgates, gpu[gid].maxgate);
        }
    }

    /*=== Compute workload distribution across all active GPUs ===*/
    fullload = 0.f;

    for (i = 0; (int)cfg->deviceid[i] > 0; i++) {
        fullload += cfg->workload[i];
    }

    if (fullload < EPS) {
        for (i = 0; (int)cfg->deviceid[i] > 0; i++) {
            cfg->workload[i] = gpu[(int)cfg->deviceid[i] - 1].core;
        }

        fullload = 0.f;

        for (i = 0; (int)cfg->deviceid[i] > 0; i++) {
            fullload += cfg->workload[i];
        }
    }

    /*=== Compute dimlen and fieldlen for the shared export buffers ===*/
    /* Use totalgates (not per-GPU maxgate) so the export buffer covers the full time range */
    int gpuid0 = (int)cfg->deviceid[0] - 1;
    cfg->maxgate = totalgates;
    uint4 dimlen;
    dimlen.x = cfg->dim.x;
    dimlen.y = cfg->dim.x * cfg->dim.y;
    dimlen.z = cfg->dim.x * cfg->dim.y * cfg->dim.z;
    dimlen.w = (uint)totalgates * ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D)
                                   ? cfg->srcnum : (cfg->srcid < 0) ? (cfg->extrasrclen + 1) : 1);

    if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
        dimlen.w *= cfg->detnum;
    }

    fieldlen = (size_t)dimlen.z * dimlen.w;  /* total export buffer size: pure_dimxyz * totalgates * srcmul [* detnum] */
    size_t fieldpergate = fieldlen / totalgates; /* per-time-gate export slice size */
    dimlen.w = (uint)fieldlen;
    param.dimlen = dimlen;
    param.maxgate = (uint)gpu[gpuid0].maxgate; /* kernel sees per-GPU maxgate; updated per-thread below */

    /*=== Shared output buffer initialization (before parallel section) ===*/
    if (cfg->exportfield == NULL) {
        if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
            cfg->exportfield = (float*)calloc(sizeof(float) * dimxyz, cfg->maxgate * 2 * (1 + (cfg->outputtype == otRF || cfg->outputtype == otRFmus)) * cfg->detnum);
        } else {
            cfg->exportfield = (float*)calloc(sizeof(float) * fieldlen, 2 * (isrfforward ? 2 : 1));
        }
    }

    if (cfg->exportdetected == NULL) {
        cfg->exportdetected = (float*)malloc(hostdetreclen * cfg->maxdetphoton * sizeof(float));
    }

    if (cfg->issaveseed && cfg->seeddata == NULL) {
        cfg->seeddata = malloc(cfg->maxdetphoton * sizeof(RandType) * RAND_BUF_LEN);
    }

    cfg->detectedcount = 0;
    cfg->his.detected = 0;
    cfg->his.respin = cfg->respin;
    cfg->his.colcount = hostdetreclen;
    cfg->energytot = 0.f;
    cfg->energyesc = 0.f;
    cfg->runtime = 0;

    tic = StartTimer();

    /*=== Print launch header (before parallel section) ===*/
    mcx_printheader(cfg);
    MCX_FPRINTF(cfg->flog, "- code name: [Infinity-CUDA] compiled by nvcc [%d.%d]\n",
                __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    MCX_FPRINTF(cfg->flog, "- compiled with: [RNG] %s [Seed Length] %d\n", MCX_RNG_NAME, RAND_SEED_LEN);

    /*=== Determine kernel specialization flags ===*/
    int ispencil = (cfg->srctype == MCX_SRC_PENCIL && cfg->nangle == 0);
    int isref = cfg->isreflect;
    int issvmc = (cfg->mediabyte == MEDIA_2LABEL_SPLIT);
    int ispolarized = (cfg->mediabyte <= 4) && (cfg->polmedianum > 0);

    for (i = 0; i < 6; i++)
        if (cfg->bc[i] == bcReflect || cfg->bc[i] == bcMirror) {
            isref = 1;
        }

    /*=== Multi-GPU parallel simulation ===*/
#ifdef _OPENMP
    omp_set_num_threads(activedev);
    #pragma omp parallel
    {
        int threadid = omp_get_thread_num();
#else
    {
        int threadid = 0;
#endif
        int gpuid = (int)cfg->deviceid[threadid] - 1;
        size_t gpuphoton;

        /* Compute per-GPU photon count based on workload fraction */
        {
            size_t prev_sum = 0;
            int d;

            for (d = 0; d < threadid; d++) {
                prev_sum += (size_t)((double)cfg->nphoton * cfg->workload[d] / fullload);
            }

            if (threadid == activedev - 1) {
                gpuphoton = cfg->nphoton - prev_sum;
            } else {
                gpuphoton = (size_t)((double)cfg->nphoton * cfg->workload[threadid] / fullload);
            }
        }

        if (gpuphoton > 0) {
            uint ltic0, ltic1, ltoc = 0;
            int liter;
            uint lj;
            float* field = NULL;
            float* Pdet  = NULL;
            float* energy = NULL;
            uint*  Pseed  = NULL;
            RandType* seeddata = NULL;
            uint   detected = 0;
            dim3   mcgrid, mcblock;
            uint   sharedbuf;

            uint*      gmedia       = NULL;
            float*     gfield       = NULL;
            float*     genergy      = NULL;
            uint*      gPseed       = NULL;
            float*     gPdet        = NULL;
            uint*      gdetected    = NULL;
            float4*    gproperty    = NULL;
            float4*    gdetpos      = NULL;
            float*     gsrcpattern  = NULL;
            float*     ginvcdf      = NULL;
            float*     gangleinvcdf = NULL;
            float4*    gsmatrix     = NULL;
            RandType*  gseeddata    = NULL;
            float*     gdebugdata   = NULL;
            uint*      gjumpdebug   = NULL;
            float*     greplayw     = NULL;
            float*     greplaytof   = NULL;
            int*       greplaydetid = NULL;
            RandType*  greplayseed  = NULL;
            volatile uint* gprogress = NULL;

            CUDA_ASSERT(cudaSetDevice(gpuid));

            /* Per-GPU kernel parameters */
            MCXParam lparam = param;
            lparam.threadphoton = (int)(gpuphoton / ((size_t)gpu[gpuid].autothread * cfg->respin));
            lparam.oddphoton    = (int)(gpuphoton / cfg->respin - (size_t)lparam.threadphoton * gpu[gpuid].autothread);
            lparam.maxgate      = (uint)gpu[gpuid].maxgate;

            mcblock  = dim3(gpu[gpuid].autoblock, 1, 1);
            mcgrid   = dim3(gpu[gpuid].autothread / gpu[gpuid].autoblock, 1, 1);
            sharedbuf = (param.nphaselen + param.nanglelen) * sizeof(float)
                        + gpu[gpuid].autoblock * (cfg->issaveseed * (RAND_BUF_LEN * sizeof(RandType))
                                                  + sizeof(float) * (param.w0offset + cfg->srcnum
                                                          + 2 * (cfg->outputtype == otRF || cfg->outputtype == otRFmus || MCX_IS_ADJOINT_TYPE(cfg->outputtype))));

            /* Per-GPU field buffer length (covers gpu[gpuid].maxgate time gates per pass) */
            size_t gfieldlen = fieldpergate * (size_t)gpu[gpuid].maxgate;

            MCX_FPRINTF(cfg->flog, "GPU=%d (%s) threadph=%d extra=%d np=%.0f nthread=%d sharedbuf=%d\n",
                        gpuid + 1, gpu[gpuid].name, lparam.threadphoton, lparam.oddphoton,
                        (double)gpuphoton, (int)gpu[gpuid].autothread, sharedbuf);

            /* Allocate host buffers */
            field  = (float*)calloc(sizeof(float) * gfieldlen, 2);
            Pdet   = (float*)calloc(cfg->maxdetphoton, sizeof(float) * hostdetreclen);
            energy = (float*)calloc(gpu[gpuid].autothread << 1, sizeof(float));

            if (cfg->seed != SEED_FROM_FILE) {
                Pseed = (uint*)malloc(sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN);
            } else {
                Pseed = (uint*)malloc(sizeof(RandType) * cfg->nphoton * RAND_BUF_LEN);
            }

            if (cfg->seed != SEED_FROM_FILE) {
                #pragma omp critical
                {
                    srand(cfg->seed + threadid);

                    for (lj = 0; lj < gpu[gpuid].autothread* RAND_SEED_LEN; lj++) {
                        Pseed[lj] = rand();
                    }
                }
            }

            if ((cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) && cfg->exportdebugdata == NULL) {
                #pragma omp critical
                {
                    if (cfg->exportdebugdata == NULL) {
                        cfg->exportdebugdata = (float*)calloc(sizeof(float), debuglen * cfg->maxjumpdebug);
                    }
                }
            }

            /* Progress bar pinned memory (per-GPU) */
            volatile uint* progress;
            CUDA_ASSERT(cudaHostAlloc((void**)&progress, sizeof(uint), cudaHostAllocMapped));
            CUDA_ASSERT(cudaHostGetDevicePointer((uint**)&gprogress, (uint*)progress, 0));
            *progress = 0;

            /* Media volume */
            if (cfg->mediabyte != MEDIA_2LABEL_SPLIT && cfg->mediabyte != MEDIA_ASGN_F2H) {
                CUDA_ASSERT(cudaMalloc((void**)&gmedia, sizeof(uint) * cfg->dim.x * cfg->dim.y * cfg->dim.z));
                CUDA_ASSERT(cudaMemcpy(gmedia, media, sizeof(uint) * cfg->dim.x * cfg->dim.y * cfg->dim.z, cudaMemcpyHostToDevice));
            } else {
                CUDA_ASSERT(cudaMalloc((void**)&gmedia, sizeof(uint) * 2 * cfg->dim.x * cfg->dim.y * cfg->dim.z));
                CUDA_ASSERT(cudaMemcpy(gmedia, media, sizeof(uint) * 2 * cfg->dim.x * cfg->dim.y * cfg->dim.z, cudaMemcpyHostToDevice));
            }

            /* Optical properties + extra source data + detector positions */
            {
                size_t propsize  = cfg->medianum * sizeof(Medium);
                size_t srcsize   = cfg->srcdata ? cfg->extrasrclen * 4 * sizeof(float4) : 0;
                size_t detsize   = (cfg->detpos && cfg->detnum) ? cfg->detnum * sizeof(float4) : 0;
                size_t totalsize = propsize + srcsize + detsize;
                char* propbuf    = (char*)malloc(totalsize);
                memcpy(propbuf, cfg->prop, propsize);

                if (cfg->srcdata) {
                    memcpy(propbuf + propsize, cfg->srcdata, srcsize);
                }

                if (cfg->detpos && cfg->detnum) {
                    memcpy(propbuf + propsize + srcsize, cfg->detpos, detsize);
                }

                CUDA_ASSERT(cudaMalloc((void**)&gproperty, totalsize));
                CUDA_ASSERT(cudaMemcpy(gproperty, propbuf, totalsize, cudaMemcpyHostToDevice));
                free(propbuf);
            }

            /* Detector positions */
            if (cfg->detpos && cfg->detnum) {
                CUDA_ASSERT(cudaMalloc((void**)&gdetpos, cfg->detnum * sizeof(float4)));
                CUDA_ASSERT(cudaMemcpy(gdetpos, cfg->detpos, cfg->detnum * sizeof(float4), cudaMemcpyHostToDevice));
            }

            /* Output field buffer (per-GPU per-pass size: gfieldlen; RF needs 4x for Im) */
            CUDA_ASSERT(cudaMalloc((void**)&gfield, sizeof(float) * gfieldlen * (isrfforward ? 4 : 2)));
            CUDA_ASSERT(cudaMemset(gfield, 0, sizeof(float) * gfieldlen * (isrfforward ? 4 : 2)));

            /* RNG seeds */
            if (cfg->seed != SEED_FROM_FILE) {
                CUDA_ASSERT(cudaMalloc((void**)&gPseed, sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN));
                CUDA_ASSERT(cudaMemcpy(gPseed, Pseed, sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN, cudaMemcpyHostToDevice));
            } else {
                CUDA_ASSERT(cudaMalloc((void**)&gPseed, sizeof(RandType) * cfg->nphoton * RAND_BUF_LEN));
                CUDA_ASSERT(cudaMemcpy(gPseed, cfg->replay.seed, sizeof(RandType) * cfg->nphoton * RAND_BUF_LEN, cudaMemcpyHostToDevice));
            }

            /* Detected photon buffer */
            if (cfg->issavedet) {
                CUDA_ASSERT(cudaMalloc((void**)&gPdet, sizeof(float) * cfg->maxdetphoton * hostdetreclen));
                CUDA_ASSERT(cudaMemset(gPdet, 0, sizeof(float) * cfg->maxdetphoton * hostdetreclen));
            }

            /* Energy */
            CUDA_ASSERT(cudaMalloc((void**)&genergy, sizeof(float) * (gpu[gpuid].autothread << 1)));
            CUDA_ASSERT(cudaMemcpy(genergy, energy, sizeof(float) * (gpu[gpuid].autothread << 1), cudaMemcpyHostToDevice));

            /* Detected count */
            CUDA_ASSERT(cudaMalloc((void**)&gdetected, sizeof(uint)));
            CUDA_ASSERT(cudaMemset(gdetected, 0, sizeof(uint)));

            /* Source pattern */
            if (cfg->srctype == MCX_SRC_PATTERN && cfg->srcpattern) {
                size_t patsize = sizeof(float) * (int)(cfg->srcparam1.w * cfg->srcparam2.w) * cfg->srcnum;
                CUDA_ASSERT(cudaMalloc((void**)&gsrcpattern, patsize));
                CUDA_ASSERT(cudaMemcpy(gsrcpattern, cfg->srcpattern, patsize, cudaMemcpyHostToDevice));
            } else if (cfg->srctype == MCX_SRC_PATTERN3D && cfg->srcpattern) {
                size_t patsize = sizeof(float) * (int)(cfg->srcparam1.x * cfg->srcparam1.y * cfg->srcparam1.z) * cfg->srcnum;
                CUDA_ASSERT(cudaMalloc((void**)&gsrcpattern, patsize));
                CUDA_ASSERT(cudaMemcpy(gsrcpattern, cfg->srcpattern, patsize, cudaMemcpyHostToDevice));
            }

            /* Phase function inverse CDF */
            if (cfg->nphase) {
                CUDA_ASSERT(cudaMalloc((void**)&ginvcdf, sizeof(float) * cfg->nphase));
                CUDA_ASSERT(cudaMemcpy(ginvcdf, cfg->invcdf, sizeof(float) * cfg->nphase, cudaMemcpyHostToDevice));
            }

            /* Launch angle inverse CDF */
            if (cfg->nangle) {
                CUDA_ASSERT(cudaMalloc((void**)&gangleinvcdf, sizeof(float) * cfg->nangle));
                CUDA_ASSERT(cudaMemcpy(gangleinvcdf, cfg->angleinvcdf, sizeof(float) * cfg->nangle, cudaMemcpyHostToDevice));
            }

            /* Polarization scattering matrix */
            if (cfg->polmedianum && cfg->smatrix) {
                CUDA_ASSERT(cudaMalloc((void**)&gsmatrix, cfg->polmedianum * NANGLES * sizeof(float4)));
                CUDA_ASSERT(cudaMemcpy(gsmatrix, cfg->smatrix, cfg->polmedianum * NANGLES * sizeof(float4), cudaMemcpyHostToDevice));
            }

            /* Detected photon seed data (for replay) */
            if (cfg->issaveseed) {
                seeddata = (RandType*)calloc(cfg->maxdetphoton, sizeof(RandType) * RAND_BUF_LEN);
                CUDA_ASSERT(cudaMalloc((void**)&gseeddata, sizeof(RandType) * cfg->maxdetphoton * RAND_BUF_LEN));
                CUDA_ASSERT(cudaMemset(gseeddata, 0, sizeof(RandType) * cfg->maxdetphoton * RAND_BUF_LEN));
            }

            /* Debug trajectory data */
            if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
                uint jumpcount = 0;
                CUDA_ASSERT(cudaMalloc((void**)&gjumpdebug, sizeof(uint)));
                CUDA_ASSERT(cudaMemcpy(gjumpdebug, &jumpcount, sizeof(uint), cudaMemcpyHostToDevice));
                CUDA_ASSERT(cudaMalloc((void**)&gdebugdata, sizeof(float) * debuglen * cfg->maxjumpdebug));
            }

            /* Replay buffers */
            if (cfg->seed == SEED_FROM_FILE) {
                if (cfg->replay.weight) {
                    CUDA_ASSERT(cudaMalloc((void**)&greplayw, sizeof(float) * cfg->nphoton));
                    CUDA_ASSERT(cudaMemcpy(greplayw, cfg->replay.weight, sizeof(float) * cfg->nphoton, cudaMemcpyHostToDevice));
                }

                if (cfg->replay.tof) {
                    CUDA_ASSERT(cudaMalloc((void**)&greplaytof, sizeof(float) * cfg->nphoton));
                    CUDA_ASSERT(cudaMemcpy(greplaytof, cfg->replay.tof, sizeof(float) * cfg->nphoton, cudaMemcpyHostToDevice));
                }

                if (cfg->replay.detid) {
                    CUDA_ASSERT(cudaMalloc((void**)&greplaydetid, sizeof(int) * cfg->nphoton));
                    CUDA_ASSERT(cudaMemcpy(greplaydetid, cfg->replay.detid, sizeof(int) * cfg->nphoton, cudaMemcpyHostToDevice));
                }
            }

            /* Copy per-GPU param to device constant memory */
            CUDA_ASSERT(cudaMemcpyToSymbol(gcfg_const, &lparam, sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

            ltic0 = GetTimeMillis();

            /*=== Time-gate loop: multi-pass if gpu[gpuid].maxgate < totalgates ===*/
            for (int timegate = 0; timegate < totalgates; timegate += (int)gpu[gpuid].maxgate) {
                int curgate = MIN((int)gpu[gpuid].maxgate, totalgates - timegate);
                size_t tg_offset    = (size_t)timegate * fieldpergate;
                size_t cur_gfldlen  = fieldpergate * (size_t)curgate;
                float  twindow0 = cfg->tstart + timegate * cfg->tstep;
                float  twindow1 = twindow0 + curgate * cfg->tstep;

                MCX_FPRINTF(cfg->flog, "launching kernel for time window [%.1fns %.1fns] ...\n",
                            twindow0 * 1e9, twindow1 * 1e9);
                fflush(cfg->flog);

                for (liter = 0; liter < cfg->respin; liter++) {
                    MCX_FPRINTF(cfg->flog, "simulation run#%2d ... \n", liter + 1);
                    fflush(cfg->flog);

                    lparam.twin0   = twindow0;
                    lparam.twin1   = twindow1;
                    lparam.maxgate = (uint)curgate;
                    CUDA_ASSERT(cudaMemcpyToSymbol(gcfg_const, &lparam, sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

                    /* Reset per-iteration buffers */
                    CUDA_ASSERT(cudaMemset(gfield, 0, sizeof(float) * cur_gfldlen * (isrfforward ? 4 : 2)));
                    CUDA_ASSERT(cudaMemset(gdetected, 0, sizeof(uint)));

                    if (cfg->issaveseed && gseeddata) {
                        CUDA_ASSERT(cudaMemset(gseeddata, 0, sizeof(RandType) * cfg->maxdetphoton * RAND_BUF_LEN));
                    }

                    if (gjumpdebug) {
                        uint jumpcount = 0;
                        CUDA_ASSERT(cudaMemcpy(gjumpdebug, &jumpcount, sizeof(uint), cudaMemcpyHostToDevice));
                    }

                    *progress = 0;
                    ltic0 = GetTimeMillis();

                    /*=== Launch kernel ===*/
                    mcx_launch_kernel(
                        ispencil, isref, (cfg->mediabyte <= 4), issvmc, ispolarized,
                        mcgrid, mcblock, sharedbuf,
                        gmedia, gfield, genergy, gPseed,
                        gPdet, gproperty, gsrcpattern,
                        gdetpos, gprogress, gdetected,
                        greplayw, greplaytof, greplaydetid,
                        gseeddata, gjumpdebug, gdebugdata,
                        ginvcdf, gangleinvcdf,
                        gsmatrix);

                    /*=== Progress bar (primary thread only to avoid interleaved output) ===*/
                    if (threadid == 0 && (param.debuglevel & MCX_DEBUG_PROGRESS)) {
                        int p0 = 0, ndone;
                        float maxval = (lparam.threadphoton >> 1) * 4.5f;
                        mcx_progressbar(-0.f, cfg);

                        do {
                            ndone = *progress;

                            if (ndone > p0) {
                                mcx_progressbar(ndone / maxval, cfg);
                                p0 = ndone;
                            }

                            sleep_ms(100);
                        } while (p0 < (int)maxval);

                        mcx_progressbar(1.0f, cfg);
                        MCX_FPRINTF(cfg->flog, "\n");
                    }

                    CUDA_ASSERT(cudaDeviceSynchronize());
                    CUDA_ASSERT(cudaGetLastError());

                    ltic1 = GetTimeMillis();
                    ltoc += ltic1 - ltic0;
                    MCX_FPRINTF(cfg->flog, "kernel complete: %d ms\nretrieving flux ...\t", ltic1 - tic);
                    fflush(cfg->flog);

                    /*=== Retrieve detected photons ===*/
                    if (cfg->issavedet) {
                        CUDA_ASSERT(cudaMemcpy(&detected, gdetected, sizeof(uint), cudaMemcpyDeviceToHost));
                        CUDA_ASSERT(cudaMemcpy(Pdet, gPdet, sizeof(float) * cfg->maxdetphoton * hostdetreclen, cudaMemcpyDeviceToHost));

                        if (cfg->issaveseed && seeddata) {
                            CUDA_ASSERT(cudaMemcpy(seeddata, gseeddata, sizeof(RandType) * cfg->maxdetphoton * RAND_BUF_LEN, cudaMemcpyDeviceToHost));
                        }

                        if (detected > cfg->maxdetphoton) {
                            MCX_FPRINTF(cfg->flog, S_RED "WARNING: detected photons (%u) > maxdetphoton (%d)\n" S_RESET, detected, cfg->maxdetphoton);
                        } else {
                            MCX_FPRINTF(cfg->flog, "detected " S_BOLD "" S_BLUE "%d photons" S_RESET ", total: " S_BOLD "" S_BLUE "%d" S_RESET "\t", detected, cfg->detectedcount + detected);
                        }

                        detected = MIN(detected, cfg->maxdetphoton);

                        #pragma omp critical
                        {
                            cfg->his.detected += detected;

                            if (cfg->exportdetected) {
                                cfg->exportdetected = (float*)realloc(cfg->exportdetected, (cfg->detectedcount + detected) * hostdetreclen * sizeof(float));
                                memcpy(cfg->exportdetected + cfg->detectedcount * hostdetreclen, Pdet, detected * hostdetreclen * sizeof(float));

                                if (cfg->issaveseed && cfg->seeddata) {
                                    cfg->seeddata = realloc(cfg->seeddata, (cfg->detectedcount + detected) * sizeof(RandType) * RAND_BUF_LEN);
                                    memcpy(((RandType*)cfg->seeddata) + cfg->detectedcount * RAND_BUF_LEN, seeddata, detected * sizeof(RandType) * RAND_BUF_LEN);
                                }

                                cfg->detectedcount += detected;
                            }
                        }
                    }

                    /*=== Retrieve debug trajectory data ===*/
                    if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
                        uint debugrec = 0;
                        CUDA_ASSERT(cudaMemcpy(&debugrec, gjumpdebug, sizeof(uint), cudaMemcpyDeviceToHost));

                        if (debugrec > 0) {
                            if (debugrec > cfg->maxjumpdebug) {
                                MCX_FPRINTF(cfg->flog, S_RED "WARNING: saved trajectory (%u) > maxjumpdebug (%d)\n" S_RESET, debugrec, cfg->maxjumpdebug);
                            } else {
                                MCX_FPRINTF(cfg->flog, "saved %u trajectory positions, total: %d\t", debugrec, cfg->debugdatalen + debugrec);
                            }

                            debugrec = MIN(debugrec, cfg->maxjumpdebug);
                            #pragma omp critical
                            {
                                cfg->exportdebugdata = (float*)realloc(cfg->exportdebugdata, (cfg->debugdatalen + debugrec) * debuglen * sizeof(float));
                                CUDA_ASSERT(cudaMemcpy(cfg->exportdebugdata + cfg->debugdatalen, gdebugdata, sizeof(float) * debuglen * debugrec, cudaMemcpyDeviceToHost));
                                cfg->debugdatalen += debugrec;
                            }
                        }
                    }

                    mcx_flush(cfg);

                    /*=== Retrieve volumetric output and accumulate ===*/
                    if (cfg->issave2pt) {
                        int rawfieldmul = isrfforward ? 4 : 2;
                        float* rawfield = (float*)malloc(sizeof(float) * cur_gfldlen * rawfieldmul);
                        CUDA_ASSERT(cudaMemcpy(rawfield, gfield, sizeof(float) * cur_gfldlen * rawfieldmul, cudaMemcpyDeviceToHost));
                        MCX_FPRINTF(cfg->flog, "transfer complete: %d ms\n", GetTimeMillis() - tic);
                        fflush(cfg->flog);

                        for (size_t ii = 0; ii < cur_gfldlen; ii++) {
                            field[ii] = rawfield[ii];

                            if (!isrfforward && cfg->outputtype != otRF && cfg->outputtype != otRFmus) {
                                field[ii] += rawfield[ii + cur_gfldlen];
                            }
                        }

                        /* Accumulate Im part for RF forward into exportfield[fieldlen + tg_offset..] */
                        #pragma omp critical
                        {
                            if (cfg->exportfield) {
                                if (isrfforward) {
                                    for (size_t ii = 0; ii < cur_gfldlen; ii++) {
                                        cfg->exportfield[fieldlen + tg_offset + ii] += rawfield[ii + cur_gfldlen * 2] + rawfield[ii + cur_gfldlen * 3];
                                    }
                                } else {
                                    for (size_t ii = 0; ii < cur_gfldlen; ii++) {
                                        cfg->exportfield[fieldlen + tg_offset + ii] += rawfield[ii + cur_gfldlen];
                                    }
                                }
                            }
                        }

                        free(rawfield);

                        if (cfg->respin > 1) {
                            for (size_t ii = 0; ii < cur_gfldlen; ii++) {
                                field[cur_gfldlen + ii] += field[ii];
                            }
                        }

                        if (liter + 1 == cfg->respin) {
                            if (cfg->respin > 1) {
                                memcpy(field, field + cur_gfldlen, sizeof(float) * cur_gfldlen);
                            }

                            /* Accumulate Re part into exportfield[tg_offset..] */
                            #pragma omp critical
                            {
                                if (cfg->exportfield) {
                                    for (size_t ii = 0; ii < cur_gfldlen; ii++) {
                                        cfg->exportfield[tg_offset + ii] += field[ii];
                                    }
                                }
                            }
                        }
                    }

                    /*=== Retrieve energy ===*/
                    {
                        float* loc_energy = (float*)calloc(gpu[gpuid].autothread << 1, sizeof(float));
                        CUDA_ASSERT(cudaMemcpy(loc_energy, genergy, sizeof(float) * (gpu[gpuid].autothread << 1), cudaMemcpyDeviceToHost));
                        #pragma omp critical
                        {
                            for (uint ei = 0; ei < (uint)gpu[gpuid].autothread; ei++) {
                                cfg->energyesc += loc_energy[ei << 1];
                                cfg->energytot += loc_energy[(ei << 1) + 1];
                            }
                        }
                        free(loc_energy);
                    }

                    /*=== Re-seed for next respin ===*/
                    if (cfg->respin > 1 && RAND_SEED_LEN > 1 && cfg->seed != SEED_FROM_FILE) {
                        uint* new_seeds = (uint*)malloc(sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN);
                        #pragma omp critical
                        {
                            for (lj = 0; lj < gpu[gpuid].autothread * RAND_SEED_LEN; lj++) {
                                new_seeds[lj] = rand();
                            }
                        }
                        CUDA_ASSERT(cudaMemcpy(gPseed, new_seeds, sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN, cudaMemcpyHostToDevice));
                        free(new_seeds);
                    }
                } /* respin */
            } /* time gates */

            #pragma omp critical
            {
                if (cfg->runtime < ltoc) {
                    cfg->runtime = ltoc;
                }
            }

            /*=== Cleanup per-GPU GPU buffers ===*/
            if (gmedia)       {
                CUDA_ASSERT(cudaFree(gmedia));
            }

            if (gfield)       {
                CUDA_ASSERT(cudaFree(gfield));
            }

            if (gPseed)       {
                CUDA_ASSERT(cudaFree(gPseed));
            }

            if (genergy)      {
                CUDA_ASSERT(cudaFree(genergy));
            }

            if (gdetected)    {
                CUDA_ASSERT(cudaFree(gdetected));
            }

            if (gPdet)        {
                CUDA_ASSERT(cudaFree(gPdet));
            }

            if (gproperty)    {
                CUDA_ASSERT(cudaFree(gproperty));
            }

            if (gdetpos)      {
                CUDA_ASSERT(cudaFree(gdetpos));
            }

            if (gsrcpattern)  {
                CUDA_ASSERT(cudaFree(gsrcpattern));
            }

            if (gseeddata)    {
                CUDA_ASSERT(cudaFree(gseeddata));
            }

            if (gdebugdata)   {
                CUDA_ASSERT(cudaFree(gdebugdata));
            }

            if (gjumpdebug)   {
                CUDA_ASSERT(cudaFree(gjumpdebug));
            }

            if (ginvcdf)      {
                CUDA_ASSERT(cudaFree(ginvcdf));
            }

            if (gangleinvcdf) {
                CUDA_ASSERT(cudaFree(gangleinvcdf));
            }

            if (gsmatrix)     {
                CUDA_ASSERT(cudaFree(gsmatrix));
            }

            if (greplayw)     {
                CUDA_ASSERT(cudaFree(greplayw));
            }

            if (greplaytof)   {
                CUDA_ASSERT(cudaFree(greplaytof));
            }

            if (greplaydetid) {
                CUDA_ASSERT(cudaFree(greplaydetid));
            }

            if (greplayseed)  {
                CUDA_ASSERT(cudaFree(greplayseed));
            }

            if (gprogress) {
                cudaFreeHost((void*)progress);
            }

            /*=== Cleanup per-GPU host buffers ===*/
            if (field)    {
                free(field);
            }

            if (Pdet)     {
                free(Pdet);
            }

            if (Pseed)    {
                free(Pseed);
            }

            if (seeddata) {
                free(seeddata);
            }
        } /* if (gpuphoton > 0) */
    } /* end parallel region */

    /*=== Reset all devices after parallel section ===*/
#ifndef MCX_DISABLE_CUDA_DEVICE_RESET

    for (i = 0; (int)cfg->deviceid[i] > 0; i++) {
        CUDA_ASSERT(cudaSetDevice((int)cfg->deviceid[i] - 1));
        CUDA_ASSERT(cudaDeviceReset());
    }

#endif
    CUDA_ASSERT(cudaSetDevice(gpuid0));

    MCX_FPRINTF(cfg->flog, "simulated %zu photons (%zu) with CUDA (repeat x%d)\nMCX speed: %.2f photon/ms\n",
                cfg->nphoton, cfg->nphoton, cfg->respin,
                ((cfg->issavedet == FILL_MAXDETPHOTON) ? cfg->energytot : ((double)cfg->nphoton * ((cfg->respin > 1) ? cfg->respin : 1))) / MAX(1, cfg->runtime));
    MCX_FPRINTF(cfg->flog, "total simulated energy: %.2f\tabsorbed: " S_BOLD "" S_BLUE "%5.5f%%" S_RESET "\n",
                cfg->energytot, (cfg->energytot - cfg->energyesc) / cfg->energytot * 100.f);
    fflush(cfg->flog);

    /*=== Normalization ===*/
    float Vvox = cfg->steps.x * cfg->steps.y * cfg->steps.z;

    if (cfg->issave2pt && cfg->isnormalized) {
        float* scale = (float*)calloc(cfg->srcnum, sizeof(float));
        scale[0] = 1.f;
        int isnormalized = 0;
        MCX_FPRINTF(cfg->flog, "normalizing raw data ...\t");
        cfg->energyabs += cfg->energytot - cfg->energyesc;

        if (cfg->outputtype == otFlux || cfg->outputtype == otFluence) {
            scale[0] = cfg->unitinmm / (cfg->energytot * Vvox * cfg->tstep);

            if (cfg->outputtype == otFluence) {
                scale[0] *= cfg->tstep;
            }
        } else if (cfg->outputtype == otEnergy || cfg->outputtype == otL) {
            scale[0] = 1.f / cfg->energytot;
        } else if (MCX_IS_ADJOINT_TYPE(cfg->outputtype) && cfg->seed != SEED_FROM_FILE) {
            scale[0] = cfg->unitinmm * (cfg->extrasrclen + 1) / cfg->energytot;
        } else if (cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS || cfg->outputtype == otRF || cfg->outputtype == otRFmus || cfg->outputtype == otWLTOF || cfg->outputtype == otWPTOF) {
            if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
                int detid;

                for (detid = 1; detid <= (int)cfg->detnum; detid++) {
                    scale[0] = 0.f;

                    for (size_t ii = 0; ii < cfg->nphoton; ii++)
                        if ((cfg->replay.detid[ii] & 0xFFFF) == detid) {
                            scale[0] += cfg->replay.weight[ii];
                        }

                    if (scale[0] > 0.f) {
                        scale[0] = cfg->unitinmm / scale[0];
                    }

                    MCX_FPRINTF(cfg->flog, "normalization factor for detector %d alpha=%f\n", detid, scale[0]);
                    fflush(cfg->flog);
                    mcx_normalize(cfg->exportfield + (detid - 1) * dimxyz * param.maxgate, scale[0], dimxyz * param.maxgate, cfg->isnormalized, 0, 1);
                }

                isnormalized = 1;
            } else {
                scale[0] = 0.f;

                for (size_t ii = 0; ii < cfg->nphoton; ii++) {
                    scale[0] += cfg->replay.weight[ii];
                }

                if (scale[0] > 0.f) {
                    scale[0] = cfg->unitinmm / scale[0];
                }
            }
        }

        if (cfg->extrasrclen && cfg->srcid < 0) {
            scale[0] *= (cfg->extrasrclen + 1);
        }

        cfg->normalizer = scale[0];
        cfg->his.normalizer = scale[0];

        if (!isnormalized) {
            for (i = 0; i < cfg->srcnum; i++) {
                MCX_FPRINTF(cfg->flog, "source %d, normalization factor alpha=%f\n", (i + 1), scale[i]);
                fflush(cfg->flog);
                mcx_normalize(cfg->exportfield, scale[i], fieldlen / cfg->srcnum, cfg->isnormalized, i, cfg->srcnum);
            }
        }

        free(scale);
    }

    /* Adjoint post-processing: phi_src * phi_det (or grad.grad) per voxel */
    if (cfg->issave2pt && MCX_IS_ADJOINT_TYPE(cfg->outputtype) && cfg->seed != SEED_FROM_FILE && cfg->detdir != NULL && cfg->exportfield) {
        int isrf   = isrfforward;
        int isdual = MCX_IS_DUAL_ADJOINT_TYPE(cfg->outputtype);
        unsigned int Ns = cfg->extrasrclen + 1 - cfg->detnum;
        unsigned int Nd = cfg->detnum;
        unsigned int pure_voxels = (unsigned int)(cfg->dim.x * cfg->dim.y * cfg->dim.z);
        size_t adjointlen       = (size_t)pure_voxels * Ns * Nd;
        size_t single_exportlen = adjointlen * (isrf ? 2 : 1);
        size_t exportlen_adj    = single_exportlen * (isdual ? 2 : 1);

        /* Upload normalized fluences to GPU */
        float* gfield_re_cu = NULL, *gfield_im_cu = NULL;
        float* gadjoint_mua_cu = NULL, *gadjoint_tmp_cu = NULL;
        CUDA_ASSERT(cudaMalloc((void**)&gfield_re_cu, sizeof(float) * fieldlen));
        CUDA_ASSERT(cudaMemcpy(gfield_re_cu, cfg->exportfield, sizeof(float) * fieldlen, cudaMemcpyHostToDevice));

        if (isrf) {
            CUDA_ASSERT(cudaMalloc((void**)&gfield_im_cu, sizeof(float) * fieldlen));
            CUDA_ASSERT(cudaMemcpy(gfield_im_cu, cfg->exportfield + fieldlen, sizeof(float) * fieldlen, cudaMemcpyHostToDevice));
        }

        if (isdual) {
            CUDA_ASSERT(cudaMalloc((void**)&gadjoint_mua_cu, sizeof(float) * single_exportlen));
            CUDA_ASSERT(cudaMemset(gadjoint_mua_cu, 0, sizeof(float) * single_exportlen));
        }

        CUDA_ASSERT(cudaMalloc((void**)&gadjoint_tmp_cu, sizeof(float) * single_exportlen));
        CUDA_ASSERT(cudaMemset(gadjoint_tmp_cu, 0, sizeof(float) * single_exportlen));

        size_t adjblocksize = 256;
        size_t adjgridsize  = (pure_voxels + (unsigned int)adjblocksize - 1) / adjblocksize;

        if (isdual || cfg->outputtype == otAdjoint) {
            mcx_adjoint_kernel <<< (unsigned int)adjgridsize, (unsigned int)adjblocksize>>>(
                gfield_re_cu, gfield_im_cu,
                isdual ? gadjoint_mua_cu : gadjoint_tmp_cu,
                pure_voxels, (unsigned int)cfg->maxgate, Ns, Nd);
            CUDA_ASSERT(cudaDeviceSynchronize());
        }

        if (isdual || cfg->outputtype != otAdjoint) {
            unsigned int Nx = (unsigned int)cfg->dim.x;
            unsigned int Ny = (unsigned int)cfg->dim.y;
            mcx_adjoint_dcoeff_kernel <<< (unsigned int)adjgridsize, (unsigned int)adjblocksize>>>(
                gfield_re_cu, gfield_im_cu, gadjoint_tmp_cu,
                pure_voxels, (unsigned int)cfg->maxgate, Ns, Nd, Nx, Ny);
            CUDA_ASSERT(cudaDeviceSynchronize());
        }

        CUDA_ASSERT(cudaFree(gfield_re_cu));

        if (gfield_im_cu) {
            CUDA_ASSERT(cudaFree(gfield_im_cu));
        }

        cfg->exportfield = (float*)realloc(cfg->exportfield, sizeof(float) * exportlen_adj);

        if (isdual) {
            float* hmua    = (float*)malloc(sizeof(float) * single_exportlen);
            float* hsecond = (float*)malloc(sizeof(float) * single_exportlen);
            CUDA_ASSERT(cudaMemcpy(hmua,    gadjoint_mua_cu, sizeof(float) * single_exportlen, cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(hsecond, gadjoint_tmp_cu, sizeof(float) * single_exportlen, cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaFree(gadjoint_mua_cu));
            CUDA_ASSERT(cudaFree(gadjoint_tmp_cu));

            for (size_t k = 0; k < single_exportlen; k++) {
                hmua[k] *= -Vvox;
            }

            for (size_t k = 0; k < single_exportlen; k++) {
                hsecond[k] *= -cfg->unitinmm;
            }

            if (cfg->outputtype == otAdjointMuaMusp) {
                for (size_t vox = 0; vox < (size_t)pure_voxels; vox++) {
                    unsigned int medid = cfg->vol[vox] & 0xFF;
                    float opscale = 0.f;

                    if (medid < cfg->medianum) {
                        float mus   = cfg->prop[medid].mus;
                        float onemg = 1.f - cfg->prop[medid].g;

                        if (mus > 0.f && onemg > 0.f) {
                            opscale = 1.f / (3.f * onemg * onemg * mus * mus);
                        }
                    }

                    for (unsigned int sd = 0; sd < Ns * Nd; sd++) {
                        hsecond[vox + (size_t)sd * pure_voxels] *= opscale;

                        if (isrf) {
                            hsecond[vox + (size_t)sd * pure_voxels + adjointlen] *= opscale;
                        }
                    }
                }
            }

            if (!isrf) {
                memcpy(cfg->exportfield,              hmua,    adjointlen * sizeof(float));
                memcpy(cfg->exportfield + adjointlen, hsecond, adjointlen * sizeof(float));
            } else {
                memcpy(cfg->exportfield,                   hmua,                 adjointlen * sizeof(float));
                memcpy(cfg->exportfield + adjointlen,      hsecond,              adjointlen * sizeof(float));
                memcpy(cfg->exportfield + 2 * adjointlen,  hmua + adjointlen,    adjointlen * sizeof(float));
                memcpy(cfg->exportfield + 3 * adjointlen,  hsecond + adjointlen, adjointlen * sizeof(float));
            }

            free(hmua);
            free(hsecond);
        } else {
            CUDA_ASSERT(cudaMemcpy(cfg->exportfield, gadjoint_tmp_cu, sizeof(float) * single_exportlen, cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaFree(gadjoint_tmp_cu));

            float adj_scale = (cfg->outputtype == otAdjoint) ? -Vvox : -cfg->unitinmm;

            for (size_t k = 0; k < single_exportlen; k++) {
                cfg->exportfield[k] *= adj_scale;
            }

            if (cfg->outputtype == otAdjointMus || cfg->outputtype == otAdjointMusp) {
                for (size_t vox = 0; vox < (size_t)pure_voxels; vox++) {
                    unsigned int medid = cfg->vol[vox] & 0xFF;
                    float opscale = 0.f;

                    if (medid < cfg->medianum) {
                        float mus   = cfg->prop[medid].mus;
                        float onemg = 1.f - cfg->prop[medid].g;

                        if (mus > 0.f && onemg > 0.f) {
                            opscale = (cfg->outputtype == otAdjointMus)
                                      ? 1.f / (3.f * onemg * mus * mus)
                                      : 1.f / (3.f * onemg * onemg * mus * mus);
                        }
                    }

                    for (unsigned int sd = 0; sd < Ns * Nd; sd++) {
                        cfg->exportfield[vox + (size_t)sd * pure_voxels] *= opscale;

                        if (isrf) {
                            cfg->exportfield[vox + (size_t)sd * pure_voxels + adjointlen] *= opscale;
                        }
                    }
                }
            }

            exportlen_adj = single_exportlen;
        }

        fieldlen = exportlen_adj;
        MCX_FPRINTF(cfg->flog, "adjoint Jacobian computation complete : %d ms\n", GetTimeMillis() - tic);
    }

#ifndef MCX_CONTAINER

    if (cfg->issave2pt && cfg->parentid == mpStandalone) {
        MCX_FPRINTF(cfg->flog, "saving data to file ... %zu %d\t", fieldlen, cfg->maxgate);
        mcx_savedata(cfg->exportfield, fieldlen, cfg);
        MCX_FPRINTF(cfg->flog, "saving data complete : %d ms\n\n", GetTimeMillis() - tic);
        fflush(cfg->flog);
    }

    if (cfg->issavedet && cfg->parentid == mpStandalone && cfg->exportdetected) {
        cfg->his.unitinmm = cfg->unitinmm;
        cfg->his.savedphoton = cfg->detectedcount;
        cfg->his.totalphoton = cfg->nphoton;

        if (cfg->issaveseed) {
            cfg->his.seedbyte = sizeof(RandType) * RAND_BUF_LEN;
        }

        cfg->his.detected = cfg->detectedcount;
        mcx_savedetphoton(cfg->exportdetected, cfg->seeddata, cfg->detectedcount, 0, cfg);
    }

    if ((cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) && cfg->parentid == mpStandalone && cfg->exportdebugdata) {
        cfg->his.colcount = MCX_DEBUG_REC_LEN;
        cfg->his.savedphoton = cfg->debugdatalen;
        cfg->his.totalphoton = cfg->nphoton;
        cfg->his.detected = 0;
        mcx_savedetphoton(cfg->exportdebugdata, NULL, cfg->debugdatalen, 0, cfg);
    }

#endif

    if (gpu) {
        free(gpu);
    }
}