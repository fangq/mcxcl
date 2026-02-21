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

    uint i, j;
    int iter;
    float minstep = MIN(MIN(cfg->steps.x, cfg->steps.y), cfg->steps.z);
    float t, twindow0, twindow1;
    float fullload = 0.f;
    float* energy;
    uint detected = 0;
    uint tic, tic0, tic1, toc = 0;
    uint debuglen = MCX_DEBUG_REC_LEN + (cfg->istrajstokes << 2);
    size_t fieldlen;

    dim3 mcgrid, mcblock;
    uint sharedbuf;

    size_t dimxyz = cfg->dim.x * cfg->dim.y * cfg->dim.z *
                    ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D)
                     ? cfg->srcnum : (cfg->srcid == -1) ? (cfg->extrasrclen + 1) : 1);

    uint* media = (uint*)(cfg->vol);
    float* field;
    float* Pdet;
    uint* Pseed;
    RandType* seeddata = NULL;
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
    param.is2d        = is2d;
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

    int gpuid = cfg->deviceid[0] - 1;

    if (gpuid < 0) {
        mcx_error(-1, "GPU ID must be non-zero", __FILE__, __LINE__);
    }

    CUDA_ASSERT(cudaSetDevice(gpuid));

    /*=== Compute thread/block sizes ===*/
    if (!cfg->autopilot) {
        uint gates = (uint)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
        gpu[gpuid].autothread = cfg->nthread;
        gpu[gpuid].autoblock = cfg->nblocksize;

        if (cfg->maxgate == 0) {
            cfg->maxgate = gates;
        } else if (cfg->maxgate > gates) {
            cfg->maxgate = gates;
        }

        gpu[gpuid].maxgate = cfg->maxgate;
    }

    if (gpu[gpuid].autothread % gpu[gpuid].autoblock) {
        gpu[gpuid].autothread = (gpu[gpuid].autothread / gpu[gpuid].autoblock) * gpu[gpuid].autoblock;
    }

    if (gpu[gpuid].maxgate == 0 && dimxyz > 0) {
        size_t needmem = dimxyz + gpu[gpuid].autothread * sizeof(float4) * 4 + sizeof(float) * cfg->maxdetphoton * hostdetreclen + 10 * 1024 * 1024;
        gpu[gpuid].maxgate = (gpu[gpuid].globalmem - needmem) / dimxyz;
        gpu[gpuid].maxgate = MIN(((cfg->tend - cfg->tstart) / cfg->tstep + 0.5), gpu[gpuid].maxgate);
    }

    cfg->maxgate = (int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
    param.maxgate = cfg->maxgate;

    mcgrid.x = gpu[gpuid].autothread / gpu[gpuid].autoblock;
    mcblock.x = gpu[gpuid].autoblock;

    /*=== Compute workload ===*/
    fullload = 0.f;

    for (i = 0; cfg->deviceid[i]; i++) {
        fullload += cfg->workload[i];
    }

    if (fullload < EPS) {
        for (i = 0; cfg->deviceid[i]; i++) {
            cfg->workload[i] = gpu[cfg->deviceid[i] - 1].core;
        }

        fullload = 0.f;

        for (i = 0; cfg->deviceid[i]; i++) {
            fullload += cfg->workload[i];
        }
    }

    param.threadphoton = (int)(cfg->nphoton * cfg->workload[0] / (fullload * gpu[gpuid].autothread * cfg->respin));
    param.oddphoton = (int)(cfg->nphoton * cfg->workload[0] / (fullload * cfg->respin) - param.threadphoton * gpu[gpuid].autothread);

    /*=== Compute dimlen and fieldlen ===*/
    uint4 dimlen;
    dimlen.x = cfg->dim.x;
    dimlen.y = cfg->dim.x * cfg->dim.y;
    dimlen.z = cfg->dim.x * cfg->dim.y * cfg->dim.z;
    dimlen.w = cfg->maxgate * ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D)
                               ? cfg->srcnum : (cfg->srcid == -1) ? (cfg->extrasrclen + 1) : 1);

    if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
        dimlen.w *= cfg->detnum;
    }

    fieldlen = dimlen.z * dimlen.w;
    dimlen.w = fieldlen;
    param.dimlen = dimlen;

    /*=== Allocate host buffers ===*/
    if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
        field = (float*)calloc(sizeof(float) * dimxyz, cfg->maxgate * 2 * cfg->detnum);
    } else {
        field = (float*)calloc(sizeof(float) * dimxyz, cfg->maxgate * 2);
    }

    Pdet = (float*)calloc(cfg->maxdetphoton, sizeof(float) * hostdetreclen);
    energy = (float*)calloc(gpu[gpuid].autothread << 1, sizeof(float));

    if (cfg->seed != SEED_FROM_FILE) {
        Pseed = (uint*)malloc(sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN);
    } else {
        Pseed = (uint*)malloc(sizeof(RandType) * cfg->nphoton * RAND_BUF_LEN);
    }

    if (cfg->seed > 0) {
        srand(cfg->seed);
    } else {
        srand(time(0));
    }

    for (j = 0; j < gpu[gpuid].autothread * RAND_SEED_LEN; j++) {
        Pseed[j] = rand();
    }

    if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY) && cfg->exportdebugdata == NULL) {
        cfg->exportdebugdata = (float*)calloc(sizeof(float), debuglen * cfg->maxjumpdebug);
    }

    /*=== Allocate ALL GPU buffers ===*/
    uint*  gmedia    = NULL;
    float* gfield    = NULL;
    float* genergy   = NULL;
    uint*  gPseed    = NULL;
    float* gPdet     = NULL;
    uint*  gdetected = NULL;
    float4* gproperty = NULL;
    float4* gdetpos  = NULL;
    float* gsrcpattern = NULL;
    float* ginvcdf   = NULL;
    float* gangleinvcdf = NULL;
    float4* gsmatrix = NULL;
    RandType* gseeddata = NULL;
    float* gdebugdata = NULL;
    uint*  gjumpdebug = NULL;
    float* greplayw  = NULL;
    float* greplaytof = NULL;
    int*   greplaydetid = NULL;
    RandType* greplayseed = NULL;
    volatile uint* gprogress = NULL;

    /* Progress bar (pinned memory) */
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
        size_t propsize = cfg->medianum * sizeof(Medium);
        size_t srcsize  = cfg->srcdata ? cfg->extrasrclen * 4 * sizeof(float4) : 0;
        size_t detsize  = (cfg->detpos && cfg->detnum) ? cfg->detnum * sizeof(float4) : 0;
        size_t totalsize = propsize + srcsize + detsize;
        char* propbuf = (char*)malloc(totalsize);
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

    /* Detector positions (separate buffer for kernel arg) */
    if (cfg->detpos && cfg->detnum) {
        CUDA_ASSERT(cudaMalloc((void**)&gdetpos, cfg->detnum * sizeof(float4)));
        CUDA_ASSERT(cudaMemcpy(gdetpos, cfg->detpos, cfg->detnum * sizeof(float4), cudaMemcpyHostToDevice));
    }

    /* Output field (with shadow buffer for double-precision accumulation) */
    CUDA_ASSERT(cudaMalloc((void**)&gfield, sizeof(float) * fieldlen * 2));
    CUDA_ASSERT(cudaMemset(gfield, 0, sizeof(float) * fieldlen * 2));

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

    /* Shared memory size */
    sharedbuf = (param.nphaselen + param.nanglelen) * sizeof(float)
                + gpu[gpuid].autoblock * (cfg->issaveseed * (RAND_BUF_LEN * sizeof(RandType))
                                          + sizeof(float) * (param.w0offset + cfg->srcnum
                                                  + 2 * (cfg->outputtype == otRF || cfg->outputtype == otRFmus)));

    /* Copy param to constant memory */
    CUDA_ASSERT(cudaMemcpyToSymbol(gcfg_const, &param, sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

    /*=== Determine kernel specialization flags ===*/
    int ispencil = (cfg->srctype == MCX_SRC_PENCIL && cfg->nangle == 0);
    int isref = cfg->isreflect;
    int issvmc = (cfg->mediabyte == MEDIA_2LABEL_SPLIT);
    int ispolarized = (cfg->mediabyte <= 4) && (cfg->polmedianum > 0);

    for (i = 0; i < 6; i++)
        if (cfg->bc[i] == bcReflect || cfg->bc[i] == bcMirror) {
            isref = 1;
        }

    /*=== Print launch info ===*/
    mcx_printheader(cfg);
    MCX_FPRINTF(cfg->flog, "- code name: [Infinity-CUDA] compiled by nvcc [%d.%d]\n",
                __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    MCX_FPRINTF(cfg->flog, "- compiled with: [RNG] %s [Seed Length] %d\n", MCX_RNG_NAME, RAND_SEED_LEN);

    MCX_FPRINTF(cfg->flog, "GPU=%d (%s) threadph=%d extra=%d np=%.0f nthread=%d sharedbuf=%d\n",
                gpuid + 1, gpu[gpuid].name, param.threadphoton, param.oddphoton,
                (double)cfg->nphoton * cfg->workload[0] / fullload,
                (int)gpu[gpuid].autothread, sharedbuf);

    tic = StartTimer();

    if (cfg->exportfield == NULL) {
        if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
            cfg->exportfield = (float*)calloc(sizeof(float) * dimxyz, cfg->maxgate * 2 * cfg->detnum);
        } else {
            cfg->exportfield = (float*)calloc(sizeof(float) * dimxyz, cfg->maxgate * 2);
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

    /*=== Time-gate loop ===*/
    tic0 = GetTimeMillis();

    for (t = cfg->tstart; t < cfg->tend; t += cfg->tstep * cfg->maxgate) {
        twindow0 = t;
        twindow1 = t + cfg->tstep * cfg->maxgate;

        MCX_FPRINTF(cfg->flog, "launching kernel for time window [%.1fns %.1fns] ...\n",
                    twindow0 * 1e9, twindow1 * 1e9);
        fflush(cfg->flog);

        for (iter = 0; iter < cfg->respin; iter++) {
            MCX_FPRINTF(cfg->flog, "simulation run#%2d ... \n", iter + 1);
            fflush(cfg->flog);

            param.twin0 = twindow0;
            param.twin1 = twindow1;
            CUDA_ASSERT(cudaMemcpyToSymbol(gcfg_const, &param, sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

            /* Reset per-iteration buffers */
            CUDA_ASSERT(cudaMemset(gfield, 0, sizeof(float) * fieldlen * 2));
            CUDA_ASSERT(cudaMemset(gdetected, 0, sizeof(uint)));

            if (cfg->issaveseed && gseeddata) {
                CUDA_ASSERT(cudaMemset(gseeddata, 0, sizeof(RandType) * cfg->maxdetphoton * RAND_BUF_LEN));
            }

            if (gjumpdebug) {
                uint jumpcount = 0;
                CUDA_ASSERT(cudaMemcpy(gjumpdebug, &jumpcount, sizeof(uint), cudaMemcpyHostToDevice));
            }

            *progress = 0;
            tic0 = GetTimeMillis();

            /*=== Launch kernel (template-specialized) ===*/
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

            /*=== Progress bar ===*/
            if (param.debuglevel & MCX_DEBUG_PROGRESS) {
                int p0 = 0, ndone;
                float maxval = (param.threadphoton >> 1) * 4.5f;
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

            tic1 = GetTimeMillis();
            toc += tic1 - tic0;
            MCX_FPRINTF(cfg->flog, "kernel complete: %d ms\nretrieving flux ...\t", tic1 - tic);
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

                cfg->his.detected += detected;
                detected = MIN(detected, cfg->maxdetphoton);

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
                    cfg->exportdebugdata = (float*)realloc(cfg->exportdebugdata, (cfg->debugdatalen + debugrec) * debuglen * sizeof(float));
                    CUDA_ASSERT(cudaMemcpy(cfg->exportdebugdata + cfg->debugdatalen, gdebugdata, sizeof(float) * debuglen * debugrec, cudaMemcpyDeviceToHost));
                    cfg->debugdatalen += debugrec;
                }
            }

            mcx_flush(cfg);

            /*=== Retrieve volumetric output ===*/
            if (cfg->issave2pt) {
                float* rawfield = (float*)malloc(sizeof(float) * fieldlen * 2);
                CUDA_ASSERT(cudaMemcpy(rawfield, gfield, sizeof(float) * fieldlen * 2, cudaMemcpyDeviceToHost));
                MCX_FPRINTF(cfg->flog, "transfer complete: %d ms\n", GetTimeMillis() - tic);
                fflush(cfg->flog);

                for (size_t ii = 0; ii < fieldlen; ii++) {
                    field[ii] = rawfield[ii];

                    if (cfg->outputtype != otRF && cfg->outputtype != otRFmus) {
                        field[ii] += rawfield[ii + fieldlen];
                    }
                }

                free(rawfield);

                if (cfg->respin > 1) {
                    for (size_t ii = 0; ii < fieldlen; ii++) {
                        field[fieldlen + ii] += field[ii];
                    }
                }

                if (iter + 1 == cfg->respin) {
                    if (cfg->respin > 1) {
                        memcpy(field, field + fieldlen, sizeof(float) * fieldlen);
                    }
                }

                if (cfg->exportfield) {
                    for (size_t ii = 0; ii < fieldlen; ii++) {
                        cfg->exportfield[ii] += field[ii];
                    }
                }
            }

            /*=== Retrieve energy ===*/
            energy = (float*)calloc(gpu[gpuid].autothread << 1, sizeof(float));
            CUDA_ASSERT(cudaMemcpy(energy, genergy, sizeof(float) * (gpu[gpuid].autothread << 1), cudaMemcpyDeviceToHost));

            for (i = 0; i < gpu[gpuid].autothread; i++) {
                cfg->energyesc += energy[i << 1];
                cfg->energytot += energy[(i << 1) + 1];
            }

            free(energy);

            /*=== Re-seed for respin ===*/
            if (cfg->respin > 1 && RAND_SEED_LEN > 1 && cfg->seed != SEED_FROM_FILE) {
                Pseed = (uint*)malloc(sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN);

                for (j = 0; j < gpu[gpuid].autothread * RAND_SEED_LEN; j++) {
                    Pseed[j] = rand();
                }

                CUDA_ASSERT(cudaMemcpy(gPseed, Pseed, sizeof(RandType) * gpu[gpuid].autothread * RAND_BUF_LEN, cudaMemcpyHostToDevice));
                free(Pseed);
            }
        } /* respin */
    } /* time gates */

    if (cfg->runtime < toc) {
        cfg->runtime = toc;
    }

    /*=== Normalization ===*/
    /* ... (same normalization logic as mcx_host.cpp — omitted for brevity) ... */

    MCX_FPRINTF(cfg->flog, "simulated %zu photons (%zu) with CUDA (repeat x%d)\nMCX speed: %.2f photon/ms\n",
                cfg->nphoton, cfg->nphoton, cfg->respin,
                ((cfg->issavedet == FILL_MAXDETPHOTON) ? cfg->energytot : ((double)cfg->nphoton * ((cfg->respin > 1) ? cfg->respin : 1))) / MAX(1, cfg->runtime));
    MCX_FPRINTF(cfg->flog, "total simulated energy: %.2f\tabsorbed: " S_BOLD "" S_BLUE "%5.5f%%" S_RESET "\n",
                cfg->energytot, (cfg->energytot - cfg->energyesc) / cfg->energytot * 100.f);
    fflush(cfg->flog);

    /*=== Cleanup GPU ===*/
    CUDA_ASSERT(cudaFree(gmedia));
    CUDA_ASSERT(cudaFree(gfield));
    CUDA_ASSERT(cudaFree(gPseed));
    CUDA_ASSERT(cudaFree(genergy));
    CUDA_ASSERT(cudaFree(gdetected));

    if (gPdet) {
        CUDA_ASSERT(cudaFree(gPdet));
    }

    if (gproperty) {
        CUDA_ASSERT(cudaFree(gproperty));
    }

    if (gdetpos) {
        CUDA_ASSERT(cudaFree(gdetpos));
    }

    if (gsrcpattern) {
        CUDA_ASSERT(cudaFree(gsrcpattern));
    }

    if (gseeddata) {
        CUDA_ASSERT(cudaFree(gseeddata));
    }

    if (gdebugdata) {
        CUDA_ASSERT(cudaFree(gdebugdata));
    }

    if (gjumpdebug) {
        CUDA_ASSERT(cudaFree(gjumpdebug));
    }

    if (ginvcdf) {
        CUDA_ASSERT(cudaFree(ginvcdf));
    }

    if (gangleinvcdf) {
        CUDA_ASSERT(cudaFree(gangleinvcdf));
    }

    if (gsmatrix) {
        CUDA_ASSERT(cudaFree(gsmatrix));
    }

    if (greplayw) {
        CUDA_ASSERT(cudaFree(greplayw));
    }

    if (greplaytof) {
        CUDA_ASSERT(cudaFree(greplaytof));
    }

    if (greplaydetid) {
        CUDA_ASSERT(cudaFree(greplaydetid));
    }

    if (greplayseed) {
        CUDA_ASSERT(cudaFree(greplayseed));
    }

    cudaFreeHost((void*)progress);

#ifndef MCX_DISABLE_CUDA_DEVICE_RESET
    CUDA_ASSERT(cudaDeviceReset());
#endif

    /*=== Cleanup host ===*/
    free(field);
    free(Pdet);

    if (seeddata) {
        free(seeddata);
    }

    if (gpu) {
        free(gpu);
    }
}