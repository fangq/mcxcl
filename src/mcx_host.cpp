/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration \
**      -- OpenCL edition
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2026
**
**  \section sref Reference:
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**         "Scalable and massively parallel Monte Carlo photon transport simulations
**         for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504 (2018)
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include "mcx_host.h"
#include "mcx_tictoc.h"
#include "mcx_const.h"

#ifndef NANGLES
    #define NANGLES 1000
#endif

#define MAX_JIT_OPT_LEN  (MAX_PATH_LENGTH << 1)

#define UPARAM_TO_MACRO(macro,a,b) snprintf(macro+strlen(macro), MAX_JIT_OPT_LEN-strlen(macro)-1," -Dgcfg%s=%u ",    #b,(a.b))
#define IPARAM_TO_MACRO(macro,a,b) snprintf(macro+strlen(macro), MAX_JIT_OPT_LEN-strlen(macro)-1," -Dgcfg%s=%d ",    #b,(a.b))
#define FPARAM_TO_MACRO(macro,a,b) snprintf(macro+strlen(macro), MAX_JIT_OPT_LEN-strlen(macro)-1," -Dgcfg%s=%.10ef ",#b,(a.b))

/* Acquisition-phase error handling. On failure these set 'status' and break out of
 * the do{...}while(0) guard that wraps the run, so the single NULL-safe teardown that
 * follows runs for both the success and the error path; the deferred error is then
 * re-reported at the very end. This matters when MCXCL runs as a library (mcxlabcl/
 * pmcxcl), where mcx_error throws a C++ exception instead of calling exit(): a
 * clCreateBuffer/clBuildProgram/clCreateKernel failure (e.g. GPU out-of-memory) would
 * otherwise leak the context and all partially-allocated device buffers.
 *
 * NOTE: these expand to a bare 'if(...) break;' (break cannot live inside a do/while
 * wrapper here), so only use them as standalone statements or inside braced blocks,
 * never as the unbraced body of an if/else. 'break' inside one of the acquisition
 * for-loops exits that loop; an 'if (status != CL_SUCCESS) break;' after the loop then
 * leaves the guard. */
#define OCL_TRY(x)           if ((status = (x)) != CL_SUCCESS) { break; }
#define MCX_ASSERT_ALLOC(p)  if ((p) == NULL) { status = CL_OUT_OF_HOST_MEMORY; break; }

cl_event kernelevent;

const char* VendorList[] = {"Unknown", "NVIDIA", "AMD", "Intel", "IntelGPU", "AppleCPU", "AppleGPU", "ArmGPU"};

const char* sourceflag[] = {"-DMCX_SRC_PENCIL", "-DMCX_SRC_ISOTROPIC", "-DMCX_SRC_CONE",
                            "-DMCX_SRC_GAUSSIAN", "-DMCX_SRC_PLANAR", "-DMCX_SRC_PATTERN", "-DMCX_SRC_FOURIER",
                            "-DMCX_SRC_ARCSINE", "-DMCX_SRC_DISK", "-DMCX_SRC_FOURIERX", "-DMCX_SRC_FOURIERX2D",
                            "-DMCX_SRC_ZGAUSSIAN", "-DMCX_SRC_LINE", "-DMCX_SRC_SLIT", "-DMCX_SRC_PENCILARRAY",
                            "-DMCX_SRC_PATTERN3D", "-DMCX_SRC_HYPERBOLOID_GAUSSIAN", "-DMCX_SRC_RING"
                           };

const char* debugopt[] = {"-DMCX_DEBUG_RNG=1", "-DMCX_DEBUG_MOVE=2", "-DMCX_DEBUG_PROGRESS=4", "-DMCX_DEBUG_MOVE_ONLY=8"};

char* print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:
            return strdup("Success!");

        case CL_DEVICE_NOT_FOUND:
            return strdup("Device not found.");

        case CL_DEVICE_NOT_AVAILABLE:
            return strdup("Device not available");

        case CL_COMPILER_NOT_AVAILABLE:
            return strdup("Compiler not available");

        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return strdup("Memory object allocation failure");

        case CL_OUT_OF_RESOURCES:
            return strdup("Out of resources");

        case CL_OUT_OF_HOST_MEMORY:
            return strdup("Out of host memory");

        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return strdup("Profiling information not available");

        case CL_MEM_COPY_OVERLAP:
            return strdup("Memory copy overlap");

        case CL_IMAGE_FORMAT_MISMATCH:
            return strdup("Image format mismatch");

        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return strdup("Image format not supported");

        case CL_BUILD_PROGRAM_FAILURE:
            return strdup("Program build failure");

        case CL_MAP_FAILURE:
            return strdup("Map failure");

        case CL_INVALID_VALUE:
            return strdup("Invalid value");

        case CL_INVALID_DEVICE_TYPE:
            return strdup("Invalid device type");

        case CL_INVALID_PLATFORM:
            return strdup("Invalid platform");

        case CL_INVALID_DEVICE:
            return strdup("Invalid device");

        case CL_INVALID_CONTEXT:
            return strdup("Invalid context");

        case CL_INVALID_QUEUE_PROPERTIES:
            return strdup("Invalid queue properties");

        case CL_INVALID_COMMAND_QUEUE:
            return strdup("Invalid command queue");

        case CL_INVALID_HOST_PTR:
            return strdup("Invalid host pointer");

        case CL_INVALID_MEM_OBJECT:
            return strdup("Invalid memory object");

        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return strdup("Invalid image format descriptor");

        case CL_INVALID_IMAGE_SIZE:
            return strdup("Invalid image size");

        case CL_INVALID_SAMPLER:
            return strdup("Invalid sampler");

        case CL_INVALID_BINARY:
            return strdup("Invalid binary");

        case CL_INVALID_BUILD_OPTIONS:
            return strdup("Invalid build options");

        case CL_INVALID_PROGRAM:
            return strdup("Invalid program");

        case CL_INVALID_PROGRAM_EXECUTABLE:
            return strdup("Invalid program executable");

        case CL_INVALID_KERNEL_NAME:
            return strdup("Invalid kernel name");

        case CL_INVALID_KERNEL_DEFINITION:
            return strdup("Invalid kernel definition");

        case CL_INVALID_KERNEL:
            return strdup("Invalid kernel");

        case CL_INVALID_ARG_INDEX:
            return strdup("Invalid argument index");

        case CL_INVALID_ARG_VALUE:
            return strdup("Invalid argument value");

        case CL_INVALID_ARG_SIZE:
            return strdup("Invalid argument size");

        case CL_INVALID_KERNEL_ARGS:
            return strdup("Invalid kernel arguments");

        case CL_INVALID_WORK_DIMENSION:
            return strdup("Invalid work dimension");

        case CL_INVALID_WORK_GROUP_SIZE:
            return strdup("Invalid work group size");

        case CL_INVALID_WORK_ITEM_SIZE:
            return strdup("Invalid work item size");

        case CL_INVALID_GLOBAL_OFFSET:
            return strdup("Invalid global offset");

        case CL_INVALID_EVENT_WAIT_LIST:
            return strdup("Invalid event wait list");

        case CL_INVALID_EVENT:
            return strdup("Invalid event");

        case CL_INVALID_OPERATION:
            return strdup("Invalid operation");

        case CL_INVALID_GL_OBJECT:
            return strdup("Invalid OpenGL object");

        case CL_INVALID_BUFFER_SIZE:
            return strdup("Invalid buffer size");

        case CL_INVALID_MIP_LEVEL:
            return strdup("Invalid mip-map level");

        default:
            return strdup("Unknown");
    }
}


/*
   assert cuda memory allocation result
*/
void ocl_assess(int cuerr, const char* file, const int linenum) {
    if (cuerr != CL_SUCCESS) {
        mcx_error(-(int)cuerr, print_cl_errstring(cuerr), file, linenum);
    }
}

/*
   zero a device buffer by copying zeros from the host: on Apple GPUs, clEnqueueFillBuffer
   returns CL_SUCCESS but often leaves the old contents of a reused allocation in place
*/
static cl_int mcx_zero_buffer(cl_command_queue queue, cl_mem buf, size_t bytes) {
    void* zeros = calloc(bytes, 1);
    cl_int status = (zeros) ? clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, zeros, 0, NULL, NULL) : CL_OUT_OF_HOST_MEMORY;
    free(zeros);
    return status;
}

/**
  obtain GPU core number per MP, this replaces
  ConvertSMVer2Cores() in libcudautils to avoid
  extra dependency.
*/
int mcx_nv_corecount(int v1, int v2) {
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

/**
 * @brief Utility function to query GPU info and set active GPU
 *
 * This function query and list all available GPUs on the system and print
 * their parameters. This is used when -L or -I is used.
 *
 * @param[in,out] cfg: the simulation configuration structure
 * @param[out] info: the GPU information structure
 */

cl_platform_id mcx_list_gpu(Config* cfg, unsigned int* activedev, cl_device_id* activedevlist, GPUInfo** info) {

    uint i, j, k, cuid = 0, devnum;
    cl_uint numPlatforms;
    cl_platform_id platform = NULL, activeplatform = NULL;
    cl_device_type devtype[] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU};
    cl_context context;                 // compute context
    const char* devname[] = {"GPU", "CPU"};
    char pbuf[100];
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, 0, 0};
    cl_int status = 0;
    size_t deviceListSize;
    int totaldevice = 0;

    OCL_ASSERT((clGetPlatformIDs(0, NULL, &numPlatforms)));

    if (activedev) {
        *activedev = 0;
    }

    *info = (GPUInfo*)calloc(MAX_DEVICE, sizeof(GPUInfo));

    if (numPlatforms > 0) {
        cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
        OCL_ASSERT((clGetPlatformIDs(numPlatforms, platforms, NULL)));

        for (i = 0; i < numPlatforms; ++i) {

            platform = platforms[i];

            if (1) {
                OCL_ASSERT((clGetPlatformInfo(platforms[i],
                                              CL_PLATFORM_NAME, sizeof(pbuf), pbuf, NULL)));

                if (cfg->isgpuinfo) {
                    MCX_FPRINTF(stdout, S_YELLOW"Platform [%d] Name %s\n" S_RESET, i, pbuf);
                }

                cps[1] = (cl_context_properties)platform;

                for (j = 0; j < 2; j++) {
                    cl_device_id* devices = NULL;
                    context = clCreateContextFromType(cps, devtype[j], NULL, NULL, &status);

                    if (status != CL_SUCCESS) {
                        clReleaseContext(context);
                        continue;
                    }

                    OCL_ASSERT((clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize)));
                    devices = (cl_device_id*)malloc(deviceListSize);
                    OCL_ASSERT((clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceListSize, devices, NULL)));
                    devnum = deviceListSize / sizeof(cl_device_id);

                    totaldevice += devnum;

                    for (k = 0; k < devnum; k++) {
                        GPUInfo cuinfo = {{0}};
                        cuinfo.platformid = i;
                        cuinfo.id = cuid + 1;
                        cuinfo.iscpu = j;
                        cuinfo.devcount = devnum;

                        OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_NAME, 100, (void*)cuinfo.name, NULL)));
                        OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), (void*)&cuinfo.sm, NULL)));
                        OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), (void*)&cuinfo.globalmem, NULL)));
                        OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), (void*)&cuinfo.sharedmem, NULL)));
                        OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), (void*)&cuinfo.constmem, NULL)));
                        OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), (void*)&cuinfo.clock, NULL)));
                        cuinfo.maxgate = cfg->maxgate;
                        cuinfo.autoblock = 64;
                        cuinfo.core = cuinfo.sm;

                        if ((strstr(cuinfo.name, "NVIDIA") || strstr(pbuf, "NVIDIA")) && !cuinfo.iscpu) {
                            if (!strstr(pbuf, "Apple")) {
                                OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), (void*)&cuinfo.major, NULL)));
                                OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), (void*)&cuinfo.minor, NULL)));
                            }

                            cuinfo.core = cuinfo.sm * mcx_nv_corecount(cuinfo.major, cuinfo.minor);
                            cuinfo.vendor = dvNVIDIA;
                        } else if ((strstr(cuinfo.name, "AMD") || strstr(pbuf, "AMD")) && !cuinfo.iscpu) {
                            int corepersm = 0;

                            if (!strstr(pbuf, "Apple") && (clGetDeviceInfo(devices[k], CL_DEVICE_GFXIP_MAJOR_AMD, sizeof(cl_uint), (void*)&cuinfo.major, NULL)) == CL_SUCCESS) {
                                OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_GFXIP_MINOR_AMD, sizeof(cl_uint), (void*)&cuinfo.minor, NULL)));
                                OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_BOARD_NAME_AMD, MAX_SESSION_LENGTH - 1, (void*)cuinfo.name, NULL)));
                                OCL_ASSERT((clGetDeviceInfo(devices[k], CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD, sizeof(cl_uint), (void*)&corepersm, NULL)));
                                corepersm = (corepersm == 0) ? 2 : corepersm;
                                cuinfo.core = (cuinfo.sm * corepersm << 4);
                            }

                            cuinfo.autoblock = 64;
                            cuinfo.vendor = dvAMD;
                        } else if (strstr(cuinfo.name, "Intel") && strstr(cuinfo.name, "Graphics") && j == 0) {
                            cuinfo.autoblock = 64;
                            cuinfo.vendor = dvIntelGPU;
                        } else if (strstr(pbuf, "Intel") || strstr(cuinfo.name, "Intel")) {
                            cuinfo.vendor = dvIntel;
                        } else if (strstr(pbuf, "Mali") || strstr(cuinfo.name, "Mali")) {
                            cuinfo.vendor = dvArmGPU;
                        }

                        if (strstr(cuinfo.name, "Apple M")) {
                            cuinfo.vendor = dvAppleGPU;
                            cuinfo.autoblock = 64;
                            cuinfo.autothread = cuinfo.core * (16 * 56); // each Apple GPU core has 16 EU
                        } else if (strstr(cuinfo.name, "Apple")) {
                            cuinfo.vendor = dvAppleCPU;
                            cuinfo.autoblock = 1;
                            cuinfo.autothread = 2048;
                        } else if (strstr(pbuf, "Mali") || strstr(cuinfo.name, "Mali")) {
                            cuinfo.autothread = cuinfo.core * 1024;
                            cuinfo.autoblock = 32;
                        } else {
                            cuinfo.autothread = cuinfo.autoblock * cuinfo.core;
                        }

                        if (cfg->isgpuinfo) {
                            MCX_FPRINTF(stdout, S_BLUE "============ %s device ID %d [%d of %d]: %s  ============\n" S_RESET, devname[j], cuid, k + 1, devnum, cuinfo.name);
                            MCX_FPRINTF(stdout, " Device %d of %d:\t\t%s\n", cuid + 1, devnum, cuinfo.name);
                            MCX_FPRINTF(stdout, " Compute units   :\t%d core(s)\n", (uint)cuinfo.sm);
                            MCX_FPRINTF(stdout, " Global memory   :\t%.0f B\n", (double)cuinfo.globalmem);
                            MCX_FPRINTF(stdout, " Local memory    :\t%.0f B\n", (double)cuinfo.sharedmem);
                            MCX_FPRINTF(stdout, " Constant memory :\t%.0f B\n", (double)cuinfo.constmem);
                            MCX_FPRINTF(stdout, " Clock speed     :\t%d MHz\n", cuinfo.clock);

                            if (strstr(cuinfo.name, "NVIDIA") && !strstr(pbuf, "Apple")) {
                                MCX_FPRINTF(stdout, " Compute Capacity:\t%d.%d\n", cuinfo.major, cuinfo.minor);
                                MCX_FPRINTF(stdout, " Stream Processor:\t%d\n", cuinfo.core);
                            } else if (strstr(cuinfo.name, "AMD") && !strstr(pbuf, "Apple") && j == 0) {
                                MCX_FPRINTF(stdout, " GFXIP version:   \t%d.%d\n", cuinfo.major, cuinfo.minor);
                                MCX_FPRINTF(stdout, " Stream Processor:\t%d\n", cuinfo.core);
                            }

                            MCX_FPRINTF(stdout, " Vendor name    :\t%s\n", VendorList[cuinfo.vendor]);
                            MCX_FPRINTF(stdout, " Auto-thread    :\t%lu\n", (unsigned long)cuinfo.autothread);
                            MCX_FPRINTF(stdout, " Auto-block     :\t%lu\n", (unsigned long)cuinfo.autoblock);
                        }

                        if (activedevlist != NULL) {
                            if (cfg->deviceid[cuid++] == '1') {
                                memcpy((*info) + (*activedev), &cuinfo, sizeof(GPUInfo));
                                activedevlist[(*activedev)++] = devices[k];

                                if (activeplatform && activeplatform != platform) {
                                    mcx_error(-(int)99, S_RED "ERROR: one can not mix devices between different platforms\n" S_RESET, __FILE__, __LINE__);
                                }

                                activeplatform = platform;
                            }
                        } else {
                            cuid++;
                            memcpy((*info) + ((*activedev)++), &cuinfo, sizeof(GPUInfo));
                        }
                    }

                    if (devices) {
                        free(devices);
                    }

                    clReleaseContext(context);
                }
            }
        }

        *info = (GPUInfo*)realloc(*info, (*activedev) * sizeof(GPUInfo));

        for (i = 0; i < *activedev; i++) {
            (*info)[i].devcount = totaldevice;
        }

        free(platforms);
    }

    if (cfg->isgpuinfo == 2 && cfg->parentid == mpStandalone) {
        exit(0);
    }

    return activeplatform;
}


/*
   master driver code to run MC simulations
*/
void mcx_run_simulation(Config* cfg, float* fluence, float* totalenergy) {

    cl_uint i, j;
    cl_int iter;
    cl_float  minstep = MIN(MIN(cfg->steps.x, cfg->steps.y), cfg->steps.z);
    cl_float t, twindow0, twindow1;
    cl_float fullload = 0.f;
    int respinsplit = (cfg->respin < 0) ? -cfg->respin : 1;  /**< a negative respin splits the photons into -respin runs, a positive one repeats them all */
    cl_float* energy = NULL;
    cl_uint* progress = NULL;
    cl_uint detected = 0, workdev;

    cl_uint tic, tic0, tic1, toc = 0, debuglen = MCX_DEBUG_REC_LEN + (cfg->istrajstokes << 2);
    size_t fieldlen, fieldbytes;
#ifdef USE_FIXEDPOINT
    double fixedscale = 1.0;      /**< integer units per unit of output */
#endif
    cl_uint4 cp0 = {{cfg->crop0.x, cfg->crop0.y, cfg->crop0.z, cfg->crop0.w}};
    cl_uint4 cp1 = {{cfg->crop1.x, cfg->crop1.y, cfg->crop1.z, cfg->crop1.w}};
    cl_uint2 cachebox = {{0, 0}};
    cl_uint4 dimlen = {{0, 0, 0, 0}};

    cl_context mcxcontext = NULL;          // compute mcxcontext
    cl_command_queue* mcxqueue = NULL;   // compute command queue
    cl_program mcxprogram = NULL;          // compute mcxprogram
    cl_kernel* mcxkernel = NULL;            // compute mcxkernel
    cl_int status = 0;
    cl_device_id devices[MAX_DEVICE];
    cl_event* waittoread = NULL;
    cl_platform_id platform = NULL;

    cl_uint  totalcucore;
    cl_uint  devid = 0;
    cl_mem* gmedia = NULL, *gproperty = NULL, *gparam = NULL;
    cl_mem* gsmatrix = NULL;
    cl_mem greplaydetid = NULL, greplayw = NULL, greplaytof = NULL, *gsrcpattern = NULL;
    cl_mem* gfield = NULL, *gdetphoton = NULL, *gseed = NULL, *genergy = NULL, *gseeddata = NULL;
    cl_mem* gprogress = NULL, *gdetected = NULL, *gdetpos = NULL, *gjumpdebug = NULL, *gdebugdata = NULL, *ginvcdf = NULL, *gangleinvcdf = NULL;

    int isrfforward = (cfg->omega > 0.f && cfg->seed != SEED_FROM_FILE);  /**< RF forward: complex photon weights */
    int isquadfield = (isrfforward || cfg->outputtype == otRFmus);         /**< 4-quarter field [Re, Re_db, Im, Im_db] */
    size_t dimxyz = cfg->dim.x * cfg->dim.y * cfg->dim.z * ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) ? cfg->srcnum : mcx_fieldslots(cfg));

    cl_uint*  media = (cl_uint*)(cfg->vol);
    int usebytemedia = 0;      /**< upload the label volume as 1 byte/voxel (see MCX_MEDIA_BYTE) */
    cl_float*  field = NULL;

    float*  Pdet = NULL;
    float*  srcpw = NULL, *energytot = NULL, *energyabs = NULL; // for multi-srcpattern
    char opt[MAX_JIT_OPT_LEN] = {'\0'};
    GPUInfo* gpu = NULL;
    RandType* seeddata = NULL;
    RandType* Pseed = NULL;
    MCXHostRNG hostrng;     /* fills Pseed with the per-thread seeds */

    /**
     *                              |----------------------------------------------->  hostdetreclen  <--------------------------------------|
     *                                        |------------------------>    partialdata   <-------------------|
     * host detected photon buffer: detid (1), partial_scat (#media), partial_path (#media), momontum (#media), p_exit (3), v_exit(3), w0 (1)
     *                                        |--------------------------------------------->    w0offset   <-----------------------------------------------||<----- w0 (#srcnum) ----->|
     * gpu detected photon buffer:             partial_scat (#media), partial_path (#media), momontum (#media), E_escape (1), E_launch (1), w0 (1), srcid(1), w0_photonsharing (#srcnum)
     */

    unsigned int partialdata = (cfg->medianum - 1) * (SAVE_NSCAT(cfg->savedetflag) + SAVE_PPATH(cfg->savedetflag) + SAVE_MOM(cfg->savedetflag)); // buf len for media-specific data, copy from gpu to host
    unsigned int w0offset = partialdata + 4;  //< the extra 4 numbers are total-escaped-energy, total-launched-energy, initial-weight, source_id
    unsigned int hostdetreclen = partialdata + SAVE_DETID(cfg->savedetflag) + 3 * (SAVE_PEXIT(cfg->savedetflag) + SAVE_VEXIT(cfg->savedetflag)) + SAVE_W0(cfg->savedetflag) + 4 * SAVE_IQUV(cfg->savedetflag); // host-side det photon data buffer length
    unsigned int is2d = (cfg->dim.x == 1 ? 1 : (cfg->dim.y == 1 ? 2 : (cfg->dim.z == 1 ? 3 : 0)));

    if (is2d) {
        /**
         *  is2d is only kept non-zero when exactly one of the 3 dimensions has a length of 1;
         *  if 2 or 3 dimensions have a length of 1, treat the domain as a 3D volume (e.g. 1x1x1).
         */
        is2d = is2d * (((cfg->dim.x > 1) + (cfg->dim.y > 1) + (cfg->dim.z > 1)) == 2);
    }

    MCXParam param = {{{cfg->srcpos.x, cfg->srcpos.y, cfg->srcpos.z, cfg->srcpos.w},
            {cfg->srcdir.x, cfg->srcdir.y, cfg->srcdir.z, cfg->srcdir.w}, {cfg->srcparam1.x, cfg->srcparam1.y, cfg->srcparam1.z, cfg->srcparam1.w},
            {cfg->srcparam2.x, cfg->srcparam2.y, cfg->srcparam2.z, cfg->srcparam2.w}
        }, cfg->extrasrclen, cfg->srcid,
        {{(float)cfg->dim.x, (float)cfg->dim.y, (float)cfg->dim.z, 0}}, dimlen, cp0, cp1, cachebox,
        minstep, 0.f, 0.f, cfg->tend, R_C0* cfg->unitinmm, 0,
        (uint)cfg->issave2pt, (uint)cfg->isreflect, (uint)cfg->isrefint,
        (uint)cfg->issavedet, 1.f / cfg->tstep, cfg->minenergy,
        cfg->sradius* cfg->sradius, minstep* R_C0* cfg->unitinmm, cfg->maxdetphoton,
        cfg->medianum - 1, cfg->detnum, 0, 0, (uint)cfg->voidtime, (uint)cfg->srctype,
        (uint)cfg->maxvoidstep, cfg->issaveexit > 0, cfg->issaveseed > 0, (uint)cfg->issaveref,
        cfg->isspecular > 0, cfg->maxgate, cfg->seed, (uint)cfg->outputtype, 0, 0,
        (uint)cfg->debuglevel, cfg->savedetflag, hostdetreclen, partialdata, w0offset, (uint)cfg->mediabyte,
        (uint)cfg->maxjumpdebug, cfg->gscatter, is2d, cfg->replaydet, cfg->srcnum, cfg->nphase,
        cfg->nphase + (cfg->nphase & 0x1), cfg->nangle, cfg->nangle + (cfg->nangle & 0x1),
        cfg->polmedianum, (uint)cfg->istrajstokes,
        {{cfg->srciquv.x, cfg->srciquv.y, cfg->srciquv.z, cfg->srciquv.w}}, cfg->omega
    };

    /**
     * With cfg.sdmap, only the source and detector slots of its channels are simulated and stored:
     * the device sees them as the slots 1..nsrcslot, the first as the primary source (gcfg->src) and
     * the others as the extra sources, and only their detectors (the adjoint mode tells a detector
     * slot from the source slots by the detector count), so the kernel itself is unchanged
     */
    unsigned int nsrcslot = 0, Nsrc = cfg->extrasrclen + 1 - cfg->detnum;
    unsigned int* srcslots = mcx_getsrcslots(cfg, &nsrcslot);
    ExtraSrc* slotsrc = NULL;
    float4* slotdet = NULL;

    if (srcslots) {
        slotsrc = (ExtraSrc*)malloc(sizeof(ExtraSrc) * nsrcslot);
        slotdet = (float4*)malloc(sizeof(float4) * nsrcslot);
        param.detnum = 0;

        for (unsigned int k = 0; k < nsrcslot; k++) {
            if (srcslots[k] == 1) {
                slotsrc[k].srcpos = cfg->srcpos;
                slotsrc[k].srcdir = cfg->srcdir;
                slotsrc[k].srcparam1 = cfg->srcparam1;
                slotsrc[k].srcparam2 = cfg->srcparam2;
            } else {
                slotsrc[k] = cfg->srcdata[srcslots[k] - 2];
            }

            if (srcslots[k] > Nsrc) {
                slotdet[param.detnum++] = cfg->detpos[srcslots[k] - Nsrc - 1];
            }
        }

        memcpy(&param.src, slotsrc, sizeof(param.src));
        param.extrasrclen = nsrcslot - 1;
        MCX_FPRINTF(cfg->flog, "%s: %u of %u\n", T_("source and detector slots used by sdmap"), nsrcslot, cfg->extrasrclen + 1);
    }

    platform = mcx_list_gpu(cfg, &workdev, devices, &gpu);

    if (workdev > MAX_DEVICE) {
        workdev = MAX_DEVICE;
    }

    if (workdev == 0) {
        mcx_error(-(int)99, (char*)("Specified GPU does not exist"), __FILE__, __LINE__);
    }

    if (devices[0] == NULL) {
        OCL_ASSERT(-1);
    }

    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

    /* Use NULL for backward compatibility */
    cl_context_properties* cprops = (platform == NULL) ? NULL : cps;

    /* Guard the whole resource-acquisition + run region: any acquisition failure
     * (OCL_TRY/MCX_ASSERT_ALLOC) breaks out to the single NULL-safe teardown below
     * instead of throwing past it. See the OCL_TRY comment near the top of the file. */
    do {
        OCL_TRY(((mcxcontext = clCreateContext(cprops, workdev, devices, NULL, NULL, &status), status)));

        /* calloc (not malloc) so every per-device handle starts NULL: the cleanup path
         * can then release only what was created, even if acquisition aborted partway. */
        mcxqueue = (cl_command_queue*)calloc(workdev, sizeof(cl_command_queue));
        waittoread = (cl_event*)calloc(workdev, sizeof(cl_event));

        gseed = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gmedia = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gproperty = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gparam = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gfield = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gdetphoton = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        genergy = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gprogress = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gdetected = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gdetpos = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gjumpdebug = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gdebugdata = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gseeddata = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gsrcpattern = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        ginvcdf = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gangleinvcdf = (cl_mem*)calloc(workdev, sizeof(cl_mem));
        gsmatrix = (cl_mem*)calloc(workdev, sizeof(cl_mem));

        if (!mcxqueue || !waittoread || !gseed || !gmedia || !gproperty || !gparam || !gfield ||
                !gdetphoton || !genergy || !gprogress || !gdetected || !gdetpos || !gjumpdebug ||
                !gdebugdata || !gseeddata || !gsrcpattern || !ginvcdf || !gangleinvcdf || !gsmatrix) {
            status = CL_OUT_OF_HOST_MEMORY;
            break;
        }

        /* The block is to move the declaration of prop closer to its use */
        cl_command_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;

        totalcucore = 0;

        for (i = 0; i < workdev; i++) {
            OCL_TRY(((mcxqueue[i] = clCreateCommandQueue(mcxcontext, devices[i], prop, &status), status)));
            totalcucore += gpu[i].core;

            if (!cfg->autopilot) {
                uint gates = (uint)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
                gpu[i].autothread = cfg->nthread;
                gpu[i].autoblock = cfg->nblocksize;

                if (cfg->maxgate == 0) {
                    cfg->maxgate = gates;
                } else if (cfg->maxgate > gates) {
                    cfg->maxgate = gates;
                }

                gpu[i].maxgate = cfg->maxgate;
            } else {
                // persistent thread mode
                if (gpu[i].vendor == dvIntelGPU || gpu[i].vendor == dvIntel) { // Intel HD graphics GPU
                    gpu[i].autoblock  = (gpu[i].vendor == dvIntelGPU) ? 64 : 1;
                    gpu[i].autothread = gpu[i].autoblock * 7 * gpu[i].sm; // 7 thread x SIMD-16 per Exec Unit (EU)
                } else if (gpu[i].vendor == dvAMD) { // AMD GPU
                    gpu[i].autoblock  = 64;
                    gpu[i].autothread = 2560 * gpu[i].sm; // 40 wavefronts * 64 threads/wavefront
                } else if (gpu[i].vendor == dvNVIDIA) {
                    if (gpu[i].major == 2 || gpu[i].major == 3) { // fermi 2.x, kepler 3.x : max 7 blks per SM, 8 works better
                        gpu[i].autoblock  = 128;
                        gpu[i].autothread = gpu[i].autoblock * 8 * gpu[i].sm;
                    } else if (gpu[i].major == 5) { // maxwell 5.x
                        gpu[i].autoblock  = 64;
                        gpu[i].autothread = gpu[i].autoblock * 16 * gpu[i].sm;
                    } else if (gpu[i].major >= 6) { // pascal 6.x : max 32 blks per SM
                        gpu[i].autoblock  = 64;
                        gpu[i].autothread = gpu[i].autoblock * 64 * gpu[i].sm;
                    }
                }
            }

            if (gpu[i].autothread % gpu[i].autoblock) {
                gpu[i].autothread = (gpu[i].autothread / gpu[i].autoblock) * gpu[i].autoblock;
            }

            if (gpu[i].maxgate == 0 && dimxyz > 0) {
                int needmem = dimxyz + gpu[i].autothread * sizeof(float4) * 4 + sizeof(float) * cfg->maxdetphoton * hostdetreclen + 10 * 1024 * 1024; /*keep 10M for other things*/
                gpu[i].maxgate = (gpu[i].globalmem - needmem) / dimxyz;
                gpu[i].maxgate = MIN(((cfg->tend - cfg->tstart) / cfg->tstep + 0.5), gpu[i].maxgate);
            }
        }

        if (status != CL_SUCCESS) {
            break;    /* a clCreateCommandQueue above failed; leave the guard */
        }

        if (is2d) {
            float* vec = &(param.src.dir.x);

            if (ABS(vec[is2d - 1]) > EPS) {
                mcx_error(-1, "input domain is 2D, the initial direction can not have non-zero value in the singular dimension", __FILE__, __LINE__);
            }
        }

        cfg->maxgate = (int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
        param.maxgate = cfg->maxgate;

        fullload = 0.f;

        for (i = 0; i < workdev; i++) {
            fullload += cfg->workload[i];
        }

        if (fullload < EPS) {
            for (i = 0; i < workdev; i++) {
                cfg->workload[i] = gpu[i].core;
            }

            fullload = totalcucore;
        }

        if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
            field = (cl_float*)calloc(sizeof(cl_float) * dimxyz, cfg->maxgate * 2 * cfg->detnum);
        } else {
            field = (cl_float*)calloc(sizeof(cl_float) * dimxyz, cfg->maxgate * 2);
        }

        Pdet = (float*)calloc(cfg->maxdetphoton, sizeof(float) * hostdetreclen);
        MCX_ASSERT_ALLOC(field);
        MCX_ASSERT_ALLOC(Pdet);

        cachebox.x = (cp1.x - cp0.x + 1);
        cachebox.y = (cp1.y - cp0.y + 1) * (cp1.x - cp0.x + 1);
        dimlen.x = cfg->dim.x;
        dimlen.y = cfg->dim.x * cfg->dim.y;
        dimlen.z = cfg->dim.x * cfg->dim.y * cfg->dim.z;
        dimlen.w = cfg->maxgate * ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) ? cfg->srcnum : mcx_fieldslots(cfg));

        /** Here we decide the total output buffer, field's length. it is Nx*Ny*Nz*Nt*Ns.
         * dimlen.z and dimlen.w are 32-bit cl_uint, so cast to size_t before multiplying
         * to avoid a 32-bit overflow that would silently under-allocate the field buffer. */
        if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
            dimlen.w *= cfg->detnum;
            fieldlen = (size_t)dimlen.z * dimlen.w;
        } else {
            fieldlen = (size_t)dimlen.z * dimlen.w;
        }

        dimlen.w = fieldlen;

#ifdef USE_FIXEDPOINT
        /** one 64-bit integer per element and plane (Re, and Im for RF), as a plane of low words and one of high words */
        fieldbytes = sizeof(cl_ulong) * fieldlen * ((isquadfield || cfg->outputtype == otRF) ? 2 : 1);
        {
            /** the most one voxel can receive in one run, times 2: energy, the launched weight; other
                outputs, the weight times the longest path (grid units), and per scattering event for the
                replayed WP/DCS/RFmus outputs, in ns for the TOF outputs. In a forward run where every
                labeled medium absorbs, a photon's weighted path is at most 1/mua_min (2/mua_min allows
                for Russian roulette); replay and continuous media keep the longest path. See mcx_core.cu
                in mcx */
            double lmax = cfg->tend / R_C0 / cfg->unitinmm, lpath, k = 1.0, musmax = 0.0, muamin = 1e30, wmax = 1.0;
            int patlen = (cfg->srctype == MCX_SRC_PATTERN3D) ? (int)(cfg->srcparam1.x * cfg->srcparam1.y * cfg->srcparam1.z * cfg->srcnum)
                         : (int)(cfg->srcparam1.w * cfg->srcparam2.w * cfg->srcnum);

            for (unsigned int m = 1; m < cfg->medianum; m++) {
                musmax = MAX(musmax, (double)cfg->prop[m].mus);
                muamin = MIN(muamin, (double)cfg->prop[m].mua);
            }

            lpath = (cfg->seed != SEED_FROM_FILE && muamin > 0.0 && (cfg->mediabyte <= 4 || cfg->mediabyte == MEDIA_2LABEL_SPLIT
                     || cfg->mediabyte == MEDIA_2LABEL_MIX)) ? MIN(lmax, 2.0 / muamin) : lmax;

            for (int m = 0; cfg->srcpattern && (cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) && m < patlen; m++) {
                wmax = MAX(wmax, fabs(cfg->srcpattern[m]));
            }

            if (cfg->outputtype != otEnergy) {
                k = lpath * ((cfg->outputtype == otWP || cfg->outputtype == otDCS || cfg->outputtype == otWPTOF || cfg->outputtype == otRFmus) ? MAX(1.0, lmax * musmax) : 1.0)
                    * ((cfg->outputtype == otWLTOF || cfg->outputtype == otWPTOF) ? MAX(1.0, cfg->tend * 1e9) : 1.0);
            }

            fixedscale = exp2(floor(log2(4.611686018427388e18 / ((double)cfg->nphoton * wmax * MAX(k, 1.0) * 2.0))));
        }
#else
        fieldbytes = sizeof(cl_float) * fieldlen * (isquadfield ? 4 : 2);
#endif

        param.dimlen = dimlen;
        param.cachebox = cachebox;

        if (cfg->seed > 0) {
            mcx_hostrng_seed(&hostrng, cfg->seed);
        } else {
            mcx_hostrng_seed(&hostrng, time(0));
        }

        if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY) && cfg->exportdebugdata == NULL) {
            cfg->exportdebugdata = (float*)calloc(sizeof(float), (debuglen * cfg->maxjumpdebug));
        }

        /* ISO C/C++ has no cast from an object pointer to a function pointer: copy the address */
        void* nvaddr = clGetExtensionFunctionAddressForPlatform(platform, "clCreateBufferNV");
        cl_mem (*clCreateBufferNV)(cl_context, cl_mem_flags, cl_mem_flags_NV, size_t, void*, cl_int*);
        memcpy(&clCreateBufferNV, &nvaddr, sizeof(nvaddr));

        if (clCreateBufferNV == NULL) {
            OCL_TRY(((gprogress[0] = clCreateBuffer(mcxcontext, RW_PTR, sizeof(cl_uint), NULL, &status), status)));
        } else {
            gprogress[0] = clCreateBufferNV(mcxcontext, CL_MEM_READ_WRITE, NV_PIN, sizeof(cl_uint), NULL, &status);

            if (status == CL_INVALID_VALUE) {
                MCX_FPRINTF(cfg->flog, "Warning: to use the progress bar feature, you need to upgrade your NVIDIA GPU driver to 399.x or newer. Without the update, your progress bar may appear to be static (despite the simulation is running)\n");
                OCL_TRY(((gprogress[0] = clCreateBuffer(mcxcontext, RW_PTR, sizeof(cl_uint), NULL, &status), status)));
            }
        }

        progress = (cl_uint*)clEnqueueMapBuffer(mcxqueue[0], gprogress[0], CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_uint), 0, NULL, NULL, NULL);
        *progress = 0;

        if (cfg->seed == SEED_FROM_FILE) {
            // replay should only work with a single device
            OCL_TRY(((gseed[0] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(RandType) * cfg->nphoton * RAND_BUF_LEN, cfg->replay.seed, &status), status)));

            if (cfg->replay.weight) {
                OCL_TRY(((greplayw = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nphoton, cfg->replay.weight, &status), status)));
            }

            if (cfg->replay.tof) {
                OCL_TRY(((greplaytof = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nphoton, cfg->replay.tof, &status), status)));
            }

            if (cfg->replay.detid) {
                OCL_TRY(((greplaydetid = clCreateBuffer(mcxcontext, RO_MEM, sizeof(int) * cfg->nphoton, cfg->replay.detid, &status), status)));
            }
        }

        /** Plain label volumes only need a byte per voxel; the per-step label lookup is a
         *  random access over the whole domain, so packing it cuts that traffic 4x. Bit 7 of
         *  the packed value carries the DET_MASK detector flag, hence the 127-label limit. */
        if (cfg->mediabyte == MEDIA_LABEL_BYTE || cfg->mediabyte == MEDIA_LABEL_SHORT || cfg->mediabyte == MEDIA_LABEL_INT) {
            size_t nvox = (size_t)cfg->dim.x * cfg->dim.y * cfg->dim.z;
            usebytemedia = 1;

            for (size_t v = 0; v < nvox; v++)
                if ((media[v] & MED_MASK) > 127) {
                    usebytemedia = 0;
                    break;
                }
        }

        param.bytemedia = usebytemedia;

        for (i = 0; i < workdev; i++) {
            if (usebytemedia) {
                size_t nvox = (size_t)cfg->dim.x * cfg->dim.y * cfg->dim.z;
                unsigned char* bytemedia = (unsigned char*)malloc(nvox);

                for (size_t v = 0; v < nvox; v++) {
                    bytemedia[v] = (unsigned char)((media[v] & 0x7FU) | ((media[v] & DET_MASK) >> 24));
                }

                OCL_TRY(((gmedia[i] = clCreateBuffer(mcxcontext, RO_MEM, nvox, bytemedia, &status), status)));
                free(bytemedia);
            } else if (cfg->mediabyte != MEDIA_2LABEL_SPLIT && cfg->mediabyte != MEDIA_ASGN_F2H) {
                OCL_TRY(((gmedia[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(cl_uint) * (cfg->dim.x * cfg->dim.y * cfg->dim.z), media, &status), status)));
            } else {
                OCL_TRY(((gmedia[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(cl_uint) * (2 * cfg->dim.x * cfg->dim.y * cfg->dim.z), media, &status), status)));
            }

            if (cfg->srcdata) {
                float4* medsrc = (float4*)malloc(cfg->extrasrclen * 4 * sizeof(float4) + cfg->medianum * sizeof(Medium) + cfg->detnum * sizeof(float4));
                memcpy(medsrc, cfg->prop, cfg->medianum * sizeof(Medium));
                memcpy(medsrc + cfg->medianum, slotsrc ? slotsrc + 1 : cfg->srcdata, param.extrasrclen * 4 * sizeof(float4));
                OCL_TRY(((gproperty[i] = clCreateBuffer(mcxcontext, RO_MEM, param.extrasrclen * 4 * sizeof(float4) + cfg->medianum * sizeof(Medium), medsrc, &status), status)));
                free(medsrc);
            } else {
                OCL_TRY(((gproperty[i] = clCreateBuffer(mcxcontext, RO_MEM, cfg->medianum * sizeof(Medium), cfg->prop, &status), status)));
            }

            OCL_TRY(((gparam[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(MCXParam), &param, &status), status)));
            energy = (cl_float*)calloc(sizeof(cl_float), gpu[i].autothread << 1);

            if (cfg->seed != SEED_FROM_FILE) {
                Pseed = (RandType*)malloc(sizeof(RandType) * gpu[i].autothread * RAND_BUF_LEN);
                cl_uint* iseed = (cl_uint*)Pseed;

                for (j = 0; j < gpu[i].autothread * RAND_SEED_LEN; j++) {
                    iseed[j] = mcx_rand_uint32(&hostrng); /* same 32-bit seed words as mcx */
                }

                OCL_TRY(((gseed[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(RandType) * gpu[i].autothread * RAND_BUF_LEN, Pseed, &status), status)));
            }

            OCL_TRY(((gfield[i] = clCreateBuffer(mcxcontext, CL_MEM_READ_WRITE, fieldbytes, NULL, &status), status)));
            OCL_TRY((mcx_zero_buffer(mcxqueue[i], gfield[i], fieldbytes)));

            if (cfg->issavedet) {
                OCL_TRY(((gdetphoton[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(float) * cfg->maxdetphoton * hostdetreclen, Pdet, &status), status)));
            }

            OCL_TRY(((genergy[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(float) * (gpu[i].autothread << 1), energy, &status), status)));
            OCL_TRY(((gdetected[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(cl_uint), &detected, &status), status)));

            if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
                uint jumpcount = 0;
                OCL_TRY(((gjumpdebug[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(cl_uint), &jumpcount, &status), status)));
                OCL_TRY(((gdebugdata[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(float) * (debuglen * cfg->maxjumpdebug), cfg->exportdebugdata, &status), status)));
            }

            if (cfg->issaveseed) {
                seeddata = (RandType*)calloc(sizeof(RandType), cfg->maxdetphoton * RAND_BUF_LEN);
                OCL_TRY(((gseeddata[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(RandType) * cfg->maxdetphoton * RAND_BUF_LEN, seeddata, &status), status)));
                free(seeddata);
            }

            if (cfg->nphase) {
                OCL_TRY(((ginvcdf[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nphase, cfg->invcdf, &status), status)));
            }

            if (cfg->nangle) {
                OCL_TRY(((gangleinvcdf[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nangle, cfg->angleinvcdf, &status), status)));
            }

            if (cfg->detnum > 0) {
                OCL_TRY(((gdetpos[i] = clCreateBuffer(mcxcontext, RO_MEM, param.detnum * sizeof(float4), slotdet ? slotdet : cfg->detpos, &status), status)));
            }

            if (cfg->seed != SEED_FROM_FILE) {
                free(Pseed);
                Pseed = NULL;
            }

            free(energy);
            energy = NULL;

            if (cfg->srctype == MCX_SRC_PATTERN) {
                OCL_TRY(((gsrcpattern[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * (int)(cfg->srcparam1.w * cfg->srcparam2.w) * cfg->srcnum, cfg->srcpattern, &status), status)));
            } else if (cfg->srctype == MCX_SRC_PATTERN3D) {
                OCL_TRY(((gsrcpattern[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * (int)(cfg->srcparam1.x * cfg->srcparam1.y * cfg->srcparam1.z) * cfg->srcnum, cfg->srcpattern, &status), status)));
            } else {
                gsrcpattern[i] = NULL;
            }

            if (cfg->polmedianum && cfg->smatrix) {
                OCL_TRY(((gsmatrix[i] = clCreateBuffer(mcxcontext, RO_MEM, cfg->polmedianum * NANGLES * sizeof(float4), cfg->smatrix, &status), status)));
            } else {
                gsmatrix[i] = NULL;
            }


        }

        if (status != CL_SUCCESS) {
            break;    /* a per-device clCreateBuffer above failed; leave the guard */
        }

        mcx_printheader(cfg);

        tic = StartTimer();

#if __OPENCL_C_VERSION__
        MCX_FPRINTF(cfg->flog, "- %s: [Juxtaposition] %s OpenCL [%d] on [%s]\n",
                    T_("code name"), T_("compiled with"), __OPENCL_C_VERSION__, __DATE__);
#else
        MCX_FPRINTF(cfg->flog, "- %s: [Juxtaposition] %s OpenCL [%d] on [%s]\n",
                    T_("code name"), T_("compiled with"), CL_VERSION_1_0, __DATE__);
#endif

        MCX_FPRINTF(cfg->flog, "- %s: [RNG] %s [%s] %d\n", T_("compiled with"), MCX_RNG_NAME, T_("seed length"), RAND_SEED_LEN);
        MCX_FPRINTF(cfg->flog, "%s ...\t", T_("initializing streams"));

        MCX_FPRINTF(cfg->flog, "%s : %d ms\n", T_("init complete"), GetTimeMillis() - tic);
        fflush(cfg->flog);
        mcx_flush(cfg);

        OCL_TRY(((mcxprogram = clCreateProgramWithSource(mcxcontext, 1, (const char**) & (cfg->clsource), NULL, &status), status)));

        if (cfg->optlevel >= 1) {
            /** -cl-unsafe-math-optimizations implies -cl-mad-enable/-cl-no-signed-zeros and lets the
             *  JIT use approximate reciprocals; unlike -cl-fast-relaxed-math it keeps inf/nan handling,
             *  which hitgrid() relies on when a direction cosine is exactly zero.  -A 0 leaves it out,
             *  together with dynamic dispatch, so fixed-seed trajectories stay comparable with mcx.
             *  Saving seeds (-q 1) and replaying them (-E) leave it out too: the two runs build the kernel
             *  with different macros, and a replayed photon only retraces its path if both evaluate the
             *  float math identically, which the reassociation it allows does not guarantee. */
            int isunsafemath = cfg->autopilot && !cfg->issaveseed && cfg->seed != SEED_FROM_FILE;
            snprintf(opt, MAX_JIT_OPT_LEN - strlen(opt), "%s ", isunsafemath ? "-cl-mad-enable -cl-unsafe-math-optimizations -DMCX_USE_NATIVE" : "-cl-mad-enable -DMCX_USE_NATIVE");
        }

        if (cfg->optlevel >= 2) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", "-DMCX_SIMPLIFY_BRANCH -DMCX_VECTOR_INDEX");
        }

        if (cfg->optlevel >= 3 && cfg->seed != SEED_FROM_FILE) { // optlevel 3 - i.e. defining parameters as macros - fails replay test, need to debug
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", "-DUSE_MACRO_CONST");
        }

        /** Dynamic per-block photon dispatch: photon lifetimes vary by orders of magnitude, so a
         *  static photon-per-thread split makes every warp wait for its unluckiest lane.  Letting
         *  the threads of a block draw from a shared counter removes that tail.  Replay needs the
         *  static thread->photon mapping to index the saved seed/weight arrays, and -A 0 keeps it
         *  so that a fixed seed reproduces each thread's photon trajectories exactly. */
        if (cfg->seed != SEED_FROM_FILE && cfg->autopilot) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", "-DGROUP_LOAD_BALANCE");
        }

        for (i = 0; i < 3; i++)
            if (cfg->debuglevel & (1 << i)) {
                snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", debugopt[i]);
            }

        if ((uint)cfg->srctype < sizeof(sourceflag) / sizeof(sourceflag[0])) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", sourceflag[(uint)cfg->srctype]);
        }

        snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "-DMED_TYPE=%d ", cfg->mediabyte);

        if (usebytemedia) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", "-DMCX_MEDIA_BYTE");
        }

        snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", cfg->compileropt);

        if (cfg->issavedet) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", "-DMCX_SAVE_DETECTORS");
        }

        if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", "-DMCX_SAVE_TRAJECTORY");
        }

        if (MCX_IS_ADJOINT_TYPE(cfg->outputtype)) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), "%s ", "-DMCX_ADJOINT_MODE");
        }

        if (strstr(opt, "USE_MACRO_CONST")) {
            UPARAM_TO_MACRO(opt, param, detnum);
            UPARAM_TO_MACRO(opt, param, doreflect);
            UPARAM_TO_MACRO(opt, param, gscatter);
            UPARAM_TO_MACRO(opt, param, is2d);
            UPARAM_TO_MACRO(opt, param, issaveref);
            UPARAM_TO_MACRO(opt, param, issaveseed);
            UPARAM_TO_MACRO(opt, param, isspecular);
            UPARAM_TO_MACRO(opt, param, maxdetphoton);
            UPARAM_TO_MACRO(opt, param, maxgate);
            UPARAM_TO_MACRO(opt, param, maxjumpdebug);
            UPARAM_TO_MACRO(opt, param, maxmedia);
            UPARAM_TO_MACRO(opt, param, maxvoidstep);
            UPARAM_TO_MACRO(opt, param, mediaformat);
            FPARAM_TO_MACRO(opt, param, minaccumtime);
            FPARAM_TO_MACRO(opt, param, minenergy);
            FPARAM_TO_MACRO(opt, param, oneoverc0);
            UPARAM_TO_MACRO(opt, param, outputtype);
            UPARAM_TO_MACRO(opt, param, partialdata);
            UPARAM_TO_MACRO(opt, param, reclen);
            UPARAM_TO_MACRO(opt, param, save2pt);
            UPARAM_TO_MACRO(opt, param, savedet);
            UPARAM_TO_MACRO(opt, param, savedetflag);
            UPARAM_TO_MACRO(opt, param, seed);
            UPARAM_TO_MACRO(opt, param, srcnum);
            UPARAM_TO_MACRO(opt, param, voidtime);
            UPARAM_TO_MACRO(opt, param, w0offset);
            UPARAM_TO_MACRO(opt, param, nphase);
            UPARAM_TO_MACRO(opt, param, nangle);
            UPARAM_TO_MACRO(opt, param, nanglelen);
            UPARAM_TO_MACRO(opt, param, nphaselen);
            UPARAM_TO_MACRO(opt, param, extrasrclen);
            FPARAM_TO_MACRO(opt, param, Rtstep);
            UPARAM_TO_MACRO(opt, param, maxpolmedia);
            FPARAM_TO_MACRO(opt, param, omega);
            UPARAM_TO_MACRO(opt, param, istrajstokes);
            UPARAM_TO_MACRO(opt, param, debuglevel);

            /* below parameters are signed integers */
            IPARAM_TO_MACRO(opt, param, replaydet);
            IPARAM_TO_MACRO(opt, param, srcid);
        }

        char allabsorb[] = {bcAbsorb, bcAbsorb, bcAbsorb, bcAbsorb, bcAbsorb, bcAbsorb, 0};
        char allunknown[] = {0, 0, 0, 0, 0, 0, 0, 0};

        if (cfg->isreflect || (strcmp(cfg->bc, allabsorb) && strcmp(cfg->bc, allunknown))) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), " -DMCX_DO_REFLECTION");
        } else {
            /** Enable reflection flag when c or m flags are used in the cfg.bc boundary condition flags */
            for (i = 0; i < 6; i++)
                if (cfg->bc[i] == bcReflect || cfg->bc[i] == bcMirror) {
                    snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), " -DMCX_DO_REFLECTION");
                }
        }

        if (workdev == 1 && gpu[0].iscpu) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), " -DMCX_USE_CPU");
        }

        char extensions[8192] = {'\0'};
        clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);

        if (gpu[0].vendor == dvNVIDIA) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), " -DUSE_NVIDIA_GPU");
        } else if (strstr(extensions, "cl_intel_global_float_atomics")) {
            snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), " -DUSE_INTEL_FLOAT_ATOMIC");
        } else if (strstr(extensions, "cl_ext_float_atomics")) {
            /* the cl_ext_float_atomics functions (atomic_float, atomic_fetch_add_explicit) are OpenCL
               C 3.0 atomics: without -cl-std=CL3.0 the default OpenCL C 1.2 build fails (Intel Arc).
               OpenCL 3.0 devices compile OpenCL C 3.0 even when CL_DEVICE_OPENCL_C_VERSION (deprecated
               in 3.0) still says 1.2, as on Intel Arc; the kernel falls back to the CAS loop if the
               device does not define __opencl_c_ext_fp32_global_atomic_add, and so do older devices */
            char clver[256] = {'\0'};
            clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, sizeof(clver) - 1, clver, NULL);

            if (strncmp(clver, "OpenCL ", 7) == 0 && atoi(clver + 7) >= 3) {
                snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), " -cl-std=CL3.0 -DUSE_OPENCL_FLOAT_ATOMIC");
            }
        }

#ifdef USE_FIXEDPOINT
        snprintf(opt + strlen(opt), MAX_JIT_OPT_LEN - strlen(opt), " -DUSE_FIXEDPOINT -DMCX_FIXEDSCALE=%.1ff -DMCX_HIOFFSET=%luul",
                 fixedscale, (unsigned long)(fieldbytes / sizeof(cl_ulong)));
#endif
        MCX_FPRINTF(cfg->flog, "building kernel with option: %s\n", opt);
        status = clBuildProgram(mcxprogram, 0, NULL, opt, NULL, NULL);

        mcx_flush(cfg);

        size_t len;
        // get the details on the error, and store it in buffer
        clGetProgramBuildInfo(mcxprogram, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

        if (len > 0) {
            char* msg;
            msg = new char[len];
            clGetProgramBuildInfo(mcxprogram, devices[0], CL_PROGRAM_BUILD_LOG, len, msg, NULL);

            /* the NVIDIA driver reports this for every kernel, even a one-line one; drop it */
            const char* nvnote = "is a kernel, so overriding noinline attribute";
            char* hit;

            while ((hit = strstr(msg, nvnote)) != NULL) {
                char* lstart = hit, *lend = strchr(hit, '\n');

                while (lstart > msg && lstart[-1] != '\n') {
                    lstart--;
                }

                lend = lend ? lend + 1 : hit + strlen(hit);
                memmove(lstart, lend, strlen(lend) + 1);
            }

            len = strlen(msg);

            for (int i = 0; i < (int)len; i++)
                if (msg[i] <= 'z' && msg[i] >= 'A') {
                    MCX_FPRINTF(cfg->flog, "Kernel build log:\n%s\n", msg);
                    break;
                }

            delete [] msg;
        }

        if (status != CL_SUCCESS) {
            MCX_FPRINTF(cfg->flog, S_RED "Error: Failed to build program executable!\n" S_RESET);
            fflush(cfg->flog);
            break;   /* status holds the build error; reported after the teardown below */
        }

        MCX_FPRINTF(cfg->flog, "build program complete : %d ms\n", GetTimeMillis() - tic);
        fflush(cfg->flog);

        mcxkernel = (cl_kernel*)calloc(workdev, sizeof(cl_kernel));
        MCX_ASSERT_ALLOC(mcxkernel);

        for (i = 0; i < workdev; i++) {
            cl_int threadphoton, oddphoton, sharedbuf;

            /**
             * The per-thread photon data lives in local memory and grows with the number of media when
             * the per-medium detected photon data is saved; with ~200 media it exceeded the per-block limit
             * and the kernel launch failed with "out of resources". Halve the block size until it fits.
             */
            {
                size_t fixedbuf = (param.nphaselen + param.nanglelen) * sizeof(float);
                size_t threadbuf = cfg->issaveseed * (RAND_BUF_LEN * sizeof(RandType)) + sizeof(float) * (param.w0offset + cfg->srcnum + 2 * (cfg->outputtype == otRF || cfg->outputtype == otRFmus || MCX_IS_ADJOINT_TYPE(cfg->outputtype)));
                size_t oldblock = gpu[i].autoblock;

                while (gpu[i].autoblock > 1 && !(gpu[i].autoblock & 1) && fixedbuf + gpu[i].autoblock * threadbuf > gpu[i].sharedmem) {
                    gpu[i].autoblock >>= 1;
                }

                if (fixedbuf + gpu[i].autoblock * threadbuf > gpu[i].sharedmem) {
                    MCX_ERROR(-1, "the per-photon data does not fit in the device local memory, reduce the number of media, the saved detected photon data (-w) or the block size (-T)");
                }

                if (gpu[i].autoblock != oldblock) {
                    MCX_FPRINTF(cfg->flog, "%s: %d -> %d\n", T_("reduce block size to fit the shared memory"), (int)oldblock, (int)gpu[i].autoblock);
                }
            }

            threadphoton = (int)(cfg->nphoton * cfg->workload[i] / (fullload * gpu[i].autothread * respinsplit));
            oddphoton = (int)(cfg->nphoton * cfg->workload[i] / (fullload * respinsplit) - threadphoton * gpu[i].autothread);
            sharedbuf = (param.nphaselen + param.nanglelen) * sizeof(float) + gpu[i].autoblock * (cfg->issaveseed * (RAND_BUF_LEN * sizeof(RandType)) + sizeof(float) * (param.w0offset + cfg->srcnum + 2 * (cfg->outputtype == otRF || cfg->outputtype == otRFmus || MCX_IS_ADJOINT_TYPE(cfg->outputtype))));

            MCX_FPRINTF(cfg->flog, "- [device %d(%d): %s] threadph=%d extra=%d np=%.0f nthread=%d nblock=%d sharedbuf=%d\n", i, gpu[i].id, gpu[i].name, threadphoton, oddphoton,
                        cfg->nphoton * cfg->workload[i] / fullload, (int)gpu[i].autothread, (int)gpu[i].autoblock, sharedbuf);

            OCL_TRY(((mcxkernel[i] = clCreateKernel(mcxprogram, "mcx_main_loop", &status), status)));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 0, sizeof(cl_mem), (void*)(gmedia + i))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 1, sizeof(cl_mem), (void*)(gfield + i))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 2, sizeof(cl_mem), (void*)(genergy + i))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 3, sizeof(cl_mem), (void*)(gseed + ((cfg->seed != SEED_FROM_FILE) ? i : 0)))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 4, sizeof(cl_mem), (cfg->issavedet ? (void*)(gdetphoton + i) : NULL))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 5, sizeof(cl_mem), (void*)(gproperty + i))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 6, sizeof(cl_mem), (void*)(gsrcpattern + i))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 7, sizeof(cl_mem), (void*)(gdetpos + i))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 8, sizeof(cl_mem), (i == 0) ? ((void*)(gprogress)) : NULL)));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 9, sizeof(cl_mem), (void*)(gdetected + i))));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 10, sizeof(cl_mem), ((cfg->seed == SEED_FROM_FILE) ? (void*)(&greplayw) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 11, sizeof(cl_mem), ((cfg->seed == SEED_FROM_FILE) ? (void*)(&greplaytof) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 12, sizeof(cl_mem), ((cfg->seed == SEED_FROM_FILE) ? (void*)(&greplaydetid) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 13, sizeof(cl_mem), ((cfg->issaveseed) ? (void*)(gseeddata + i) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 14, sizeof(cl_mem), ((cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) ? (void*)(gjumpdebug + i) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 15, sizeof(cl_mem), ((cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) ? (void*)(gdebugdata + i) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 16, sizeof(cl_mem), ((cfg->nphase) ? (void*)(ginvcdf + i) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 17, sizeof(cl_mem), ((cfg->nangle) ? (void*)(gangleinvcdf + i) : NULL) )));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 18, sharedbuf, NULL)));
            OCL_ASSERT((clSetKernelArg(mcxkernel[i], 19, sizeof(cl_mem), (void*)(gsmatrix + i))));
        }

        if (status != CL_SUCCESS) {
            break;    /* a clCreateKernel above failed; leave the guard */
        }

        MCX_FPRINTF(cfg->flog, "set kernel arguments complete : %d ms\n", GetTimeMillis() - tic);
        fflush(cfg->flog);

        if (cfg->exportfield == NULL && cfg->issave2pt) {
            if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
                cfg->exportfield = (float*)calloc(sizeof(float) * dimxyz, cfg->maxgate * 2 * (1 + (cfg->outputtype == otRF || cfg->outputtype == otRFmus)) * cfg->detnum);
            } else {
                cfg->exportfield = (float*)calloc(sizeof(float) * fieldlen, 2 * (isrfforward ? 2 : 1));
            }
        }

        /* a volume times its time gates that does not fit in memory gives NULL,
           which everything after this writes through */
        if (cfg->exportfield == NULL && cfg->issave2pt) {
            MCX_ERROR(-2, "not enough memory for the output: reduce Dim, the number of time gates, or the detectors replayed");
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

        //simulate for all time-gates in maxgate groups per run

        cl_float Vvox;
        Vvox = cfg->steps.x * cfg->steps.y * cfg->steps.z;
        memcpy(&(param.bc), cfg->bc, 12);

        tic0 = GetTimeMillis();

        for (t = cfg->tstart; t < cfg->tend; t += cfg->tstep * cfg->maxgate) {
            twindow0 = t;
            twindow1 = t + cfg->tstep * cfg->maxgate;

            MCX_FPRINTF(cfg->flog, S_CYAN "%s [%.1fns %.1fns] ...\n" S_RESET
                        , T_("launching MCX simulation for time window"), twindow0 * 1e9, twindow1 * 1e9);
            fflush(cfg->flog);

            //total number of repetition for the simulations, results will be accumulated to field
            for (iter = 0; iter < ABS(cfg->respin); iter++) {
                MCX_FPRINTF(cfg->flog, "%s%2d ... \n", T_("simulation run#"), iter + 1);
                fflush(cfg->flog);
                fflush(cfg->flog);
                mcx_flush(cfg);

                param.twin0 = twindow0;
                param.twin1 = twindow1;

                for (devid = 0; devid < workdev; devid++) {
                    int nblock = gpu[devid].autothread / gpu[devid].autoblock;

                    param.threadphoton = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * gpu[devid].autothread * respinsplit));
                    param.oddphoton   = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * respinsplit) - param.threadphoton * gpu[devid].autothread);
                    param.blockphoton = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * nblock * respinsplit));
                    param.blockextra  = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * respinsplit) - param.blockphoton * nblock);
                    OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid], gparam[devid], CL_TRUE, 0, sizeof(MCXParam), &param, 0, NULL, NULL)));
                    OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 20, sizeof(cl_mem), (void*)(gparam + devid))));

                    // launch mcxkernel
                    OCL_ASSERT((clEnqueueNDRangeKernel(mcxqueue[devid], mcxkernel[devid], 1, NULL, &gpu[devid].autothread, &gpu[devid].autoblock, 0, NULL, &waittoread[devid])));
                    OCL_ASSERT((clFlush(mcxqueue[devid])));
                }

                if ((param.debuglevel & MCX_DEBUG_PROGRESS)) {
                    int p0 = 0, ndone = -1, kernelstatus = 0;
                    int threadphoton = (int)(cfg->nphoton * cfg->workload[0] / (fullload * gpu[0].autothread * respinsplit));
                    float maxval = ((threadphoton >> 1) * 4.5f);

                    if (workdev == 1 && gpu[0].iscpu) {
                        maxval = cfg->nphoton * 0.95f;
                    }

                    mcx_progressbar(-0.f, cfg);

                    do {
                        ndone = *progress;

                        if (ndone > p0) {
                            mcx_progressbar(ndone / maxval, cfg);
                            p0 = ndone;
                        }

                        sleep_ms(100);
                        OCL_ASSERT((clGetEventInfo(waittoread[0], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &kernelstatus, NULL)));

                        if (kernelstatus == CL_COMPLETE) {
                            break;
                        }
                    } while (p0 < (int)maxval);

                    mcx_progressbar(1.0f, cfg);
                    MCX_FPRINTF(cfg->flog, "\n");
                }

                /* NOTE: gprogress is mapped once before the loops and unmapped once in the
                 * teardown. Unmapping here (inside the gate/repetition loop) was a latent
                 * use-after-unmap: the next iteration's progress poll reads *progress after
                 * the buffer had already been unmapped (only safe today because the default
                 * single-gate/single-repetition case unmaps exactly once). */

                //clWaitForEvents(workdev,waittoread);
                for (devid = 0; devid < workdev; devid++) {
                    OCL_ASSERT((clFinish(mcxqueue[devid])));

                    /* release the kernel-launch event produced at clEnqueueNDRangeKernel
                     * (used only for the progress poll above). It is overwritten every
                     * gate/repetition iteration, so releasing it here (after the kernel has
                     * completed) avoids leaking one cl_event per iteration. On NVIDIA's
                     * OpenCL runtime a live event keeps an implicit reference to the queue/
                     * context, which prevents clReleaseContext from freeing the GPU
                     * allocations -> per-run device-memory leak. */
                    if (waittoread[devid]) {
                        clReleaseEvent(waittoread[devid]);
                        waittoread[devid] = NULL;
                    }
                }

                tic1 = GetTimeMillis();
                toc += tic1 - tic0;
                MCX_FPRINTF(cfg->flog, "%s:  \t%d ms\n%s ... \t", T_("kernel complete"), tic1 - tic, T_("retrieving fields"));
                fflush(cfg->flog);

                for (devid = 0; devid < workdev; devid++) {
                    if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
                        uint debugrec = 0;
                        OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gjumpdebug[devid], CL_TRUE, 0, sizeof(uint),
                                                        &debugrec, 0, NULL, NULL)));

                        if (debugrec > 0) {
                            if (debugrec > cfg->maxjumpdebug) {
                                MCX_FPRINTF(cfg->flog, S_RED "%s (%u > %d), %s\n" S_RESET,
                                            T_("WARNING: the saved trajectory positions are more than what your have specified"), debugrec, cfg->maxjumpdebug,
                                            T_("please use the --maxjumpdebug option to specify a greater number"));
                            } else {
                                MCX_FPRINTF(cfg->flog, "%s: %u, total: %d\t", T_("saved trajectory positions"), debugrec, cfg->debugdatalen + debugrec);
                            }

                            debugrec = MIN(debugrec, cfg->maxjumpdebug);
                            cfg->exportdebugdata = (float*)realloc(cfg->exportdebugdata, (cfg->debugdatalen + debugrec) * debuglen * sizeof(float));
                            OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gdebugdata[devid], CL_FALSE, 0, sizeof(float)*debuglen * debugrec,
                                                            cfg->exportdebugdata + cfg->debugdatalen, 0, NULL, NULL)));
                            cfg->debugdatalen += debugrec;
                        }
                    }

                    if (cfg->issavedet) {
                        OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gdetected[devid], CL_FALSE, 0, sizeof(uint),
                                                        &detected, 0, NULL, NULL)));
                        OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gdetphoton[devid], CL_TRUE, 0, sizeof(float)*cfg->maxdetphoton * hostdetreclen,
                                                        Pdet, 0, NULL, NULL)));

                        /** the detector (bits 0-15) and source slot (bits 16-31) of a detected photon are those on the device */
                        if (srcslots && SAVE_DETID(cfg->savedetflag)) {
                            unsigned int nsrcused = nsrcslot - param.detnum;

                            for (unsigned int p = 0; p < MIN(detected, (uint)cfg->maxdetphoton); p++) {
                                int detid = (int)Pdet[p * hostdetreclen], k = detid >> 16, d = detid & 0xFFFF;
                                d = (d >= 1 && d <= (int)param.detnum) ? (int)(srcslots[nsrcused + d - 1] - Nsrc) : d;
                                k = (k >= 1 && k <= (int)nsrcslot) ? (int)srcslots[k - 1] : k;
                                Pdet[p * hostdetreclen] = (float)(d | (k << 16));
                            }
                        }

                        if (cfg->issaveseed) {
                            seeddata = (RandType*)calloc(sizeof(RandType), cfg->maxdetphoton * RAND_BUF_LEN);
                            OCL_ASSERT(clEnqueueReadBuffer(mcxqueue[devid], gseeddata[devid], CL_TRUE, 0,
                                                           sizeof(RandType)*cfg->maxdetphoton * RAND_BUF_LEN, seeddata, 0, NULL, NULL));
                        }

                        if (detected > cfg->maxdetphoton) {
                            MCX_FPRINTF(cfg->flog, S_RED "%s (%u > %d), %s\t" S_RESET,
                                        T_("WARNING: the detected photon number is more than what your have specified"), detected, cfg->maxdetphoton,
                                        T_("please use the -H option to specify a greater number"));
                        } else {
                            MCX_FPRINTF(cfg->flog, "%s " S_BOLD S_BLUE "%d %s" S_RESET ", total: " S_BOLD S_BLUE "%d" S_RESET "\t", T_("detected"), detected, T_("photons"), cfg->detectedcount + detected);
                        }

                        cfg->his.detected += detected;
                        detected = MIN(detected, cfg->maxdetphoton);

                        /* realloc to 0 bytes, with no photon detected yet, would free the buffer */
                        if (cfg->exportdetected && detected > 0) {
                            cfg->exportdetected = (float*)realloc(cfg->exportdetected, (cfg->detectedcount + detected) * hostdetreclen * sizeof(float));

                            if (cfg->issaveseed && cfg->seeddata) {
                                cfg->seeddata = (RandType*)realloc(cfg->seeddata, (cfg->detectedcount + detected) * sizeof(RandType) * RAND_BUF_LEN);
                            }

                            memcpy(cfg->exportdetected + cfg->detectedcount * (hostdetreclen), Pdet, detected * (hostdetreclen)*sizeof(float));

                            if (cfg->issaveseed && cfg->seeddata) {
                                memcpy(((RandType*)cfg->seeddata) + cfg->detectedcount * RAND_BUF_LEN, seeddata, detected * sizeof(RandType)*RAND_BUF_LEN);
                            }

                            cfg->detectedcount += detected;
                        }

                        if (cfg->issaveseed) {
                            free(seeddata);
                        }
                    }

                    mcx_flush(cfg);

                    //handling the 2pt distributions
                    if (cfg->issave2pt) {
                        int rawfieldmul = isquadfield ? 4 : 2;
                        float* rawfield = (float*)malloc(sizeof(float) * fieldlen * rawfieldmul);

#ifdef USE_FIXEDPOINT

                        if (!(param.debuglevel & MCX_DEBUG_RNG)) {
                            /** convert the fixed-point Re (and Im) sums, low words then high words, to the float layout merged below */
                            cl_uint* q = (cl_uint*)malloc(fieldbytes);
                            size_t nel = fieldbytes / sizeof(cl_ulong);
                            OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gfield[devid], CL_TRUE, 0, fieldbytes, q, 0, NULL, NULL)));
                            memset(rawfield, 0, sizeof(float) * fieldlen * rawfieldmul);

                            for (i = 0; i < fieldlen; i++) {
                                rawfield[i] = (float)((cl_long)(((cl_ulong)q[nel + i] << 32) | q[i]) / fixedscale);

                                if (nel > fieldlen) {
                                    rawfield[i + (isquadfield ? 2 : 1) * fieldlen] = (float)((cl_long)(((cl_ulong)q[nel + i + fieldlen] << 32) | q[i + fieldlen]) / fixedscale);
                                }
                            }

                            free(q);
                        } else
#endif
                            OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gfield[devid], CL_TRUE, 0, sizeof(cl_float) * fieldlen * rawfieldmul,
                                                            rawfield, 0, NULL, NULL)));

                        MCX_FPRINTF(cfg->flog, "%s:\t%d ms\n", T_("transfer complete"), GetTimeMillis() - tic);
                        fflush(cfg->flog);

                        if (!(param.debuglevel & MCX_DEBUG_RNG)) {
                            for (i = 0; i < fieldlen; i++) { //accumulate field, can be done in the GPU
                                field[i] = rawfield[i];

                                if (isquadfield || cfg->outputtype != otRF) {
                                    field[i] += rawfield[i + fieldlen]; /* add the Re double-buffer, except in RF replay where [F..2F) holds Im */
                                }
                            }
                        } else {
                            memcpy(field, rawfield, sizeof(cl_float)*fieldlen);
                        }

                        if (isquadfield) {
                            /* RF forward/rfmus replay: Im part in 3rd quarter [2F..3F), shadow at [3F..4F) */
                            if (cfg->exportfield) {
                                for (i = 0; i < fieldlen; i++) {
                                    cfg->exportfield[i + fieldlen] += rawfield[i + fieldlen * 2] + rawfield[i + fieldlen * 3];
                                }
                            }
                        } else if (cfg->outputtype == otRF && cfg->omega > 0.f) {
                            if (cfg->exportfield) {
                                for (i = 0; i < fieldlen; i++) {
                                    cfg->exportfield[i + fieldlen] += rawfield[i + fieldlen];
                                }
                            }
                        }

                        free(rawfield);

                        if (ABS(cfg->respin) > 1) {
                            for (i = 0; i < fieldlen; i++) { //accumulate field, can be done in the GPU
                                field[fieldlen + i] += field[i];
                            }
                        }

                        if (iter + 1 == ABS(cfg->respin)) {
                            if (ABS(cfg->respin) > 1) { //copy the accumulated fields back
                                memcpy(field, field + fieldlen, sizeof(cl_float)*fieldlen);
                            }
                        }

                        /* the output takes the field once, after the last repetition has been
                           accumulated; adding it after every repetition counted the earlier ones again */
                        if (cfg->exportfield && iter + 1 == ABS(cfg->respin)) {
                            for (i = 0; i < fieldlen; i++) {
                                cfg->exportfield[i] += field[i];
                            }
                        }
                    }

                    energy = (cl_float*)calloc(sizeof(cl_float), gpu[devid].autothread << 1);
                    OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], genergy[devid], CL_TRUE, 0, sizeof(cl_float) * (gpu[devid].autothread << 1),
                                                    energy, 0, NULL, NULL)));

                    for (i = 0; i < gpu[devid].autothread; i++) {
                        cfg->energyesc += energy[(i << 1)];
                        cfg->energytot += energy[(i << 1) + 1];
                    }

                    free(energy);
                    energy = NULL;

                    //initialize the next simulation
                    if (twindow1 < cfg->tend && iter < ABS(cfg->respin)) {
                        OCL_ASSERT((mcx_zero_buffer(mcxqueue[devid], gfield[devid], fieldbytes)));
                        OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 1, sizeof(cl_mem), (void*)(gfield + devid))));
                    }

                    if (ABS(cfg->respin) > 1 && RAND_SEED_LEN > 1 && cfg->seed != SEED_FROM_FILE) {
                        Pseed = (RandType*)malloc(sizeof(RandType) * gpu[devid].autothread * RAND_BUF_LEN);
                        cl_uint* iseed = (cl_uint*)Pseed;

                        for (j = 0; j < gpu[devid].autothread * RAND_SEED_LEN; j++) {
                            iseed[j] = mcx_rand_uint32(&hostrng);
                        }

                        OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid], gseed[devid], CL_TRUE, 0, sizeof(RandType)*gpu[devid].autothread * RAND_BUF_LEN,
                                                         Pseed, 0, NULL, NULL)));
                        OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 3, sizeof(cl_mem), (void*)(gseed + devid))));
                        free(Pseed);
                        Pseed = NULL;
                    }

                    OCL_ASSERT((clFinish(mcxqueue[devid])));

                    if (twindow1 < cfg->tend) {
                        cl_float* tmpenergy = (cl_float*)calloc(sizeof(cl_float), gpu[devid].autothread * 3);
                        OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid], genergy[devid], CL_TRUE, 0, sizeof(cl_float) * (gpu[devid].autothread << 1),
                                                         tmpenergy, 0, NULL, NULL)));
                        OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 2, sizeof(cl_mem), (void*)(genergy + devid))));
                        free(tmpenergy);
                    }
                }// loop over work devices
            }// iteration
        }// time gates

        if (cfg->runtime < toc) {
            cfg->runtime = toc;
        }

        if (cfg->issave2pt && (cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) && cfg->srcnum > 1) { // post-processing only for multi-srcpattern
            srcpw = (float*)calloc(cfg->srcnum, sizeof(float));
            energytot = (float*)calloc(cfg->srcnum, sizeof(float));
            energyabs = (float*)calloc(cfg->srcnum, sizeof(float));
            uint psize = (int)cfg->srcparam1.w * (int)cfg->srcparam2.w;

            if (cfg->srctype == MCX_SRC_PATTERN3D) {
                psize = (int)cfg->srcparam1.x * (int)cfg->srcparam1.y * (int)cfg->srcparam1.z;
            }

            for (uint i = 0; i < cfg->srcnum; i++) {
                float kahanc = 0.f;

                for (uint j = 0; j < psize; j++) {
                    mcx_kahanSum(&srcpw[i], &kahanc, cfg->srcpattern[j * cfg->srcnum + i]);
                }

                energytot[i] = cfg->nphoton * srcpw[i] / (float)psize;
                kahanc = 0.f;

                if (cfg->outputtype == otEnergy) {
                    int fieldlenPsrc = fieldlen / cfg->srcnum;

                    for (int j = 0; j < fieldlenPsrc; j++) {
                        mcx_kahanSum(&energyabs[i], &kahanc, cfg->exportfield[j * cfg->srcnum + i]);
                    }
                } else {
                    for (uint j = 0; j < cfg->maxgate; j++)
                        for (uint k = 0; k < dimlen.z; k++) {
                            mcx_kahanSum(&energyabs[i], &kahanc, cfg->exportfield[j * dimxyz + (k * cfg->srcnum + i)]*mcx_updatemua((uint)cfg->vol[k], cfg));
                        }
                }
            }
        }

        if (cfg->issave2pt && cfg->isnormalized) {
            float* scale = (float*)calloc(cfg->srcnum, sizeof(float));
            scale[0] = 1.f;
            int isnormalized = 0;
            MCX_FPRINTF(cfg->flog, "%s\t", T_("normalizing raw data ..."));
            cfg->energyabs += cfg->energytot - cfg->energyesc;

            if (cfg->outputtype == otFlux || cfg->outputtype == otFluence || MCX_IS_ADJOINT_TYPE(cfg->outputtype) || (cfg->omega > 0.f && cfg->seed != SEED_FROM_FILE)) {
                scale[0] = cfg->unitinmm / (cfg->energytot * Vvox * cfg->tstep); /* Vvox (in mm^3 already) * (Tstep) * (Eabsorp/U) */

                if (cfg->outputtype == otFluence || MCX_IS_ADJOINT_TYPE(cfg->outputtype)) {
                    scale[0] *= cfg->tstep;
                }
            } else if (cfg->outputtype == otEnergy || cfg->outputtype == otL) {
                scale[0] = 1.f / cfg->energytot;
            } else if (cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS || cfg->outputtype == otRF || cfg->outputtype == otRFmus || cfg->outputtype == otWLTOF || cfg->outputtype == otWPTOF) {
                /**
                 * \c isvoxelunit flags the replay outputs whose raw data carries a length in voxel unit:
                 * an accumulated path length (jacobian/rf/wltof), or the 1/mus divided inside the kernel
                 * in SVMC mode (wp/wptof/rfmus). wltof/wptof are accumulated in nanoseconds by the kernel.
                 */
                int isvoxelunit = (cfg->outputtype == otJacobian || cfg->outputtype == otRF || cfg->outputtype == otWLTOF) ||
                                  (cfg->mediabyte == MEDIA_2LABEL_SPLIT && (cfg->outputtype == otWP || cfg->outputtype == otWPTOF || cfg->outputtype == otRFmus));
                float unitscale = ((isvoxelunit) ? cfg->unitinmm : 1.f) *
                                  ((cfg->outputtype == otWLTOF || cfg->outputtype == otWPTOF) ? 1e-9f : 1.f);

                if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
                    int detid;

                    for (detid = 1; detid <= (int)cfg->detnum; detid++) {
                        if (cfg->isnormalized != 2) {
                            scale[0] = 0.f; // the cfg->normalizer and cfg.his.normalizer are inaccurate in this case, but this is ok

                            for (size_t i = 0; i < cfg->nphoton; i++)
                                if ((cfg->replay.detid[i] & 0xFFFF) == detid) {
                                    scale[0] += cfg->replay.weight[i];
                                }

                            if (scale[0] > 0.f) {
                                scale[0] = unitscale / scale[0];
                            }
                        } else {
                            scale[0] = unitscale;
                        }

                        MCX_FPRINTF(cfg->flog, "%s %d alpha=%f\n", T_("normalization factor for detector"), detid, scale[0]);
                        fflush(cfg->flog);
                        mcx_normalize(cfg->exportfield + (detid - 1)*dimxyz * param.maxgate, scale[0], dimxyz * param.maxgate, 0, 1);

                        if (cfg->outputtype == otRF || cfg->outputtype == otRFmus) {
                            mcx_normalize(cfg->exportfield + fieldlen + (detid - 1)*dimxyz * param.maxgate, scale[0], dimxyz * param.maxgate, 0, 1);
                        }
                    }

                    isnormalized = 1;
                } else {
                    if (cfg->isnormalized != 2) {
                        scale[0] = 0.f;

                        for (size_t i = 0; i < cfg->nphoton; i++) {
                            scale[0] += cfg->replay.weight[i];
                        }

                        if (scale[0] > 0.f) {
                            scale[0] = unitscale / scale[0];
                        }
                    } else {
                        scale[0] = unitscale;
                    }
                }
            }

            if ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) && cfg->srcnum > 1) { // post-processing only for multi-srcpattern
                float scaleref = scale[0];
                int psize = (int)cfg->srcparam1.w * (int)cfg->srcparam2.w;

                if (cfg->srctype == MCX_SRC_PATTERN3D) {
                    psize = (int)cfg->srcparam1.x * (int)cfg->srcparam1.y * (int)cfg->srcparam1.z;
                }

                for (i = 0; i < cfg->srcnum; i++) {
                    scale[i] = psize / srcpw[i] * scaleref;
                }
            }

            if (cfg->extrasrclen && cfg->srcid < 0) { // when multiple sources are simulated, the total photons are evenly divided
                scale[0] *= mcx_fieldslots(cfg);
            }

            cfg->normalizer = scale[0];
            cfg->his.normalizer = scale[0];

            if (!isnormalized) {
                for (i = 0; i < cfg->srcnum; i++) {
                    MCX_FPRINTF(cfg->flog, "%s %d, %s alpha=%f\n", T_("source"), (i + 1), T_("normalization factor"), scale[i]);
                    fflush(cfg->flog);
                    mcx_normalize(cfg->exportfield, scale[i], fieldlen / cfg->srcnum * ((cfg->outputtype == otRF || cfg->outputtype == otRFmus || (cfg->omega > 0.f && cfg->seed != SEED_FROM_FILE) ? 1 : 0) + 1), i, cfg->srcnum);
                }
            }

            free(scale);
        } else if (cfg->issave2pt && (cfg->outputtype == otWLTOF || cfg->outputtype == otWPTOF) && cfg->exportfield) {
            /* unnormalized wltof/wptof output: the kernel accumulates the time-of-flight in
               nanoseconds only so that the double-buffer engages, an internal precision measure
               rather than a change of unit, so it is undone here too and the raw output stays in
               seconds, as with normalization (mcx 585f607) */
            mcx_normalize(cfg->exportfield, 1e-9f, fieldlen, 0, 1);
        }

        /* Adjoint post-processing: phi_src * phi_det (or grad.grad) per voxel */
        if (cfg->issave2pt && MCX_IS_ADJOINT_TYPE(cfg->outputtype) && cfg->seed != SEED_FROM_FILE && cfg->detdir != NULL && cfg->exportfield) {
            int isrf    = isrfforward;
            int isdual  = MCX_IS_DUAL_ADJOINT_TYPE(cfg->outputtype);
            unsigned int Ns = cfg->extrasrclen + 1 - cfg->detnum;
            cl_uint nch = (cl_uint)mcx_jacchannels(cfg); /* Ns*Nd, or the rows of cfg.sdmap */
            unsigned int pure_voxels = (unsigned int)(cfg->dim.x * cfg->dim.y * cfg->dim.z);
            size_t adjointlen    = (size_t)pure_voxels * nch;
            size_t single_exportlen = adjointlen * (isrf ? 2 : 1);
            size_t exportlen_adj = single_exportlen * (isdual ? 2 : 1);

            /* Upload normalized forward fluences to GPU (Re and Im parts as separate buffers) */
            cl_mem gfield_re = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * fieldlen, cfg->exportfield, &status);
            OCL_ASSERT(status);
            cl_mem gfield_im = (isrf) ? clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * fieldlen, cfg->exportfield + fieldlen, &status) : NULL;

            if (isrf) {
                OCL_ASSERT(status);
            }

            cl_mem gadjoint_mua = NULL, gadjoint_tmp = NULL;
            float* hmua = NULL, *hsecond = NULL;

            /* the 0-based (source, detector) pair of each Jacobian channel */
            unsigned int* hsdmap = mcx_getsdmap(cfg);

            /** with cfg.sdmap, the field only stores the slots in srcslots: address the channels by their field slots */
            if (srcslots) {
                for (unsigned int c = 0; c < 2 * nch; c++) {
                    unsigned int slot = ((c & 1) ? Ns : 0) + hsdmap[c] + 1, k = 0;

                    while (k + 1 < nsrcslot && srcslots[k] != slot) {
                        k++;
                    }

                    hsdmap[c] = k;
                }

                Ns = 0;
            }

            cl_mem gsdmap = clCreateBuffer(mcxcontext, RO_MEM, sizeof(cl_uint) * 2 * (nch > 0 ? nch : 1), hsdmap, &status);
            OCL_ASSERT(status);
            free(hsdmap);

            if (isdual) {
                gadjoint_mua = clCreateBuffer(mcxcontext, CL_MEM_READ_WRITE, sizeof(float) * single_exportlen, NULL, &status);
                OCL_ASSERT(status);
            }

            gadjoint_tmp = clCreateBuffer(mcxcontext, CL_MEM_READ_WRITE, sizeof(float) * single_exportlen, NULL, &status);
            OCL_ASSERT(status);

            size_t adjblocksize = 256;
            size_t adjgridsize  = (pure_voxels + (unsigned int)adjblocksize - 1) / adjblocksize * adjblocksize;

            if (isdual || cfg->outputtype == otAdjoint) {
                cl_kernel kadj = clCreateKernel(mcxprogram, "mcx_adjoint_kernel", &status);
                OCL_ASSERT(status);
                cl_mem gnull = NULL;
                OCL_ASSERT(clSetKernelArg(kadj, 0, sizeof(cl_mem), &gfield_re));
                OCL_ASSERT(clSetKernelArg(kadj, 1, sizeof(cl_mem), isrf ? &gfield_im : &gnull));
                OCL_ASSERT(clSetKernelArg(kadj, 2, sizeof(cl_mem), isdual ? &gadjoint_mua : &gadjoint_tmp));
                OCL_ASSERT(clSetKernelArg(kadj, 3, sizeof(cl_uint), &pure_voxels));
                cl_uint mgval = (cl_uint)cfg->maxgate;
                OCL_ASSERT(clSetKernelArg(kadj, 4, sizeof(cl_uint), &mgval));
                OCL_ASSERT(clSetKernelArg(kadj, 5, sizeof(cl_uint), &Ns));
                OCL_ASSERT(clSetKernelArg(kadj, 6, sizeof(cl_mem), &gsdmap));
                OCL_ASSERT(clSetKernelArg(kadj, 7, sizeof(cl_uint), &nch));
                OCL_ASSERT(clEnqueueNDRangeKernel(mcxqueue[0], kadj, 1, NULL, &adjgridsize, &adjblocksize, 0, NULL, NULL));
                OCL_ASSERT(clFinish(mcxqueue[0]));
                clReleaseKernel(kadj);
            }

            if (isdual || cfg->outputtype != otAdjoint) {
                cl_kernel kadj2 = clCreateKernel(mcxprogram, "mcx_adjoint_dcoeff_kernel", &status);
                OCL_ASSERT(status);
                cl_mem gnull = NULL;
                cl_uint Nx = (cl_uint)cfg->dim.x;
                cl_uint Ny = (cl_uint)cfg->dim.y;
                OCL_ASSERT(clSetKernelArg(kadj2, 0, sizeof(cl_mem), &gfield_re));
                OCL_ASSERT(clSetKernelArg(kadj2, 1, sizeof(cl_mem), isrf ? &gfield_im : &gnull));
                OCL_ASSERT(clSetKernelArg(kadj2, 2, sizeof(cl_mem), &gadjoint_tmp));
                OCL_ASSERT(clSetKernelArg(kadj2, 3, sizeof(cl_uint), &pure_voxels));
                cl_uint mgval = (cl_uint)cfg->maxgate;
                OCL_ASSERT(clSetKernelArg(kadj2, 4, sizeof(cl_uint), &mgval));
                OCL_ASSERT(clSetKernelArg(kadj2, 5, sizeof(cl_uint), &Ns));
                OCL_ASSERT(clSetKernelArg(kadj2, 6, sizeof(cl_mem), &gsdmap));
                OCL_ASSERT(clSetKernelArg(kadj2, 7, sizeof(cl_uint), &nch));
                OCL_ASSERT(clSetKernelArg(kadj2, 8, sizeof(cl_uint), &Nx));
                OCL_ASSERT(clSetKernelArg(kadj2, 9, sizeof(cl_uint), &Ny));
                OCL_ASSERT(clEnqueueNDRangeKernel(mcxqueue[0], kadj2, 1, NULL, &adjgridsize, &adjblocksize, 0, NULL, NULL));
                OCL_ASSERT(clFinish(mcxqueue[0]));
                clReleaseKernel(kadj2);
            }

            clReleaseMemObject(gfield_re);
            clReleaseMemObject(gsdmap);

            if (gfield_im) {
                clReleaseMemObject(gfield_im);
            }

            /* Allocate separate Jacobian buffer; exportfield retains the normalized forward fluence */
            if (cfg->exportjacob) {
                free(cfg->exportjacob);
            }

            cfg->exportjacob = (float*)malloc(sizeof(float) * exportlen_adj);

            if (isdual) {
                hmua    = (float*)malloc(sizeof(float) * single_exportlen);
                hsecond = (float*)malloc(sizeof(float) * single_exportlen);
                OCL_ASSERT(clEnqueueReadBuffer(mcxqueue[0], gadjoint_mua, CL_TRUE, 0, sizeof(float) * single_exportlen, hmua, 0, NULL, NULL));
                OCL_ASSERT(clEnqueueReadBuffer(mcxqueue[0], gadjoint_tmp, CL_TRUE, 0, sizeof(float) * single_exportlen, hsecond, 0, NULL, NULL));
                clReleaseMemObject(gadjoint_mua);
                clReleaseMemObject(gadjoint_tmp);

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

                        for (unsigned int sd = 0; sd < nch; sd++) {
                            hsecond[vox + (size_t)sd * pure_voxels] *= opscale;

                            if (isrf) {
                                hsecond[vox + (size_t)sd * pure_voxels + adjointlen] *= opscale;
                            }
                        }
                    }
                }

                if (!isrf) {
                    memcpy(cfg->exportjacob,              hmua,    adjointlen * sizeof(float));
                    memcpy(cfg->exportjacob + adjointlen, hsecond, adjointlen * sizeof(float));
                } else {
                    memcpy(cfg->exportjacob,                   hmua,                 adjointlen * sizeof(float));
                    memcpy(cfg->exportjacob + adjointlen,      hsecond,              adjointlen * sizeof(float));
                    memcpy(cfg->exportjacob + 2 * adjointlen,  hmua + adjointlen,    adjointlen * sizeof(float));
                    memcpy(cfg->exportjacob + 3 * adjointlen,  hsecond + adjointlen, adjointlen * sizeof(float));
                }

                free(hmua);
                free(hsecond);
            } else {
                OCL_ASSERT(clEnqueueReadBuffer(mcxqueue[0], gadjoint_tmp, CL_TRUE, 0, sizeof(float) * single_exportlen, cfg->exportjacob, 0, NULL, NULL));
                clReleaseMemObject(gadjoint_tmp);

                float adj_scale = (cfg->outputtype == otAdjoint) ? -Vvox : -cfg->unitinmm;

                for (size_t k = 0; k < single_exportlen; k++) {
                    cfg->exportjacob[k] *= adj_scale;
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

                        for (unsigned int sd = 0; sd < nch; sd++) {
                            cfg->exportjacob[vox + (size_t)sd * pure_voxels] *= opscale;

                            if (isrf) {
                                cfg->exportjacob[vox + (size_t)sd * pure_voxels + adjointlen] *= opscale;
                            }
                        }
                    }
                }
            }

            MCX_FPRINTF(cfg->flog, "adjoint Jacobian computation complete : %d ms\n", GetTimeMillis() - tic);
        }

#ifndef MCX_CONTAINER

        if (cfg->issave2pt && cfg->parentid == mpStandalone) {
            MCX_FPRINTF(cfg->flog, "%s ... %lu %d\t", T_("saving data to file"), (unsigned long)fieldlen, cfg->maxgate);

            /* the adjoint modes save the Jacobians, whose dimensions mcx_savedata uses, rather
               than the forward fluence of each source/detector slot kept in exportfield */
            if (MCX_IS_ADJOINT_TYPE(cfg->outputtype) && cfg->exportjacob && cfg->seed != SEED_FROM_FILE && cfg->detdir != NULL) {
                size_t jacoblen = (size_t)cfg->dim.x * cfg->dim.y * cfg->dim.z * mcx_jacchannels(cfg)
                                  * (MCX_IS_DUAL_ADJOINT_TYPE(cfg->outputtype) ? 2 : 1) * ((cfg->omega > 0.f) ? 2 : 1);
                mcx_savedata(cfg->exportjacob, jacoblen, cfg);
            } else {
                mcx_savedata(cfg->exportfield, fieldlen, cfg);
            }

            MCX_FPRINTF(cfg->flog, "%s : %d ms\n\n", T_("saving data complete"), GetTimeMillis() - tic);
            fflush(cfg->flog);
        }

        if (cfg->issavedet && cfg->parentid == mpStandalone && cfg->exportdetected && cfg->detectedcount > 0) {
            cfg->his.unitinmm = cfg->unitinmm;
            cfg->his.savedphoton = cfg->detectedcount;
            cfg->his.totalphoton = cfg->nphoton;

            if (cfg->issaveseed) {
                cfg->his.seedbyte = sizeof(RandType) * RAND_BUF_LEN;
            }

            cfg->his.detected = cfg->detectedcount;
            mcx_savedetphoton(cfg->exportdetected, cfg->seeddata, cfg->detectedcount, 0, cfg);
        }

        /** the 6th value of a trajectory record is its slot on the device */
        for (unsigned int r = 0; srcslots && cfg->exportdebugdata && r < cfg->debugdatalen; r++) {
            float* rec = cfg->exportdebugdata + (size_t)r * debuglen + 5;
            *rec = (*rec >= 1.f && *rec <= nsrcslot) ? srcslots[(int)(*rec) - 1] : *rec;
        }

        if ((cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) && cfg->parentid == mpStandalone && cfg->exportdebugdata) {
            cfg->his.colcount = debuglen;
            cfg->his.savedphoton = cfg->debugdatalen;
            cfg->his.totalphoton = cfg->nphoton;
            cfg->his.detected = 0;
            mcx_savedetphoton(cfg->exportdebugdata, NULL, cfg->debugdatalen, 0, cfg);
        }

#endif

        // total energy here equals total simulated photons+unfinished photons for all threads
        MCX_FPRINTF(cfg->flog, "%s %.0f %s (%.0f) with %d devices (repeat x%d)\n%s: " S_BOLD S_BLUE "%.2f photon/ms" S_RESET "\n",
                    T_("simulated"), (double)cfg->nphoton * ((cfg->respin > 1) ? (cfg->respin) : 1), T_("photons"),
                    (double)cfg->nphoton * ((cfg->respin > 1) ? (cfg->respin) : 1), workdev, ABS(cfg->respin), T_("MCX simulation speed"),
                    ((cfg->issavedet == FILL_MAXDETPHOTON) ? cfg->energytot : ((double)cfg->nphoton * ((cfg->respin > 1) ? (cfg->respin) : 1))) / MAX(1, cfg->runtime));

        if (cfg->issave2pt && cfg->srctype == MCX_SRC_PATTERN && cfg->srcnum > 1) {
            for (i = 0; i < cfg->srcnum; i++) {
                MCX_FPRINTF(cfg->flog, "%s #%d %s: %.2f\t%s: " S_BOLD S_BLUE "%5.5f%%" S_RESET "\n(%s)\n",
                            T_("source"), i + 1, T_("total simulated energy"), energytot[i], T_("absorbed"), energyabs[i] / energytot[i] * 100.f,
                            T_("loss due to initial specular reflection is excluded in the total"));
                fflush(cfg->flog);
            }
        } else {
            MCX_FPRINTF(cfg->flog, "%s: %.2f\t%s: " S_BOLD S_BLUE "%5.5f%%" S_RESET "\n(%s)\n",
                        T_("total simulated energy"), cfg->energytot, T_("absorbed"), (cfg->energytot - cfg->energyesc) / cfg->energytot * 100.f,
                        T_("loss due to initial specular reflection is excluded in the total"));
            fflush(cfg->flog);
        }

        cfg->energyabs = cfg->energytot - cfg->energyesc;

        status = CL_SUCCESS;   /* reached the end of the run with no error */
    } while (0);

    /* ---- single NULL-safe teardown, reached on both the success and error paths ----
     * Releases only what was actually created (handle arrays are calloc'd to NULL,
     * scalar handles are NULL-initialized), uses plain clRelease* (not OCL_ASSERT) so a
     * stray release error cannot mask the original failure, and re-reports any deferred
     * acquisition error at the very end. */

    /* gprogress is mapped once (above) and unmapped exactly once here before release */
    if (progress && mcxqueue && mcxqueue[0] && gprogress && gprogress[0]) {
        clEnqueueUnmapMemObject(mcxqueue[0], gprogress[0], progress, 0, NULL, NULL);
        clFinish(mcxqueue[0]);
    }

    if (gprogress && gprogress[0]) {
        clReleaseMemObject(gprogress[0]);
    }

    if (cfg->seed == SEED_FROM_FILE) {
        if (gseed && gseed[0]) {
            clReleaseMemObject(gseed[0]);
        }

        if (greplayw) {
            clReleaseMemObject(greplayw);
        }

        if (greplaytof) {
            clReleaseMemObject(greplaytof);
        }

        if (greplaydetid) {
            clReleaseMemObject(greplaydetid);
        }
    }

    for (i = 0; i < workdev; i++) {
        if (mcxqueue && mcxqueue[i]) {
            clFlush(mcxqueue[i]);
            clFinish(mcxqueue[i]);
        }

        if (gmedia && gmedia[i])         {
            clReleaseMemObject(gmedia[i]);
        }

        if (gproperty && gproperty[i])   {
            clReleaseMemObject(gproperty[i]);
        }

        if (gparam && gparam[i])         {
            clReleaseMemObject(gparam[i]);
        }

        if (gfield && gfield[i])         {
            clReleaseMemObject(gfield[i]);
        }

        if (gseed && gseed[i] && cfg->seed != SEED_FROM_FILE) {
            clReleaseMemObject(gseed[i]);
        }

        if (gdetphoton && gdetphoton[i]) {
            clReleaseMemObject(gdetphoton[i]);
        }

        if (genergy && genergy[i])       {
            clReleaseMemObject(genergy[i]);
        }

        if (gdetected && gdetected[i])   {
            clReleaseMemObject(gdetected[i]);
        }

        if (gsrcpattern && gsrcpattern[i]) {
            clReleaseMemObject(gsrcpattern[i]);
        }

        if (gjumpdebug && gjumpdebug[i]) {
            clReleaseMemObject(gjumpdebug[i]);
        }

        if (gdebugdata && gdebugdata[i]) {
            clReleaseMemObject(gdebugdata[i]);
        }

        if (gdetpos && gdetpos[i])       {
            clReleaseMemObject(gdetpos[i]);
        }

        if (gseeddata && gseeddata[i])   {
            clReleaseMemObject(gseeddata[i]);
        }

        if (ginvcdf && ginvcdf[i])       {
            clReleaseMemObject(ginvcdf[i]);
        }

        if (gangleinvcdf && gangleinvcdf[i]) {
            clReleaseMemObject(gangleinvcdf[i]);
        }

        if (gsmatrix && gsmatrix[i])     {
            clReleaseMemObject(gsmatrix[i]);
        }

        if (mcxkernel && mcxkernel[i])   {
            clReleaseKernel(mcxkernel[i]);
        }

        if (waittoread && waittoread[i]) {
            clReleaseEvent(waittoread[i]);
        }
    }

    free(gmedia);
    free(gproperty);
    free(gparam);
    free(gsrcpattern);
    free(gfield);
    free(gseed);
    free(gdetphoton);
    free(genergy);
    free(gprogress);
    free(gdetected);
    free(gdetpos);
    free(gjumpdebug);
    free(gdebugdata);
    free(gseeddata);
    free(ginvcdf);
    free(gangleinvcdf);
    free(gsmatrix);
    free(mcxkernel);
    free(waittoread);
    free(Pdet);

    if (gpu) {
        free(gpu);
    }

    for (devid = 0; devid < workdev; devid++) {
        if (mcxqueue && mcxqueue[devid]) {
            clReleaseCommandQueue(mcxqueue[devid]);
        }
    }

    free(mcxqueue);

    if (mcxprogram) {
        clReleaseProgram(mcxprogram);
    }

    if (mcxcontext) {
        clReleaseContext(mcxcontext);
    }

    if (kernelevent) {
        clReleaseEvent(kernelevent);
    }

    free(field);
    free(srcpw);
    free(srcslots);
    free(slotsrc);
    free(slotdet);
    free(energytot);
    free(energyabs);
    free(Pseed);    /* NULL on the success path; non-NULL only if acquisition aborted mid buffer-loop */
    free(energy);

    if (status != CL_SUCCESS) {
        ocl_assess(status, __FILE__, __LINE__);   /* re-raise the deferred acquisition error after full cleanup */
    }
}
