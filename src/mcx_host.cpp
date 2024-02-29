/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration \
**      -- OpenCL edition
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2023
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

#define IPARAM_TO_MACRO(macro,a,b) sprintf(macro+strlen(macro)," -Dgcfg%s=%u ",   #b,(a.b))
#define FPARAM_TO_MACRO(macro,a,b) sprintf(macro+strlen(macro)," -Dgcfg%s=%.10e ",#b,(a.b))

cl_event kernelevent;

const char* VendorList[] = {"Unknown", "NVIDIA", "AMD", "Intel", "IntelGPU", "AppleCPU"};

const char* sourceflag[] = {"-DMCX_SRC_PENCIL", "-DMCX_SRC_ISOTROPIC", "-DMCX_SRC_CONE",
                            "-DMCX_SRC_GAUSSIAN", "-DMCX_SRC_PLANAR", "-DMCX_SRC_PATTERN", "-DMCX_SRC_FOURIER",
                            "-DMCX_SRC_ARCSINE", "-DMCX_SRC_DISK", "-DMCX_SRC_FOURIERX", "-DMCX_SRC_FOURIERX2D",
                            "-DMCX_SRC_ZGAUSSIAN", "-DMCX_SRC_LINE", "-DMCX_SRC_SLIT", "-DMCX_SRC_PENCILARRAY",
                            "-DMCX_SRC_PATTERN3D"
                           };

const char* debugopt[] = {"-DMCX_DEBUG_RNG", "-DMCX_DEBUG_MOVE", "-DMCX_DEBUG_PROGRESS", "-DMCX_DEBUG_MOVE_ONLY"};

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
    } else if (v < 60 || v == 61) {
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
                        } else if (strstr(cuinfo.name, "Intel")) {
                            cuinfo.vendor = dvIntel;
                        }

                        if (strstr(pbuf, "Apple") && j > 0) {
                            cuinfo.vendor = dvAppleCPU;
                            cuinfo.autoblock = 1;
                            cuinfo.autothread = 2048;
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
                            MCX_FPRINTF(stdout, " Auto-thread    :\t%ld\n", cuinfo.autothread);
                            MCX_FPRINTF(stdout, " Auto-block     :\t%ld\n", cuinfo.autoblock);
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

    cl_uint i, j, iter;
    cl_float  minstep = MIN(MIN(cfg->steps.x, cfg->steps.y), cfg->steps.z);
    cl_float t, twindow0, twindow1;
    cl_float fullload = 0.f;
    cl_float* energy;
    cl_uint* progress = NULL;
    cl_uint detected = 0, workdev;

    cl_uint tic, tic0, tic1, toc = 0, debuglen = MCX_DEBUG_REC_LEN;
    size_t fieldlen;
    cl_uint4 cp0 = {{cfg->crop0.x, cfg->crop0.y, cfg->crop0.z, cfg->crop0.w}};
    cl_uint4 cp1 = {{cfg->crop1.x, cfg->crop1.y, cfg->crop1.z, cfg->crop1.w}};
    cl_uint2 cachebox = {{0, 0}};
    cl_uint4 dimlen = {{0, 0, 0, 0}};

    cl_context mcxcontext;                 // compute mcxcontext
    cl_command_queue* mcxqueue;          // compute command queue
    cl_program mcxprogram;                 // compute mcxprogram
    cl_kernel* mcxkernel;                   // compute mcxkernel
    cl_int status = 0;
    cl_device_id devices[MAX_DEVICE];
    cl_event* waittoread;
    cl_platform_id platform = NULL;

    cl_uint  totalcucore;
    cl_uint  devid = 0;
    cl_mem* gmedia = NULL, *gproperty = NULL, *gparam = NULL;
    cl_mem greplaydetid = NULL, greplayw = NULL, greplaytof = NULL, *gsrcpattern = NULL;
    cl_mem* gfield = NULL, *gdetphoton, *gseed = NULL, *genergy = NULL, *gseeddata = NULL;
    cl_mem* gprogress = NULL, *gdetected = NULL, *gdetpos = NULL, *gjumpdebug = NULL, *gdebugdata = NULL, *ginvcdf = NULL, *gangleinvcdf = NULL;

    cl_uint dimxyz = cfg->dim.x * cfg->dim.y * cfg->dim.z * ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) ? cfg->srcnum : 1);

    cl_uint*  media = (cl_uint*)(cfg->vol);
    cl_float*  field;

    float*  Pdet = NULL;
    float*  srcpw = NULL, *energytot = NULL, *energyabs = NULL; // for multi-srcpattern
    char opt[MAX_PATH_LENGTH << 1] = {'\0'};
    GPUInfo* gpu = NULL;
    RandType* seeddata = NULL;
    RandType* Pseed = NULL;

    /*
                                  |----------------------------------------------->  hostdetreclen  <--------------------------------------|
                                            |------------------------>    partialdata   <-------------------|
     host detected photon buffer: detid (1), partial_scat (#media), partial_path (#media), momontum (#media), p_exit (3), v_exit(3), w0 (1)
                                            |--------------------------------------------->    w0offset   <-------------------------------------||<----- w0 (#srcnum) ----->|
      gpu detected photon buffer:            partial_scat (#media), partial_path (#media), momontum (#media), E_escape (1), E_launch (1), w0 (1), w0_photonsharing (#srcnum)
    */
    unsigned int partialdata = (cfg->medianum - 1) * (SAVE_NSCAT(cfg->savedetflag) + SAVE_PPATH(cfg->savedetflag) + SAVE_MOM(cfg->savedetflag)); // buf len for media-specific data, copy from gpu to host
    unsigned int w0offset = partialdata + 3; // offset for photon sharing buffer
    unsigned int hostdetreclen = partialdata + SAVE_DETID(cfg->savedetflag) + 3 * (SAVE_PEXIT(cfg->savedetflag) + SAVE_VEXIT(cfg->savedetflag)) + SAVE_W0(cfg->savedetflag); // host-side det photon data buffer length
    unsigned int is2d = (cfg->dim.x == 1 ? 1 : (cfg->dim.y == 1 ? 2 : (cfg->dim.z == 1 ? 3 : 0)));

    MCXParam param = {{{cfg->srcpos.x, cfg->srcpos.y, cfg->srcpos.z, cfg->srcpos.w}},
        {{cfg->srcdir.x, cfg->srcdir.y, cfg->srcdir.z, cfg->srcdir.w}},
        {{(float)cfg->dim.x, (float)cfg->dim.y, (float)cfg->dim.z, 0}}, dimlen, cp0, cp1, cachebox,
        minstep, 0.f, 0.f, cfg->tend, R_C0* cfg->unitinmm, (uint)cfg->isrowmajor,
        (uint)cfg->issave2pt, (uint)cfg->isreflect, (uint)cfg->isrefint,
        (uint)cfg->issavedet, 1.f / cfg->tstep, cfg->minenergy,
        cfg->sradius* cfg->sradius, minstep* R_C0* cfg->unitinmm, cfg->maxdetphoton,
        cfg->medianum - 1, cfg->detnum, 0, 0, 0, 0, (uint)cfg->voidtime, (uint)cfg->srctype,
        {{cfg->srcparam1.x, cfg->srcparam1.y, cfg->srcparam1.z, cfg->srcparam1.w}},
        {{cfg->srcparam2.x, cfg->srcparam2.y, cfg->srcparam2.z, cfg->srcparam2.w}},
        (uint)cfg->maxvoidstep, cfg->issaveexit > 0, cfg->issaveseed > 0, (uint)cfg->issaveref,
        cfg->isspecular > 0, cfg->maxgate, cfg->seed, (uint)cfg->outputtype, 0, 0,
        (uint)cfg->debuglevel, cfg->savedetflag, hostdetreclen, partialdata, w0offset, (uint)cfg->mediabyte,
        (uint)cfg->maxjumpdebug, cfg->gscatter, is2d, cfg->replaydet, cfg->srcnum, cfg->nphase,
        cfg->nphase + (cfg->nphase & 0x1), cfg->nangle, cfg->nangle + (cfg->nangle & 0x1)
    };

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
    OCL_ASSERT(((mcxcontext = clCreateContext(cprops, workdev, devices, NULL, NULL, &status), status)));

    mcxqueue = (cl_command_queue*)malloc(workdev * sizeof(cl_command_queue));
    waittoread = (cl_event*)malloc(workdev * sizeof(cl_event));

    gseed = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gmedia = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gproperty = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gparam = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gfield = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gdetphoton = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    genergy = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gprogress = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gdetected = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gdetpos = (cl_mem*)calloc(workdev, sizeof(cl_mem));
    gjumpdebug = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gdebugdata = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gseeddata = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gsrcpattern = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    ginvcdf = (cl_mem*)malloc(workdev * sizeof(cl_mem));
    gangleinvcdf = (cl_mem*)malloc(workdev * sizeof(cl_mem));

    /* The block is to move the declaration of prop closer to its use */
    cl_command_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;

    totalcucore = 0;

    for (i = 0; i < workdev; i++) {
        OCL_ASSERT(((mcxqueue[i] = clCreateCommandQueue(mcxcontext, devices[i], prop, &status), status)));
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
            if (gpu[i].vendor == dvIntelGPU) { // Intel HD graphics GPU
                gpu[i].autoblock  = 64;
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

    if (is2d) {
        float* vec = &(param.c0.x);

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

    if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
        fieldlen = dimxyz * cfg->maxgate * cfg->detnum;
    } else {
        fieldlen = dimxyz * cfg->maxgate;
    }

    cachebox.x = (cp1.x - cp0.x + 1);
    cachebox.y = (cp1.y - cp0.y + 1) * (cp1.x - cp0.x + 1);
    dimlen.x = cfg->dim.x;
    dimlen.y = cfg->dim.x * cfg->dim.y;
    dimlen.z = cfg->dim.x * cfg->dim.y * cfg->dim.z;
    dimlen.w = fieldlen;

    memcpy(&(param.dimlen.x), &(dimlen.x), sizeof(uint4));
    memcpy(&(param.cachebox.x), &(cachebox.x), sizeof(uint2));

    if (param.ps.x < 0.f || param.ps.y < 0.f || param.ps.z < 0.f || param.ps.x >= cfg->dim.x || param.ps.y >= cfg->dim.y || param.ps.z >= cfg->dim.z) {
        param.idx1dorig = 0;
        param.mediaidorig = 0;
    } else {
        param.idx1dorig = (int(floorf(param.ps.z)) * dimlen.y + int(floorf(param.ps.y)) * dimlen.x + int(floorf(param.ps.x)));
        param.mediaidorig = (cfg->vol[param.idx1dorig] & MED_MASK);
    }

    if (cfg->seed > 0) {
        srand(cfg->seed);
    } else {
        srand(time(0));
    }

    if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY) && cfg->exportdebugdata == NULL) {
        cfg->exportdebugdata = (float*)calloc(sizeof(float), (debuglen * cfg->maxjumpdebug));
    }

    cl_mem (*clCreateBufferNV)(cl_context, cl_mem_flags, cl_mem_flags_NV, size_t, void*, cl_int*) = (cl_mem (*)(cl_context, cl_mem_flags, cl_mem_flags_NV, size_t, void*, cl_int*)) clGetExtensionFunctionAddressForPlatform(platform, "clCreateBufferNV");

    if (clCreateBufferNV == NULL) {
        OCL_ASSERT(((gprogress[0] = clCreateBuffer(mcxcontext, RW_PTR, sizeof(cl_uint), NULL, &status), status)));
    } else {
        gprogress[0] = clCreateBufferNV(mcxcontext, CL_MEM_READ_WRITE, NV_PIN, sizeof(cl_uint), NULL, &status);

        if (status == CL_INVALID_VALUE) {
            MCX_FPRINTF(cfg->flog, "Warning: to use the progress bar feature, you need to upgrade your NVIDIA GPU driver to 399.x or newer. Without the update, your progress bar may appear to be static (despite the simulation is running)\n");
            OCL_ASSERT(((gprogress[0] = clCreateBuffer(mcxcontext, RW_PTR, sizeof(cl_uint), NULL, &status), status)));
        }
    }

    progress = (cl_uint*)clEnqueueMapBuffer(mcxqueue[0], gprogress[0], CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_uint), 0, NULL, NULL, NULL);
    *progress = 0;

    if (cfg->seed == SEED_FROM_FILE) {
        // replay should only work with a single device
        OCL_ASSERT(((gseed[0] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(RandType) * cfg->nphoton * RAND_BUF_LEN, cfg->replay.seed, &status), status)));

        if (cfg->replay.weight) {
            OCL_ASSERT(((greplayw = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nphoton, cfg->replay.weight, &status), status)));
        }

        if (cfg->replay.tof) {
            OCL_ASSERT(((greplaytof = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nphoton, cfg->replay.tof, &status), status)));
        }

        if (cfg->replay.detid) {
            OCL_ASSERT(((greplaydetid = clCreateBuffer(mcxcontext, RO_MEM, sizeof(int) * cfg->nphoton, cfg->replay.detid, &status), status)));
        }
    }

    for (i = 0; i < workdev; i++) {
        if (cfg->mediabyte != MEDIA_2LABEL_SPLIT) {
            OCL_ASSERT(((gmedia[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(cl_uint) * (cfg->dim.x * cfg->dim.y * cfg->dim.z), media, &status), status)));
        } else {
            OCL_ASSERT(((gmedia[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(cl_uint) * (2 * cfg->dim.x * cfg->dim.y * cfg->dim.z), media, &status), status)));
        }

        OCL_ASSERT(((gproperty[i] = clCreateBuffer(mcxcontext, RO_MEM, cfg->medianum * sizeof(Medium), cfg->prop, &status), status)));
        OCL_ASSERT(((gparam[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(MCXParam), &param, &status), status)));
        energy = (cl_float*)calloc(sizeof(cl_float), gpu[i].autothread << 1);

        if (cfg->seed != SEED_FROM_FILE) {
            Pseed = (RandType*)malloc(sizeof(RandType) * gpu[i].autothread * RAND_BUF_LEN);
            cl_uint* iseed = (cl_uint*)Pseed;

            for (j = 0; j < gpu[i].autothread * RAND_SEED_LEN; j++) {
                iseed[j] = rand();
            }

            OCL_ASSERT(((gseed[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(RandType) * gpu[i].autothread * RAND_BUF_LEN, Pseed, &status), status)));
        }

        OCL_ASSERT(((gfield[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(cl_float) * fieldlen * 2, field, &status), status)));

        if (cfg->issavedet) {
            OCL_ASSERT(((gdetphoton[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(float) * cfg->maxdetphoton * hostdetreclen, Pdet, &status), status)));
        }

        OCL_ASSERT(((genergy[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(float) * (gpu[i].autothread << 1), energy, &status), status)));
        OCL_ASSERT(((gdetected[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(cl_uint), &detected, &status), status)));

        if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
            uint jumpcount = 0;
            OCL_ASSERT(((gjumpdebug[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(cl_uint), &jumpcount, &status), status)));
            OCL_ASSERT(((gdebugdata[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(float) * (debuglen * cfg->maxjumpdebug), cfg->exportdebugdata, &status), status)));
        }

        if (cfg->issaveseed) {
            seeddata = (RandType*)calloc(sizeof(RandType), cfg->maxdetphoton * RAND_BUF_LEN);
            OCL_ASSERT(((gseeddata[i] = clCreateBuffer(mcxcontext, RW_MEM, sizeof(RandType) * cfg->maxdetphoton * RAND_BUF_LEN, seeddata, &status), status)));
            free(seeddata);
        }

        if (cfg->nphase) {
            OCL_ASSERT(((ginvcdf[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nphase, cfg->invcdf, &status), status)));
        }

        if (cfg->nangle) {
            OCL_ASSERT(((gangleinvcdf[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * cfg->nangle, cfg->angleinvcdf, &status), status)));
        }

        if (cfg->detnum > 0) {
            OCL_ASSERT(((gdetpos[i] = clCreateBuffer(mcxcontext, RO_MEM, cfg->detnum * sizeof(float4), cfg->detpos, &status), status)));
        }

        if (cfg->seed != SEED_FROM_FILE) {
            free(Pseed);
        }

        free(energy);

        if (cfg->srctype == MCX_SRC_PATTERN) {
            OCL_ASSERT(((gsrcpattern[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * (int)(cfg->srcparam1.w * cfg->srcparam2.w) * cfg->srcnum, cfg->srcpattern, &status), status)));
        } else if (cfg->srctype == MCX_SRC_PATTERN3D) {
            OCL_ASSERT(((gsrcpattern[i] = clCreateBuffer(mcxcontext, RO_MEM, sizeof(float) * (int)(cfg->srcparam1.x * cfg->srcparam1.y * cfg->srcparam1.z) * cfg->srcnum, cfg->srcpattern, &status), status)));
        } else {
            gsrcpattern[i] = NULL;
        }

    }

    mcx_printheader(cfg);

    tic = StartTimer();

#if __OPENCL_C_VERSION__
    MCX_FPRINTF(cfg->flog, "- code name: [Fractal] compiled with OpenCL [%d] on [%s]\n",
                __OPENCL_C_VERSION__, __DATE__);
#else
    MCX_FPRINTF(cfg->flog, "- code name: [Fractal] compiled with OpenCL [%d] on [%s]\n",
                CL_VERSION_1_0, __DATE__);
#endif

    MCX_FPRINTF(cfg->flog, "- compiled with: [RNG] %s [Seed Length] %d\n", MCX_RNG_NAME, RAND_SEED_LEN);
    MCX_FPRINTF(cfg->flog, "initializing streams ...\t");

    MCX_FPRINTF(cfg->flog, "init complete : %d ms\n", GetTimeMillis() - tic);
    fflush(cfg->flog);
    mcx_flush(cfg);

    OCL_ASSERT(((mcxprogram = clCreateProgramWithSource(mcxcontext, 1, (const char**) & (cfg->clsource), NULL, &status), status)));

    if (cfg->optlevel >= 1) {
        sprintf(opt, "%s ", "-cl-mad-enable -DMCX_USE_NATIVE");
    }

    if (cfg->optlevel >= 2) {
        sprintf(opt + strlen(opt), "%s ", "-DMCX_SIMPLIFY_BRANCH -DMCX_VECTOR_INDEX");
    }

    if (cfg->optlevel >= 3) {
        sprintf(opt + strlen(opt), "%s ", "-DGROUP_LOAD_BALANCE");
    }

    if (cfg->optlevel >= 4 && cfg->seed != SEED_FROM_FILE) { // optlevel 2 - i.e. defining parameters as macros - fails replay test, need to debug
        sprintf(opt + strlen(opt), "%s ", "-DUSE_MACRO_CONST");
    }

    for (i = 0; i < 3; i++)
        if (cfg->debuglevel & (1 << i)) {
            sprintf(opt + strlen(opt), "%s ", debugopt[i]);
        }

    if ((uint)cfg->srctype < sizeof(sourceflag) / sizeof(sourceflag[0])) {
        sprintf(opt + strlen(opt), "%s ", sourceflag[(uint)cfg->srctype]);
    }

    sprintf(opt + strlen(opt), "-DMED_TYPE=%d ", cfg->mediabyte);

    sprintf(opt + strlen(opt), "%s ", cfg->compileropt);

    if (cfg->isatomic) {
        sprintf(opt + strlen(opt), "%s ", "-DUSE_ATOMIC");
    }

    if (cfg->issavedet) {
        sprintf(opt + strlen(opt), "%s ", "-DMCX_SAVE_DETECTORS");
    }

    if (strstr(opt, "USE_MACRO_CONST")) {
        IPARAM_TO_MACRO(opt, param, detnum);
        IPARAM_TO_MACRO(opt, param, doreflect);
        IPARAM_TO_MACRO(opt, param, gscatter);
        IPARAM_TO_MACRO(opt, param, idx1dorig);
        IPARAM_TO_MACRO(opt, param, is2d);
        IPARAM_TO_MACRO(opt, param, issaveref);
        IPARAM_TO_MACRO(opt, param, issaveseed);
        IPARAM_TO_MACRO(opt, param, isspecular);
        IPARAM_TO_MACRO(opt, param, maxdetphoton);
        IPARAM_TO_MACRO(opt, param, maxgate);
        IPARAM_TO_MACRO(opt, param, maxjumpdebug);
        IPARAM_TO_MACRO(opt, param, maxmedia);
        IPARAM_TO_MACRO(opt, param, maxvoidstep);
        IPARAM_TO_MACRO(opt, param, mediaformat);
        IPARAM_TO_MACRO(opt, param, mediaidorig);
        FPARAM_TO_MACRO(opt, param, minaccumtime);
        FPARAM_TO_MACRO(opt, param, minenergy);
        FPARAM_TO_MACRO(opt, param, oneoverc0);
        IPARAM_TO_MACRO(opt, param, outputtype);
        IPARAM_TO_MACRO(opt, param, partialdata);
        IPARAM_TO_MACRO(opt, param, reclen);
        IPARAM_TO_MACRO(opt, param, replaydet);
        IPARAM_TO_MACRO(opt, param, save2pt);
        IPARAM_TO_MACRO(opt, param, savedet);
        IPARAM_TO_MACRO(opt, param, savedetflag);
        IPARAM_TO_MACRO(opt, param, seed);
        IPARAM_TO_MACRO(opt, param, srcnum);
        IPARAM_TO_MACRO(opt, param, voidtime);
        IPARAM_TO_MACRO(opt, param, w0offset);
        FPARAM_TO_MACRO(opt, param, Rtstep);
    }

    char allabsorb[] = {bcAbsorb, bcAbsorb, bcAbsorb, bcAbsorb, bcAbsorb, bcAbsorb, 0};
    char allunknown[] = {0, 0, 0, 0, 0, 0, 0, 0};

    if (cfg->isreflect || (strcmp(cfg->bc, allabsorb) && strcmp(cfg->bc, allunknown))) {
        sprintf(opt + strlen(opt), " -DMCX_DO_REFLECTION");
    } else {
        /** Enable reflection flag when c or m flags are used in the cfg.bc boundary condition flags */
        for (i = 0; i < 6; i++)
            if (cfg->bc[i] == bcReflect || cfg->bc[i] == bcMirror) {
                sprintf(opt + strlen(opt), " -DMCX_DO_REFLECTION");
            }
    }

    if (cfg->internalsrc || (param.mediaidorig && (cfg->srctype == MCX_SRC_PENCIL || cfg->srctype == MCX_SRC_CONE || cfg->srctype == MCX_SRC_ISOTROPIC))) {
        sprintf(opt + strlen(opt), " -DINTERNAL_SOURCE");
    }

    if (workdev == 1 && gpu[0].iscpu) {
        sprintf(opt + strlen(opt), " -DMCX_USE_CPU");
    }

    if (gpu[0].vendor == dvNVIDIA) {
        sprintf(opt + strlen(opt), " -DUSE_NVIDIA_GPU");
    }

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

        for (int i = 0; i < (int)len; i++)
            if (msg[i] <= 'z' && msg[i] >= 'A') {
                MCX_FPRINTF(cfg->flog, "Kernel build log:\n%s\n", msg);
                break;
            }

        delete [] msg;
    }

    if (status != CL_SUCCESS) {
        mcx_error(-(int)status, (char*)("Error: Failed to build program executable!"), __FILE__, __LINE__);
    }

    MCX_FPRINTF(cfg->flog, "build program complete : %d ms\n", GetTimeMillis() - tic);
    fflush(cfg->flog);

    mcxkernel = (cl_kernel*)malloc(workdev * sizeof(cl_kernel));

    for (i = 0; i < workdev; i++) {
        cl_int threadphoton, oddphoton, sharedbuf;

        threadphoton = (int)(cfg->nphoton * cfg->workload[i] / (fullload * gpu[i].autothread * cfg->respin));
        oddphoton = (int)(cfg->nphoton * cfg->workload[i] / (fullload * cfg->respin) - threadphoton * gpu[i].autothread);
        sharedbuf = (param.nphaselen + param.nanglelen) * sizeof(float) + gpu[i].autoblock * (cfg->issaveseed * (RAND_BUF_LEN * sizeof(RandType)) + sizeof(float) * (param.w0offset + cfg->srcnum));

        MCX_FPRINTF(cfg->flog, "- [device %d(%d): %s] threadph=%d extra=%d np=%.0f nthread=%d nblock=%d sharedbuf=%d\n", i, gpu[i].id, gpu[i].name, threadphoton, oddphoton,
                    cfg->nphoton * cfg->workload[i] / fullload, (int)gpu[i].autothread, (int)gpu[i].autoblock, sharedbuf);

        OCL_ASSERT(((mcxkernel[i] = clCreateKernel(mcxprogram, "mcx_main_loop", &status), status)));
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
    }

    MCX_FPRINTF(cfg->flog, "set kernel arguments complete : %d ms\n", GetTimeMillis() - tic);
    fflush(cfg->flog);

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

    //simulate for all time-gates in maxgate groups per run

    cl_float Vvox;
    Vvox = cfg->steps.x * cfg->steps.y * cfg->steps.z;
    memcpy(&(param.bc), cfg->bc, 12);

    tic0 = GetTimeMillis();

    for (t = cfg->tstart; t < cfg->tend; t += cfg->tstep * cfg->maxgate) {
        twindow0 = t;
        twindow1 = t + cfg->tstep * cfg->maxgate;

        MCX_FPRINTF(cfg->flog, "lauching mcx_main_loop for time window [%.1fns %.1fns] ...\n"
                    , twindow0 * 1e9, twindow1 * 1e9);
        fflush(cfg->flog);

        //total number of repetition for the simulations, results will be accumulated to field
        for (iter = 0; iter < cfg->respin; iter++) {
            MCX_FPRINTF(cfg->flog, "simulation run#%2d ... \n", iter + 1);
            fflush(cfg->flog);
            fflush(cfg->flog);
            mcx_flush(cfg);

            param.twin0 = twindow0;
            param.twin1 = twindow1;

            for (devid = 0; devid < workdev; devid++) {
                int nblock = gpu[devid].autothread / gpu[devid].autoblock;

                param.threadphoton = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * gpu[devid].autothread * cfg->respin));
                param.oddphoton   = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * cfg->respin) - param.threadphoton * gpu[devid].autothread);
                param.blockphoton = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * nblock * cfg->respin));
                param.blockextra  = (int)(cfg->nphoton * cfg->workload[devid] / (fullload * cfg->respin) - param.blockphoton * nblock);
                OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid], gparam[devid], CL_TRUE, 0, sizeof(MCXParam), &param, 0, NULL, NULL)));
                OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 17, sizeof(cl_mem), (void*)(gparam + devid))));

                // launch mcxkernel
                OCL_ASSERT((clEnqueueNDRangeKernel(mcxqueue[devid], mcxkernel[devid], 1, NULL, &gpu[devid].autothread, &gpu[devid].autoblock, 0, NULL, &waittoread[devid])));
                OCL_ASSERT((clFlush(mcxqueue[devid])));
            }

            if ((param.debuglevel & MCX_DEBUG_PROGRESS)) {
                int p0 = 0, ndone = -1, kernelstatus = 0;
                int threadphoton = (int)(cfg->nphoton * cfg->workload[0] / (fullload * gpu[0].autothread * cfg->respin));
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

            clEnqueueUnmapMemObject(mcxqueue[0], gprogress[0], progress, 0, NULL, NULL);

            //clWaitForEvents(workdev,waittoread);
            for (devid = 0; devid < workdev; devid++) {
                OCL_ASSERT((clFinish(mcxqueue[devid])));
            }

            tic1 = GetTimeMillis();
            toc += tic1 - tic0;
            MCX_FPRINTF(cfg->flog, "kernel complete:  \t%d ms\nretrieving flux ... \t", tic1 - tic);
            fflush(cfg->flog);

            for (devid = 0; devid < workdev; devid++) {
                if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
                    uint debugrec = 0;
                    OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gjumpdebug[devid], CL_TRUE, 0, sizeof(uint),
                                                    &debugrec, 0, NULL, waittoread + devid)));

                    if (debugrec > 0) {
                        if (debugrec > cfg->maxdetphoton) {
                            MCX_FPRINTF(cfg->flog, S_RED "WARNING: the saved trajectory positions (%d) \
  are more than what your have specified (%d), please use the --maxjumpdebug option to specify a greater number\n" S_RESET
                                        , debugrec, cfg->maxjumpdebug);
                        } else {
                            MCX_FPRINTF(cfg->flog, "saved %u trajectory positions, total: %d\t", debugrec, cfg->maxjumpdebug + debugrec);
                        }

                        debugrec = MIN(debugrec, cfg->maxjumpdebug);
                        cfg->exportdebugdata = (float*)realloc(cfg->exportdebugdata, (cfg->debugdatalen + debugrec) * debuglen * sizeof(float));
                        OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gdebugdata[devid], CL_FALSE, 0, sizeof(float)*debuglen * debugrec,
                                                        cfg->exportdebugdata + cfg->debugdatalen, 0, NULL, waittoread + devid)));
                        cfg->debugdatalen += debugrec;
                    }
                }

                if (cfg->issavedet) {
                    OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gdetected[devid], CL_FALSE, 0, sizeof(uint),
                                                    &detected, 0, NULL, waittoread + devid)));
                    OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gdetphoton[devid], CL_TRUE, 0, sizeof(float)*cfg->maxdetphoton * hostdetreclen,
                                                    Pdet, 0, NULL, NULL)));

                    if (cfg->issaveseed) {
                        seeddata = (RandType*)calloc(sizeof(RandType), cfg->maxdetphoton * RAND_BUF_LEN);
                        OCL_ASSERT(clEnqueueReadBuffer(mcxqueue[devid], gseeddata[devid], CL_TRUE, 0,
                                                       sizeof(RandType)*cfg->maxdetphoton * RAND_BUF_LEN, seeddata, 0, NULL, NULL));
                    }

                    if (detected > cfg->maxdetphoton) {
                        MCX_FPRINTF(cfg->flog, S_RED "WARNING: the detected photon (%d) \
is more than what your have specified (%d), please use the -H option to specify a greater number\t" S_RESET
                                    , detected, cfg->maxdetphoton);
                    } else {
                        MCX_FPRINTF(cfg->flog, "detected " S_BOLD "" S_BLUE "%d photons" S_RESET", total: " S_BOLD "" S_BLUE "%d" S_RESET"\t", detected, cfg->detectedcount + detected);
                    }

                    cfg->his.detected += detected;
                    detected = MIN(detected, cfg->maxdetphoton);

                    if (cfg->exportdetected) {
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
                    float* rawfield = (float*)malloc(sizeof(float) * fieldlen * 2);

                    OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid], gfield[devid], CL_TRUE, 0, sizeof(cl_float)*fieldlen * 2,
                                                    rawfield, 0, NULL, NULL)));
                    MCX_FPRINTF(cfg->flog, "transfer complete:        %d ms\n", GetTimeMillis() - tic);
                    fflush(cfg->flog);

                    if (!(param.debuglevel & MCX_DEBUG_RNG)) {
                        for (i = 0; i < fieldlen; i++) { //accumulate field, can be done in the GPU
                            field[i] = rawfield[i] + rawfield[i + fieldlen];
                        }
                    } else {
                        memcpy(field, rawfield, sizeof(cl_float)*fieldlen);
                    }

                    free(rawfield);

                    if (cfg->respin > 1) {
                        for (i = 0; i < fieldlen; i++) { //accumulate field, can be done in the GPU
                            field[fieldlen + i] += field[i];
                        }
                    }

                    if (iter + 1 == cfg->respin) {
                        if (cfg->respin > 1) { //copy the accumulated fields back
                            memcpy(field, field + fieldlen, sizeof(cl_float)*fieldlen);
                        }
                    }

                    if (cfg->exportfield) {
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

                //initialize the next simulation
                if (twindow1 < cfg->tend && iter < cfg->respin) {
                    memset(field, 0, sizeof(cl_float)*dimxyz * cfg->maxgate);
                    OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid], gfield[devid], CL_TRUE, 0, sizeof(cl_float)*dimxyz * cfg->maxgate,
                                                     field, 0, NULL, NULL)));
                    OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 1, sizeof(cl_mem), (void*)(gfield + devid))));
                }

                if (cfg->respin > 1 && RAND_SEED_LEN > 1 && cfg->seed != SEED_FROM_FILE) {
                    Pseed = (RandType*)malloc(sizeof(RandType) * gpu[devid].autothread * RAND_BUF_LEN);
                    cl_uint* iseed = (cl_uint*)Pseed;

                    for (j = 0; j < gpu[devid].autothread * RAND_SEED_LEN; j++) {
                        iseed[j] = rand();
                    }

                    OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid], gseed[devid], CL_TRUE, 0, sizeof(RandType)*gpu[devid].autothread * RAND_BUF_LEN,
                                                     Pseed, 0, NULL, NULL)));
                    OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 3, sizeof(cl_mem), (void*)(gseed + devid))));
                    free(Pseed);
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

    if (cfg->issave2pt && cfg->srctype == MCX_SRC_PATTERN && cfg->srcnum > 1) { // post-processing only for multi-srcpattern
        srcpw = (float*)calloc(cfg->srcnum, sizeof(float));
        energytot = (float*)calloc(cfg->srcnum, sizeof(float));
        energyabs = (float*)calloc(cfg->srcnum, sizeof(float));
        uint psize = (int)cfg->srcparam1.w * (int)cfg->srcparam2.w;

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
        MCX_FPRINTF(cfg->flog, "normalizing raw data ...\t");
        cfg->energyabs += cfg->energytot - cfg->energyesc;

        if (cfg->outputtype == otFlux || cfg->outputtype == otFluence) {
            scale[0] = cfg->unitinmm / (cfg->energytot * Vvox * cfg->tstep); /* Vvox (in mm^3 already) * (Tstep) * (Eabsorp/U) */

            if (cfg->outputtype == otFluence) {
                scale[0] *= cfg->tstep;
            }
        } else if (cfg->outputtype == otEnergy || cfg->outputtype == otL) {
            scale[0] = 1.f / cfg->energytot;
        } else if (cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS) {
            if (cfg->seed == SEED_FROM_FILE && cfg->replaydet == -1) {
                int detid;

                for (detid = 1; detid <= (int)cfg->detnum; detid++) {
                    scale[0] = 0.f; // the cfg->normalizer and cfg.his.normalizer are inaccurate in this case, but this is ok

                    for (size_t i = 0; i < cfg->nphoton; i++)
                        if (cfg->replay.detid[i] == detid) {
                            scale[0] += cfg->replay.weight[i];
                        }

                    if (scale[0] > 0.f) {
                        scale[0] = cfg->unitinmm / scale[0];
                    }

                    MCX_FPRINTF(cfg->flog, "normalization factor for detector %d alpha=%f\n", detid, scale[0]);
                    fflush(cfg->flog);
                    mcx_normalize(cfg->exportfield + (detid - 1)*dimxyz * param.maxgate, scale[0], dimxyz * param.maxgate, cfg->isnormalized, 0, 1);
                }

                isnormalized = 1;
            } else {
                scale[0] = 0.f;

                for (size_t i = 0; i < cfg->nphoton; i++) {
                    scale[0] += cfg->replay.weight[i];
                }

                if (scale[0] > 0.f) {
                    scale[0] = cfg->unitinmm / scale[0];
                }
            }
        }

        if (cfg->srctype == MCX_SRC_PATTERN && cfg->srcnum > 1) { // post-processing only for multi-srcpattern
            float scaleref = scale[0];
            int psize = (int)cfg->srcparam1.w * (int)cfg->srcparam2.w;

            for (i = 0; i < cfg->srcnum; i++) {
                scale[i] = psize / srcpw[i] * scaleref;
            }
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

#ifndef MCX_CONTAINER

    if (cfg->issave2pt && cfg->parentid == mpStandalone) {
        MCX_FPRINTF(cfg->flog, "saving data to file ... %ld %d\t", fieldlen, cfg->maxgate);
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

    // total energy here equals total simulated photons+unfinished photons for all threads
    MCX_FPRINTF(cfg->flog, "simulated %ld photons (%ld) with %d devices (repeat x%d)\nMCX simulation speed: " S_BOLD "" S_BLUE "%.2f photon/ms" S_RESET"\n",
                cfg->nphoton, cfg->nphoton, workdev, cfg->respin, ((cfg->issavedet == FILL_MAXDETPHOTON) ? cfg->energytot : ((double)cfg->nphoton * ((cfg->respin > 1) ? (cfg->respin) : 1))) / MAX(1, cfg->runtime));

    if (cfg->srctype == MCX_SRC_PATTERN && cfg->srcnum > 1) {
        for (i = 0; i < cfg->srcnum; i++) {
            MCX_FPRINTF(cfg->flog, "source #%d total simulated energy: %.2f\tabsorbed: " S_BOLD "" S_BLUE "%5.5f%%" S_RESET"\n(loss due to initial specular reflection is excluded in the total)\n",
                        i + 1, energytot[i], energyabs[i] / energytot[i] * 100.f);
            fflush(cfg->flog);
        }
    } else {
        MCX_FPRINTF(cfg->flog, "total simulated energy: %.2f\tabsorbed: " S_BOLD "" S_BLUE "%5.5f%%" S_RESET"\n(loss due to initial specular reflection is excluded in the total)\n",
                    cfg->energytot, (cfg->energytot - cfg->energyesc) / cfg->energytot * 100.f);
        fflush(cfg->flog);
    }

    clReleaseMemObject(gprogress[0]);

    if (cfg->seed == SEED_FROM_FILE) {
        clReleaseMemObject(gseed[0]);

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
        clReleaseMemObject(gmedia[i]);
        clReleaseMemObject(gproperty[i]);
        clReleaseMemObject(gparam[i]);
        clReleaseMemObject(gfield[i]);

        if (cfg->seed != SEED_FROM_FILE) {
            clReleaseMemObject(gseed[i]);
        }

        if (cfg->issavedet) {
            clReleaseMemObject(gdetphoton[i]);
        }

        clReleaseMemObject(genergy[i]);
        clReleaseMemObject(gdetected[i]);

        if (cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) {
            clReleaseMemObject(gsrcpattern[i]);
        }

        if (cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
            clReleaseMemObject(gjumpdebug[i]);
            clReleaseMemObject(gdebugdata[i]);
        }

        if (cfg->detpos) {
            clReleaseMemObject(gdetpos[i]);
        }

        if (cfg->issaveseed) {
            clReleaseMemObject(gseeddata[i]);
        }

        if (cfg->nphase) {
            clReleaseMemObject(ginvcdf[i]);
        }

        if (cfg->nangle) {
            clReleaseMemObject(gangleinvcdf[i]);
        }

        clReleaseKernel(mcxkernel[i]);
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
    free(mcxkernel);
    free(waittoread);
    free(Pdet);

    if (gpu) {
        free(gpu);
    }

    for (devid = 0; devid < workdev; devid++) {
        clReleaseCommandQueue(mcxqueue[devid]);
    }

    free(mcxqueue);
    clReleaseProgram(mcxprogram);
    clReleaseContext(mcxcontext);
    clReleaseEvent(kernelevent);

    free(field);
    free(srcpw);
    free(energytot);
    free(energyabs);
}
