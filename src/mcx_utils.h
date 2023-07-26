/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2022
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
\file    mcx_utils.h

@brief   MCX configuration header
*******************************************************************************/

#ifndef _MCEXTREME_UTILITIES_H
#define _MCEXTREME_UTILITIES_H

#include <stdio.h>

#include "vector_types.h"
#include "cjson/cJSON.h"
#include "nifti1.h"

#ifdef _OPENMP                      ///< use multi-threading for running simulation on multiple GPUs
    #include <omp.h>
#endif

#define MAX_FULL_PATH       2048
#define MAX_PATH_LENGTH     1024
#define MAX_SESSION_LENGTH  64
#define MAX_DEVICE          256

#define MCX_ASSERT(x)  mcx_assess((x),"assert error",__FILE__,__LINE__)
#define MCX_ERROR(id,msg)   mcx_error(id,msg,__FILE__,__LINE__)
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MAX(a,b)           ((a)>(b)?(a):(b))

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352
#define GL_RGBA32F 0x8814

enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otWP, otDCS, otL};   /**< types of output */
enum TMCXParent  {mpStandalone, mpMATLAB, mpPython};                   /**< whether MCX is run in binary or mex mode */
enum TOutputFormat {ofMC2, ofNifti, ofAnalyze, ofUBJSON, ofTX3, ofJNifti, ofBJNifti};           /**< output data format */
enum TDeviceVendor {dvUnknown, dvNVIDIA, dvAMD, dvIntel, dvIntelGPU, dvAppleCPU};
enum TBoundary {bcUnknown, bcReflect, bcAbsorb, bcMirror, bcCyclic};            /**< boundary conditions */
enum TBJData {JDB_mixed, JDB_nulltype, JDB_noop, JDB_true, JDB_false,
              JDB_char, JDB_string, JDB_hp, JDB_int8, JDB_uint8, JDB_int16, JDB_int32,
              JDB_int64, JDB_single, JDB_double, JDB_array, JDB_object, JDB_numtypes,
              JDB_uint16 = 10, JDB_uint32, JDB_uint64
             };

/**
 * The structure to store optical properties
 * Four relevant optical properties are needed
 */
typedef struct MCXMedium {
    float mua;                     /**< absorption coefficient (in 1/mm) */
    float mus;                     /**< scattering coefficient (in 1/mm) */
    float g;                       /**< anisotropy factor g: -1 to 1 */
    float n;                       /**< refractive index */
} Medium POST_ALIGN(16);  /*this order shall match prop.{xyzw} in mcx_main_loop*/

/**
 * Header data structure in .mch/.mct files to store detected photon data
 * This header has a total of 256 bytes
 */

typedef struct MCXHistoryHeader {
    char magic[4];                 /**< magic bits= 'M','C','X','H' */
    unsigned int  version;         /**< version of the mch file format */
    unsigned int  maxmedia;        /**< number of media in the simulation */
    unsigned int  detnum;          /**< number of detectors in the simulation */
    unsigned int  colcount;        /**< how many output files per detected photon */
    unsigned int  totalphoton;     /**< how many total photon simulated */
    unsigned int  detected;        /**< how many photons are detected (not necessarily all saved) */
    unsigned int  savedphoton;     /**< how many detected photons are saved in this file */
    float unitinmm;                /**< what is the voxel size of the simulation */
    unsigned int  seedbyte;        /**< how many bytes per RNG seed */
    float normalizer;              /**< what is the normalization factor */
    int respin;                    /**< if positive, repeat count so total photon=totalphoton*respin; if negative, total number is processed in respin subset */
    unsigned int  srcnum;          /**< number of sources for simultaneous pattern sources */
    unsigned int  savedetflag;     /**< number of sources for simultaneous pattern sources */
    int reserved[2];               /**< reserved fields for future extension */
} History;

/**
 * Data structure for photon replay
 */

typedef struct PhotonReplay {
    int*   detid;                 /**< pointer to the detector index */
    void*  seed;                  /**< pointer to the seeds of the replayed photon */
    float* weight;                /**< pointer to the detected photon weight array */
    float* tof;                   /**< pointer to the detected photon time-of-fly array */
} Replay;

typedef struct MCXGPUInfo {
    char name[MAX_SESSION_LENGTH];/**< name of the GPU */
    int id;                       /**< global index of the GPU, starting from 0 */
    int devcount;                 /**< total GPU count */
    int platformid;
    int major;                    /**< major version of the CUDA device */
    int minor;                    /**< minor version of the CUDA device */
    size_t globalmem;             /**< size of the global memory in the GPU */
    size_t constmem;              /**< size of the constant memory in the GPU */
    size_t sharedmem;             /**< size of the shared memory in the GPU */
    int regcount;                 /**< size of the register file in the GPU */
    int clock;                    /**< clock in Hz of the GPU processor */
    int sm;                       /**< number of multi processors */
    int core;                     /**< number of stream processors */
    size_t autoblock;             /**< optimized number of blocks to launch */
    size_t autothread;            /**< optimized number of threads to launch */
    int maxgate;                  /**< max number of time gates that can be saved in one call */
    int maxmpthread;              /**< maximum thread number per multi-processor */
    int iscpu;
    enum TDeviceVendor vendor;
} GPUInfo;

typedef struct MCXConfig {
    size_t nphoton;               /**<total simulated photon number*/
    unsigned int nblocksize;      /**<thread block size*/
    unsigned int nthread;         /**<num of total threads, multiple of 128*/
    int seed;                     /**<random number generator seed*/


    float4 srcpos;                /**<src position in mm*/
    float4 srcdir;                /**<src normal direction*/
    float tstart;                 /**<start time in second*/
    float tstep;                  /**<time step in second*/
    float tend;                   /**<end time in second*/
    float4 steps;                 /**<voxel sizes along x/y/z in mm*/

    uint4 dim;                    /**<domain size*/
    uint4 crop0;                  /**<sub-volume for cache*/
    uint4 crop1;                  /**<the other end of the caching box*/
    unsigned int medianum;        /**<total types of media*/
    unsigned int detnum;          /**<total detector numbers*/
    unsigned int maxdetphoton;    /**<anticipated maximum detected photons*/
    float detradius;              /**<default detector radius*/
    float sradius;                /**<source region radius, if set to non-zero, accumulation will not perform for dist<sradius*/

    Medium* prop;                 /**<optical property mapping table*/
    float4* detpos;               /**<detector positions and radius, overwrite detradius*/

    unsigned int maxgate;         /**<simultaneous recording gates*/
    unsigned int respin;          /**<number of repeatitions (if positive), or number of divisions (if negative)*/
    unsigned int printnum;        /**<number of printed threads (for debugging)*/
    int gpuid;                    /**<the ID of the GPU to use, starting from 1, 0 for auto*/

    unsigned int* vol;            /**<pointer to the volume*/
    char session[MAX_SESSION_LENGTH]; /**<session id, a string*/
    char isrowmajor;             /**<1 for C-styled array in vol, 0 for matlab-styled array*/
    char isreflect;              /**<1 for reflecting photons at boundary,0 for exiting*/
    char isref3;                 /**<1 considering maximum 3 ref. interfaces; 0 max 2 ref*/
    char isrefint;               /**<1 to consider reflections at internal boundaries; 0 do not*/
    char isnormalized;           /**<1 to normalize the fluence, 0 for raw fluence*/
    char issavedet;              /**<1 to count all photons hits the detectors*/
    char issave2pt;              /**<1 to save the 2-point distribution, 0 do not save*/
    char isgpuinfo;              /**<1 to print gpu info when attach, 0 do not print*/
    char isspecular;             /**<1 calculate the initial specular ref if outside the mesh, 0 do not calculate*/
    char isverbose;              /**<1 print debug info, 0 do not*/
    char issrcfrom0;             /**<1 do not subtract 1 from src/det positions, 0 subtract 1*/
    char ismomentum;             /**<1 to save momentum transfer for detected photons, implies issavedet=1*/
    char internalsrc;            /**<1 all photons launch positions are inside non-zero voxels, 0 let mcx search entry point*/
    char isdumpmask;             /**<1 dump detector mask; 0 not*/
    char issaveseed;             /**<1 save the seed for a detected photon, 0 do not save*/
    char issaveexit;             /**<1 save the exit position and dir of a detected photon, 0 do not save*/
    char isatomic;               /**<1 use atomic operations, 0 no atomic*/
    char issaveref;              /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
    char isdumpjson;             /**<1 to save json */
    int  zipid;                  /**<data zip method "zlib","gzip","base64","lzip","lzma","lz4","lz4hc"*/
    char srctype;                /**<0:pencil,1:isotropic,2:cone,3:gaussian,4:planar,5:pattern,\
                                         6:fourier,7:arcsine,8:disk,9:fourierx,10:fourierx2d,11:zgaussian,12:line,13:slit*/
    char autopilot;              /**<1 optimal setting for dedicated card, 2, for non dedicated card*/
    char outputtype;             /**<'X' output is flux, 'F' output is fluence, 'E' energy deposit*/
    char outputformat;           /**<'mc2' output is text, 'nii': binary, 'img': regular json, 'ubj': universal binary json*/
    float minenergy;             /**<minimum energy to propagate photon*/
    float unitinmm;              /**<defines the length unit in mm for grid*/
    FILE* flog;                  /**<stream handle to print log information*/
    History his;                 /**<header info of the history file*/
    double energytot;            /**<total launched photon packet weights*/
    double energyabs;            /**<total absorbed photon packet weights*/
    double energyesc;            /**<total escaped photon packet weights*/
    char rootpath[MAX_PATH_LENGTH]; /**<sets the input and output root folder*/
    char kernelfile[MAX_SESSION_LENGTH];
    char compileropt[MAX_PATH_LENGTH];
    char* shapedata;             /**<a pointer points to a string defining the JSON-formatted shape data*/
    char* extrajson;             /**<a pointer points to a string defining the extra JSON input data*/
    char* clsource;
    int maxvoidstep;             /**< max number of steps that a photon can advance before reaching a non-zero voxel*/
    int voidtime;                /**<1 start counting photon time when moves inside 0 voxels; 0: count time only after enters non-zero voxel*/
    float4 srcparam1;            /**<a quadruplet {x,y,z,w} for additional source parameters*/
    float4 srcparam2;            /**<a quadruplet {x,y,z,w} for additional source parameters*/
    unsigned int srcnum;         /**<total number of pattern sources */
    float* srcpattern;           /**<a string for the source form, options include "pencil","isotropic", etc*/
    char deviceid[MAX_DEVICE];
    float workload[MAX_DEVICE];
    float* exportfield;          /**<memory buffer when returning the flux to external programs such as matlab*/
    float* exportdetected;       /**<memory buffer when returning the partial length info to external programs such as matlab*/
    unsigned int debuglevel;     /**<a flag to control the printing of the debug information*/
    char faststep;               /**<1 use tMCimg-like approximated photon stepping (obsolete) */
    float normalizer;            /**<normalization factor*/
    Replay replay;               /**<a structure to prepare for photon replay*/
    void* seeddata;              /**<poiinter to a buffer where detected photon seeds are stored*/
    int replaydet;               /**<the detector id for which to replay the detected photons, start from 1*/
    char seedfile[MAX_PATH_LENGTH];/**<if the seed is specified as a file (mch), mcx will replay the photons*/
    char jsonfile[MAX_PATH_LENGTH];/**<if the seed is specified as a file (mch), mcx will replay the photons*/
    unsigned int maxjumpdebug;   /**<num of  photon scattering events to save when saving photon trajectory is enabled*/
    unsigned int debugdatalen;   /**<max number of photon trajectory position length*/
    unsigned int gscatter;       /**<after how many scattering events that we can use mus' instead of mus */
    float* exportdebugdata;      /**<pointer to the buffer where the photon trajectory data are stored*/
    unsigned int savedetflag;    /**<a flag to control the output fields of detected photon data*/
    unsigned int detectedcount;  /**<total number of detected photons*/
    unsigned int runtime;        /**<variable to store the total kernel simulation time in ms*/
    int parentid;                /**<flag for testing if mcx is executed inside matlab*/
    uint optlevel;               /**<OpenCL JIT compilation optimization level*/
    uint mediabyte;              /**< how many bytes per media index, mcx supports 1, 2 and 4, 4 is the default*/
    char bc[12];                 /**<boundary condition flag for [-x,-y,-z,+x,+y,+z, det(-x,-y,-z,+x,+y,+z)] */
} Config;

#ifdef  __cplusplus
extern "C" {
#endif
void mcx_savedata(float* dat, size_t len, Config* cfg);
void mcx_savenii(float* dat, size_t len, char* name, int type32bit, int outputformatid, Config* cfg);
void mcx_error(const int id, const char* msg, const char* file, const int linenum);
void mcx_assess(const int id, const char* msg, const char* file, const int linenum);
void mcx_cleargpuinfo(GPUInfo** gpuinfo);
void mcx_loadconfig(FILE* in, Config* cfg);
void mcx_saveconfig(FILE* in, Config* cfg);
void mcx_readconfig(char* fname, Config* cfg);
void mcx_writeconfig(const char* fname, Config* cfg);
void mcx_initcfg(Config* cfg);
void mcx_clearcfg(Config* cfg);
void mcx_parsecmd(int argc, char* argv[], Config* cfg);
void mcx_usage(Config* cfg, char* exename);
void mcx_loadvolume(char* filename, Config* cfg, int isbuf);
void mcx_normalize(float field[], float scale, int fieldlen, int option, int pidx, int srcnum);
int  mcx_readarg(int argc, char* argv[], int id, void* output, const char* type);
void mcx_printlog(Config* cfg, const char* str);
int  mcx_remap(char* opt);
void mcx_maskdet(Config* cfg);
void mcx_prepdomain(char* filename, Config* cfg);
void mcx_createfluence(float** fluence, Config* cfg);
void mcx_clearfluence(float** fluence);
void mcx_convertrow2col(unsigned int** vol, uint4* dim);
void mcx_convertcol2row(unsigned int** vol, uint3* dim);
void mcx_convertcol2row4d(unsigned int** vol, uint4* dim);
void mcx_savedetphoton(float* ppath, void* seeds, int count, int seedbyte, Config* cfg);
int  mcx_loadjson(cJSON* root, Config* cfg);
int  mcx_keylookup(char* key, const char* table[]);
int  mcx_lookupindex(char* key, const char* index);
int  mcx_parsedebugopt(char* debugopt, const char* debugflag);
void mcx_printheader(Config* cfg);
void mcx_dumpmask(Config* cfg);
void mcx_version(Config* cfg);
int  mcx_isbinstr(const char* str);
void mcx_progressbar(float percent, Config* cfg);
void mcx_flush(Config* cfg);
void mcx_loadseedfile(Config* cfg);
void mcx_kahanSum(float* sum, float* kahanc, float input);
float mcx_updatemua(unsigned int mediaid, Config* cfg);
void mcx_savejdata(char* filename, Config* cfg);
int  mcx_jdataencode(void* vol,  int ndim, uint* dims, char* type, int byte, int zipid, void* obj, int isubj, Config* cfg);
int  mcx_jdatadecode(void** vol, int* ndim, uint* dims, int maxdim, char** type, cJSON* obj, Config* cfg);
void mcx_savejnii(float* vol, int ndim, uint* dims, float* voxelsize, char* name, int isfloat, Config* cfg);
void mcx_savebnii(float* vol, int ndim, uint* dims, float* voxelsize, char* name, int isfloat, Config* cfg);
void mcx_savejdet(float* ppath, void* seeds, uint count, int doappend, Config* cfg);
void mcx_replayprep(Config* cfg, float* detps, unsigned int dimdetps[2], int seedbyte);
void mcx_validateconfig(Config* cfg);


#ifdef MCX_CONTAINER
int  mcx_throw_exception(const int id, const char* msg, const char* filename, const int linenum);
void mcx_matlab_flush(void);
void mcx_python_flush(void);
#endif

#if defined(MCX_CONTAINER) && (defined(MATLAB_MEX_FILE) || defined(OCTAVE_API_VERSION_NUMBER))
#ifdef _OPENMP
#define MCX_FPRINTF(fp,...) {if(omp_get_thread_num()==0) mexPrintf(__VA_ARGS__);}  /**< macro to print messages, calls mexPrint if inside MATLAB */
#else
#define MCX_FPRINTF(fp,...) mexPrintf(__VA_ARGS__) /**< macro to print messages, calls mexPrint in MATLAB */
#endif
#else
#define MCX_FPRINTF(fp,...) fprintf(fp,__VA_ARGS__) /**< macro to print messages, calls fprintf in command line mode */
#endif

#if defined(MATLAB_MEX_FILE) || defined(OCTAVE_API_VERSION_NUMBER)
int mexPrintf(const char* format, ... );
#else
int mexPrintf(const char* format, ... );
#endif
int mexEvalString(const char* command);

#ifdef  __cplusplus
}
#endif

#endif
