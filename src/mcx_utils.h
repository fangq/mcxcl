#ifndef _MCEXTREME_UTILITIES_H
#define _MCEXTREME_UTILITIES_H

#include <stdio.h>

#include "vector_types.h"
#include "cjson/cJSON.h"
#include "nifti1.h"

#define MAX_PATH_LENGTH     1024
#define MAX_SESSION_LENGTH  256
#define MAX_DEVICE          256

#define MCX_ASSERT(x)  mcx_assess((x),"assert error",__FILE__,__LINE__)
#define MCX_ERROR(id,msg)   mcx_error(id,msg,__FILE__,__LINE__)
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MAX(a,b)           ((a)>(b)?(a):(b))

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352

enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otWP};   /**< types of output */
enum TMCXParent  {mpStandalone, mpMATLAB};
enum TOutputFormat {ofMC2, ofNifti, ofAnalyze, ofUBJSON};
enum TDeviceVendor {dvUnknown, dvNVIDIA, dvAMD, dvIntel, dvIntelGPU};

typedef struct MCXMedium{
	float mua;
	float mus;
	float g;
	float n;
} Medium __attribute__ ((aligned (16)));  /*this order shall match prop.{xyzw} in mcx_main_loop*/

typedef struct MCXHistoryHeader{
	char magic[4];
	unsigned int  version;
	unsigned int  maxmedia;
	unsigned int  detnum;
	unsigned int  colcount;
	unsigned int  totalphoton;
	unsigned int  detected;
	unsigned int  savedphoton;
	float unitinmm;
	unsigned int  seedbyte;
        float normalizer;              /**< what is the normalization factor */
	int reserved[5];
} History;

typedef struct PhotonReplay{
	void  *seed;
	float *weight;
	float *tof;
} Replay;

typedef struct MCXGPUInfo {
        char name[MAX_SESSION_LENGTH];
        int id;
	int devcount;
        int platformid;
        int major, minor;
        size_t globalmem, constmem, sharedmem;
        int regcount;
        int clock;
        int sm, core;
        size_t autoblock, autothread;
        int maxgate;
        int maxmpthread;  /**< maximum thread number per multi-processor */
        enum TDeviceVendor vendor;
} GPUInfo;

typedef struct MCXConfig{
	int nphoton;      /*(total simulated photon number) we now use this to 
	                     temporarily alias totalmove, as to specify photon
			     number is causing some troubles*/
        unsigned int nblocksize;   /*thread block size*/
	unsigned int nthread;      /*num of total threads, multiple of 128*/
	int seed;         /*random number generator seed*/
	
	float4 srcpos;    /*src position in mm*/
	float4 srcdir;    /*src normal direction*/
	float tstart;     /*start time in second*/
	float tstep;      /*time step in second*/
	float tend;       /*end time in second*/
	float4 steps;     /*voxel sizes along x/y/z in mm*/
	
	uint4 dim;        /*domain size*/
	uint4 crop0;      /*sub-volume for cache*/
	uint4 crop1;      /*the other end of the caching box*/
	unsigned int medianum;     /*total types of media*/
	unsigned int detnum;       /*total detector numbers*/
        unsigned int maxdetphoton; /*anticipated maximum detected photons*/
	float detradius;  /*detector radius*/
        float sradius;    /*source region radius: if set to non-zero, accumulation 
                            will not perform for dist<sradius; this can reduce
                            normalization error when using non-atomic write*/

	Medium *prop;     /*optical property mapping table*/
	float4 *detpos;   /*detector positions and radius, overwrite detradius*/

	unsigned int maxgate;        /*simultaneous recording gates*/
	unsigned int respin;         /*number of repeatitions*/
	int printnum;       /*number of printed threads (for debugging)*/
	unsigned int reseedlimit;     /**<number of scattering events per thread before the RNG is reseeded*/
	int gpuid;                    /**<the ID of the GPU to use, starting from 1, 0 for auto*/

	unsigned int *vol; /*pointer to the volume*/
	char session[MAX_SESSION_LENGTH]; /*session id, a string*/
	char isrowmajor;    /*1 for C-styled array in vol, 0 for matlab-styled array*/
	char isreflect;     /*1 for reflecting photons at boundary,0 for exiting*/
        char isref3;        /*1 considering maximum 3 ref. interfaces; 0 max 2 ref*/
        char isrefint;      /*1 to consider reflections at internal boundaries; 0 do not*/
	char isnormalized;  /*1 to normalize the fluence, 0 for raw fluence*/
	char issavedet;     /*1 to count all photons hits the detectors*/
	char issave2pt;     /*1 to save the 2-point distribution, 0 do not save*/
	char isgpuinfo;     /*1 to print gpu info when attach, 0 do not print*/
	char iscpu;         /*1 use CPU for simulation, 0 use GPU*/
	char isverbose;     /*1 print debug info, 0 do not*/
	char issrcfrom0;    /*1 do not subtract 1 from src/det positions, 0 subtract 1*/
        char isdumpmask;    /*1 dump detector mask; 0 not*/
	char issaveseed;             /**<1 save the seed for a detected photon, 0 do not save*/
	char issaveexit;             /**<1 save the exit position and dir of a detected photon, 0 do not save*/
	char isatomic;      /*1 use atomic operations, 0 no atomic*/
	char issaveref;              /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
	char srctype;                /**<0:pencil,1:isotropic,2:cone,3:gaussian,4:planar,5:pattern,\
                                         6:fourier,7:arcsine,8:disk,9:fourierx,10:fourierx2d,11:zgaussian,12:line,13:slit*/
        char autopilot;     /**<1 optimal setting for dedicated card, 2, for non dedicated card*/
        char outputtype;    /**<'X' output is flux, 'F' output is fluence, 'E' energy deposit*/
        char outputformat;  /**<'mc2' output is text, 'nii': binary, 'img': regular json, 'ubj': universal binary json*/
        float minenergy;    /*minimum energy to propagate photon*/
        float unitinmm;     /*defines the length unit in mm for grid*/
        FILE *flog;         /*stream handle to print log information*/
        History his;        /*header info of the history file*/
	double energytot, energyabs, energyesc;
        char rootpath[MAX_PATH_LENGTH];
        char kernelfile[MAX_SESSION_LENGTH];
	char compileropt[MAX_PATH_LENGTH];
        char *shapedata;    /**<a pointer points to a string defining the JSON-formatted shape data*/
	char *clsource;
	int maxvoidstep;             /**< max number of steps that a photon can advance before reaching a non-zero voxel*/
	int voidtime;                /**<1 start counting photon time when moves inside 0 voxels; 0: count time only after enters non-zero voxel*/
	float4 srcparam1;            /**<a quadruplet {x,y,z,w} for additional source parameters*/
	float4 srcparam2;            /**<a quadruplet {x,y,z,w} for additional source parameters*/
        float* srcpattern;           /**<a string for the source form, options include "pencil","isotropic", etc*/
        char deviceid[MAX_DEVICE];
	float workload[MAX_DEVICE];
	float *exportfield;     /*memory buffer when returning the flux to external programs such as matlab*/
	float *exportdetected;  /*memory buffer when returning the partial length info to external programs such as matlab*/
        unsigned int debuglevel;     /**<a flag to control the printing of the debug information*/
	char faststep;               /**<1 use tMCimg-like approximated photon stepping (obsolete) */
	float normalizer;            /**<normalization factor*/

	Replay replay;               /**<a structure to prepare for photon replay*/
	void *seeddata;              /**<poiinter to a buffer where detected photon seeds are stored*/
        int replaydet;               /**<the detector id for which to replay the detected photons, start from 1*/
        char seedfile[MAX_PATH_LENGTH];/**<if the seed is specified as a file (mch), mcx will replay the photons*/

	unsigned int maxjumpdebug;   /**<num of  photon scattering events to save when saving photon trajectory is enabled*/
	unsigned int debugdatalen;   /**<max number of photon trajectory position length*/
	unsigned int gscatter;       /**<after how many scattering events that we can use mus' instead of mus */
	float *exportdebugdata;      /**<pointer to the buffer where the photon trajectory data are stored*/

	unsigned int detectedcount; /**<total number of detected photons*/
	unsigned int runtime;
	int parentid;
        uint optlevel;
	uint mediabyte;
} Config;

#ifdef	__cplusplus
extern "C" {
#endif
void mcx_savedata(float *dat, int len, Config *cfg);
void mcx_savenii(float *dat, int len, char* name, int type32bit, int outputformatid, Config *cfg);
void mcx_error(const int id,const char *msg,const char *file,const int linenum);
void mcx_assess(const int id,const char *msg,const char *file,const int linenum);
void mcx_cleargpuinfo(GPUInfo **gpuinfo);
void mcx_loadconfig(FILE *in, Config *cfg);
void mcx_saveconfig(FILE *in, Config *cfg);
void mcx_readconfig(char *fname, Config *cfg);
void mcx_writeconfig(const char *fname, Config *cfg);
void mcx_initcfg(Config *cfg);
void mcx_clearcfg(Config *cfg);
void mcx_parsecmd(int argc, char* argv[], Config *cfg);
void mcx_usage(Config *cfg,char *exename);
void mcx_loadvolume(char *filename,Config *cfg);
void mcx_normalize(float field[], float scale, int fieldlen, int option);
int  mcx_readarg(int argc, char *argv[], int id, void *output,const char *type);
void mcx_printlog(Config *cfg, const char *str);
int  mcx_remap(char *opt);
void mcx_maskdet(Config *cfg);
void mcx_prepdomain(char *filename, Config *cfg);
void mcx_createfluence(float **fluence, Config *cfg);
void mcx_clearfluence(float **fluence);
void mcx_convertrow2col(unsigned int **vol, uint4 *dim);
void mcx_savedetphoton(float *ppath, void *seeds, int count, int seedbyte, Config *cfg);
int  mcx_loadjson(cJSON *root, Config *cfg);
int  mcx_keylookup(char *key, const char *table[]);
int  mcx_lookupindex(char *key, const char *index);
int  mcx_parsedebugopt(char *debugopt,const char *debugflag);
void mcx_printheader(Config *cfg);
void mcx_dumpmask(Config *cfg);
void mcx_version(Config *cfg);
int  mcx_isbinstr(const char * str);
void mcx_progressbar(float percent, Config *cfg);
void mcx_flush(Config *cfg);


#ifdef MCX_CONTAINER
#ifdef __cplusplus
extern "C"
#endif
 int  mcx_throw_exception(const int id, const char *msg, const char *filename, const int linenum);
 void mcx_matlab_flush(void);
#endif

#ifdef MCX_CONTAINER
  #define MCX_FPRINTF(fp,...) mexPrintf(__VA_ARGS__)
#else
  #define MCX_FPRINTF(fp,...) fprintf(fp,__VA_ARGS__)
#endif

#if defined(MATLAB_MEX_FILE) || defined(OCTAVE_API_VERSION_NUMBER)
    int mexPrintf(const char * format, ... );
#else
    int mexPrintf(const char * format, ... );
#endif
int mexEvalString(const char *command);

#ifdef	__cplusplus
}
#endif

#endif
