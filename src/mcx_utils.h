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

enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otTaylor};
enum TMCXParent  {mpStandalone, mpMATLAB};
enum TOutputFormat {ofMC2, ofNifti, ofAnalyze, ofUBJSON};

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
	int reserved[6];
} History;

typedef struct PhotonReplay{
	void  *seed;
	float *weight;
	float *tof;
} Replay;

typedef struct MCXGPUInfo {
        char name[MAX_SESSION_LENGTH];
        int id;
        int platformid;
        int major, minor;
        size_t globalmem, constmem, sharedmem;
        int regcount;
        int clock;
        int sm, core;
        size_t autoblock, autothread;
        int maxgate;
        int maxmpthread;  /**< maximum thread number per multi-processor */
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
        char deviceid[MAX_DEVICE];
	float workload[MAX_DEVICE];
	float *exportfield;     /*memory buffer when returning the flux to external programs such as matlab*/
	float *exportdetected;  /*memory buffer when returning the partial length info to external programs such as matlab*/
	unsigned int detectedcount; /**<total number of detected photons*/
	unsigned int runtime;
	int parentid;
	void *seeddata;
	uint mediabyte;
} Config;

#ifdef	__cplusplus
extern "C" {
#endif
void mcx_savedata(float *dat, int len, Config *cfg);
void mcx_savenii(float *dat, int len, char* name, int type32bit, int outputformatid, Config *cfg);
void mcx_error(const int id,const char *msg,const char *file,const int linenum);
void mcx_assess(const int id,const char *msg,const char *file,const int linenum);
void mcx_loadconfig(FILE *in, Config *cfg);
void mcx_saveconfig(FILE *in, Config *cfg);
void mcx_readconfig(const char *fname, Config *cfg);
void mcx_writeconfig(const char *fname, Config *cfg);
void mcx_initcfg(Config *cfg);
void mcx_clearcfg(Config *cfg);
void mcx_parsecmd(int argc, char* argv[], Config *cfg);
void mcx_usage(Config *cfg,char *exename);
void mcx_loadvolume(char *filename,Config *cfg);
void mcx_normalize(float field[], float scale, int fieldlen);
int  mcx_readarg(int argc, char *argv[], int id, void *output,const char *type);
void mcx_printlog(Config *cfg, const char *str);
int  mcx_remap(char *opt);
void mcx_maskdet(Config *cfg);
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

#ifdef	__cplusplus
}
#endif

#endif
