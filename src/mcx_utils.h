#ifndef _MCEXTREME_UTILITIES_H
#define _MCEXTREME_UTILITIES_H

#include <stdio.h>
#ifndef MCX_OPENCL
  #include <vector_types.h>
#else
  #include <CL/cl.h>
/* #ifdef CL_PLATFORM_NVIDIA*/
  typedef struct vec_float4{
       float x,y,z,w;
  }float4;
  typedef struct vec_uint4{
       unsigned int x,y,z,w;
  }uint4;
  typedef struct vec_uint2{
       unsigned int x,y;
  }uint2;
/* #endif*/
#endif

#define MAX_PATH_LENGTH     1024
#define MAX_SESSION_LENGTH  256
#define MAX_DEVICE          256

typedef struct MCXMedium{
	float mua;
	float mus;
	float n;
	float g;
} Medium;  /*this order shall match prop.{xyzw} in mcx_main_loop*/

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
	int reserved[7];
} History;

typedef struct MCXConfig{
	int nphoton;      /*(total simulated photon number) we now use this to 
	                     temporarily alias totalmove, as to specify photon
			     number is causing some troubles*/
	//int totalmove;   /* [depreciated] total move per photon*/
        int nblocksize;   /*thread block size*/
	int nthread;      /*num of total threads, multiple of 128*/
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

	int maxgate;        /*simultaneous recording gates*/
	int respin;         /*number of repeatitions*/
	int printnum;       /*number of printed threads (for debugging)*/

	unsigned char *vol; /*pointer to the volume*/
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
        float minenergy;    /*minimum energy to propagate photon*/
        float unitinmm;     /*defines the length unit in mm for grid*/
        FILE *flog;         /*stream handle to print log information*/
        History his;        /*header info of the history file*/
        char rootpath[MAX_PATH_LENGTH];
        char kernelfile[MAX_SESSION_LENGTH];
	char *clsource;
        char deviceid[MAX_DEVICE];
	float workload[MAX_DEVICE];
	float *exportfield;     /*memory buffer when returning the flux to external programs such as matlab*/
	float *exportdetected;  /*memory buffer when returning the partial length info to external programs such as matlab*/
} Config;

#ifdef	__cplusplus
extern "C" {
#endif
void mcx_savedata(float *dat,int len,int doappend, const char *suffix, Config *cfg);
void mcx_error(int id,const char *msg);
void mcx_loadconfig(FILE *in, Config *cfg);
void mcx_saveconfig(FILE *in, Config *cfg);
void mcx_readconfig(const char *fname, Config *cfg);
void mcx_writeconfig(const char *fname, Config *cfg);
void mcx_initcfg(Config *cfg);
void mcx_clearcfg(Config *cfg);
void mcx_parsecmd(int argc, char* argv[], Config *cfg);
void mcx_usage(char *exename);
void mcx_loadvolume(char *filename,Config *cfg);
void mcx_normalize(float field[], float scale, int fieldlen);
int  mcx_readarg(int argc, char *argv[], int id, void *output,const char *type);
void mcx_printlog(Config *cfg, const char *str);
int  mcx_remap(char *opt);
void mcx_createfluence(float **fluence, Config *cfg);
void mcx_clearfluence(float **fluence);
void mcx_convertrow2col(unsigned char **vol, uint4 *dim);

#ifdef	__cplusplus
}
#endif

#endif
