#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include <CL/cl.h>
#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct KernelParams {
  cl_float4 ps,c0;
  cl_float4 maxidx;
  cl_uint4  dimlen,cp0,cp1;
  cl_uint2  cachebox;
  cl_float  minstep;
  cl_float  twin0,twin1,tmax;
  cl_float  oneoverc0;
  cl_uint isrowmajor,save2pt,doreflect,dorefint,savedet;
  cl_float  Rtstep;
  cl_float  minenergy;
  cl_float  skipradius2;
  cl_float  minaccumtime;
  cl_uint maxdetphoton;
  cl_uint maxmedia;
  cl_uint detnum;
  cl_uint idx1dorig;
  cl_uint mediaidorig;
}MCXParam __attribute__ ((aligned (16)));

void mcx_run_simulation(Config *cfg,int activedev,float *fluence,float *totalenergy);
cl_platform_id mcx_set_gpu(Config *cfg,unsigned int *activedev);
void ocl_assess(int cuerr,const char *file,const int linenum);

#ifdef  __cplusplus
}
#endif

#endif
