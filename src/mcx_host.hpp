#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include <CL/cl.h>
#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct KernelParams {
  float4 ps,c0;
  float4 maxidx;
  uint4  dimlen,cp0,cp1;
  uint2  cachebox;
  float  minstep;
  float  twin0,twin1,tmax;
  float  oneoverc0;
  unsigned int isrowmajor,save2pt,doreflect,dorefint,savedet;
  float  Rtstep;
  float  minenergy;
  float  skipradius2;
  float  minaccumtime;
  unsigned int maxdetphoton;
  unsigned int maxmedia;
  unsigned int detnum;
  unsigned int idx1dorig;
  unsigned int mediaidorig;
}MCXParam __attribute__ ((aligned (16)));

void mcx_run_simulation(Config *cfg,int threadid,int activedev,float *fluence,float *totalenergy);
cl_platform_id mcx_set_gpu(Config *cfg,unsigned int *activedev);
void mcx_assess(int cuerr);

#ifdef  __cplusplus
}
#endif

#endif
