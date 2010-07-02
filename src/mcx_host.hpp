#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include <CL/cl.h>
#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

void mcx_run_simulation(Config *cfg,int threadid,float *fluence,float *totalenergy);
cl_platform_id mcx_set_gpu(Config *cfg,unsigned int *activedev);
void mcx_assess(int cuerr);

#ifdef  __cplusplus
}
#endif

#endif
