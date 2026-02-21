/***************************************************************************//**
**  \file    mcx_cu_host.h
**  \brief   Header file for the CUDA host code (dual-backend MCX)
**
**  This is the CUDA counterpart to mcx_host.h (OpenCL).
**  The shared kernel mcx_core.cl is included by mcx_cu_host.cu.
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025, GPL v3
*******************************************************************************/

#ifndef _MCEXTREME_CUDA_HOST_H
#define _MCEXTREME_CUDA_HOST_H

#include "mcx_utils.h"
#include "mcx_param.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define ABS(a)  ((a)<0?-(a):(a))

#define MCX_RNG_NAME       "xoroshiro128+"

void mcx_run_cuda(Config* cfg, float* fluence, float* totalenergy);
int  mcx_list_cuda_gpu(Config* cfg, GPUInfo** info);

#ifdef  __cplusplus
}
#endif

#endif /* _MCEXTREME_CUDA_HOST_H */