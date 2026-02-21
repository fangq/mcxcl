/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration \
**      -- OpenCL/CUDA dual-backend edition
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025
**
**  \section sref Reference:
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504 (2018)
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

#include <stdio.h>
#include "mcx_tictoc.h"
#include "mcx_utils.h"

#ifdef USE_OPENCL
    #include "mcx_host.h"
#endif

#ifdef USE_CUDA
    #include "mcx_cu_host.h"
#endif

int main(int argc, char* argv[]) {
    Config mcxconfig;
    float* fluence = NULL, totalenergy = 0.f;

    mcx_initcfg(&mcxconfig);

    // parse command line options to initialize the configurations
    mcx_parsecmd(argc, argv, &mcxconfig);

    mcx_createfluence(&fluence, &mcxconfig);

    /*
     * Backend dispatch:
     * - If both backends are compiled (trinity), use mcxconfig.compute to select
     * - If only one backend is compiled, always use that one
     */

#if defined(USE_CUDA) && defined(USE_OPENCL)

    /* Trinity mode: dispatch based on user selection (-K flag) */
    if (mcxconfig.compute == cbCUDA) {
        mcx_run_cuda(&mcxconfig, fluence, &totalenergy);
    } else {
        mcx_run_simulation(&mcxconfig, fluence, &totalenergy);
    }

#elif defined(USE_CUDA)
    /* CUDA-only build */
    mcx_run_cuda(&mcxconfig, fluence, &totalenergy);
#elif defined(USE_OPENCL)
    /* OpenCL-only build */
    mcx_run_simulation(&mcxconfig, fluence, &totalenergy);
#else
#error "At least one of USE_OPENCL or USE_CUDA must be defined"
#endif

    // clean up the allocated memory in the config
    mcx_clearfluence(&fluence);
    mcx_clearcfg(&mcxconfig);
    return 0;
}