/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration \
**      -- OpenCL edition
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
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
#include "tictoc.h"
#include "mcx_utils.h"
#include "mcx_host.hpp"


int main (int argc, char *argv[]) {
     Config mcxconfig;
     float *fluence=NULL,totalenergy=0.f;

     mcx_initcfg(&mcxconfig);

     // parse command line options to initialize the configurations
     mcx_parsecmd(argc,argv,&mcxconfig);

     mcx_createfluence(&fluence,&mcxconfig);

     // this launches the MC simulation
     mcx_run_simulation(&mcxconfig,fluence,&totalenergy);

     // clean up the allocated memory in the config
     mcx_clearfluence(&fluence);
     mcx_clearcfg(&mcxconfig);
     return 0;
}
