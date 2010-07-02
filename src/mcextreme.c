/*******************************************************************************
**
**  Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**  
**  Author     : Qianqian Fang
**  Email      : <fangq at nmr.mgh.harvard.edu>
**  Institution: Massachusetts General Hospital / Harvard Medical School
**  Address    : Bldg. 149, 13th Street, Charlestown, MA 02148, USA
**  Homepage   : http://nmr.mgh.harvard.edu/~fangq/
**
**  MCX Web    : http://mcx.sourceforge.net
**
**  Unpublished work, see LICENSE.txt
**
*******************************************************************************/

#include <stdio.h>
#include "tictoc.h"
#include "mcx_utils.h"
#include "mcx_host.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

int main (int argc, char *argv[]) {
     Config mcxconfig;
     unsigned int threadid=0,activedev=0;
     float *fluence=NULL,totalenergy=0.f;

     mcx_initcfg(&mcxconfig);

     // parse command line options to initialize the configurations
     mcx_parsecmd(argc,argv,&mcxconfig);

     // identify gpu number and set one gpu active
     if(!mcx_set_gpu(&mcxconfig,&activedev)){
         mcx_error(-1,"No compute platform was found\n");
     }
     if(activedev==0)
     	return 0;

     mcx_createfluence(&fluence,&mcxconfig);

#ifdef _OPENMP
     omp_set_num_threads(activedev);
#endif

#pragma omp parallel private(threadid)
{
#ifdef _OPENMP
     threadid=omp_get_thread_num();
#endif

     // this launches the MC simulation
     mcx_run_simulation(&mcxconfig,threadid,fluence,&totalenergy);
}
     // clean up the allocated memory in the config

     mcx_clearfluence(&fluence);
     mcx_clearcfg(&mcxconfig);
     return 0;
}
