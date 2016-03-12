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
