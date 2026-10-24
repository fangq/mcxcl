/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2026
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    test_main.c

@brief   Entry point of the MCX-CL host unit tests

Runs the host tests of test_utils.c, which is shared with mcx, against the
mcx-cl objects. No OpenCL device is needed; build and run it with
"make unittest" under src/.
*******************************************************************************/

#include <stdio.h>

int ntest = 0, nfail = 0;

void run_utils_tests(void);

int main(void) {
    run_utils_tests();

    printf("%d checks, %d failures\n", ntest, nfail);
    return nfail > 0;
}
