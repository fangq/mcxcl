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

#define _DEFAULT_SOURCE  /* on_exit() under -std=c99 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int ntest = 0, nfail = 0;

void run_utils_tests(void);

#ifdef __GLIBC__
#include <pthread.h>

/** a forked child leaves through _exit(): exit() would run the destructors of the GPU
    runtimes the parent loaded, and the Intel OpenCL runtime waits there for its threads,
    which fork() did not copy, so the child spun forever */
static void ut_quickexit(int status, void* arg) {
    (void)arg;
    _exit(status);
}

static void ut_child_quickexit(void) {
    on_exit(ut_quickexit, NULL);
}
#endif

int main(void) {
#ifdef __GLIBC__
    pthread_atfork(NULL, NULL, ut_child_quickexit);
#endif

    run_utils_tests();

    printf("%d checks, %d failures\n", ntest, nfail);
    return nfail > 0;
}
