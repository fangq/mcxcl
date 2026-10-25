= README for the quicktest example =

As the name indicates, this example runs a short simulation
for the purpose of testing the command line options build
with MCX-CL. 

To run this example, you need to first compile the mcxcl binary,
or download a pre-compiled mcxcl package, so that ../../bin/mcxcl exists.
Then you can choose to run the following shell scripts:

listgpu.sh
    A script to call mcxcl to print the CPU/GPU information 
    only. Run this script and find out how many GPUs you 
    have in your graphics card and the related parameters 
    (memory, cores etc) 

run_qtest.sh
    This will run MCX-CL for a simple homogeneous medium as 
    the example used in the Fang2009 paper. It launches 
    1e7 photons; the fluence is saved in qtest.jnii.

run_qtest_silent.sh
    This script does the same thing as run_qtest.sh, 
    except it lets mcxcl print messages to a log file 
    rather than printing on the screen (so called silent mode) 


[Fang2009]   Qianqian Fang and David A. Boas, "Monte Carlo 
   Simulation of Photon Migration in 3D Turbid Media Accelerated 
   by Graphics Processing Units," Opt. Express, 
   vol. 17, issue 22, pp. 20178-20190 (2009)
