---------------------------------------------------------------------
                   Monte Carlo eXtreme  (MCX)
                         OpenCL Edition
---------------------------------------------------------------------

Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
License: GNU General Public License version 3 (GPLv3)
Version: 0.7.9.pre (Charm Quarks)

---------------------------------------------------------------------

Table of Content:

I.  Introduction
II. Requirement and Installation
III.Running Simulations
IV. Using MCX Studio GUI
V.  Interpreting the Outputs
VI. Reference

---------------------------------------------------------------------

I.  Introduction

Monte Carlo eXtreme (MCX) is a fast photon transport simulation 
software for 3D heterogeneous turbid media. By taking advantage of 
the massively parallel threads and extremely low memory latency in a 
modern graphics processing unit (GPU), this program is able to perform Monte 
Carlo (MC) simulations at a blazing speed, typically hundreds to
a thousand times faster than a fully optimized CPU-based MC 
implementation.

The algorithm of this software is detailed in the Reference [1]. 
A short summary of the main features includes:

*. 3D heterogeneous media represented by voxelated array
*. boundary reflection support
*. time-resolved photon transport simulation
*. saving photon partial path lengths at the detectors
*. optimized random number generators
*. build-in flux/fluence normalization to output Green's functions
*. user adjustable voxel resolution
*. improved accuracy near the source with atomic operations
*. cross-platform graphics user interface
*. native Matlab/Octave support for high usability
*. flexible JSON interface for future extensions

This software can be used on Windows, Linux and Mac OS. 
MCX is written in CUDA and can be used with NVIDIA hardware
with the native NVIDIA drivers, or used with GPU ocelot open-source
libraries for CPUs and AMD GPUs. An OpenCL implementation of
MCX, i.e. MCXCL, will be announced soon and can support 
NVIDIA/AMD/Intel hardware out-of-box.


---------------------------------------------------------------------------
II. Requirement and Installation

For MCXCL, the requirements for using this software are

*. a single/multi-core CPU, or
*. a CUDA capable nVidia graphics card, or
*. a AMD/ATI graphics card, or
*. pre-installed graphics driver and a valid OpenCL library (libOpenCL.* or OpenCL.dll)

To install MCXCL, you simply download the binary executable corresponding to your 
computer architecture (32 or 64bit) and platform, extract the package 
and run the executable under the <mcxcl root>/bin directory. For Linux
and MacOS users, please double check and make sure libOpenCL.so is installed under 
the /usr/lib directory. If it is installed under a different directory, please
define environment variable LD_LIBRARY_PATH to include the path.

If libOpenCL.so or OpenCL.dll does not exist on your system or, please
make sure you have installed CUDA SDK (if you are using an nVidia card)
or AMD APP SDK (if you are using an AMD card). 


---------------------------------------------------------------------------
III.Running Simulations

To run a simulation, the minimum input is a configuration (text) file,
and a volume (a binary file with each byte representing a medium 
index). Typing the name of the executable without any parameters, 
will print the help information and a list of supported parameters, 
such as the following:

 usage: mcxcl <param1> <param2> ...
 where possible parameters include (the first item in [] is the default value)
  -i 	        (--interactive) interactive mode
  -f config      (--input)	read config from a file
  -t [1024|int]  (--thread)	total thread number
  -T [64|int]    (--blocksize)	thread number per block
  -n [0|int]     (--photon)	total photon number
  -r [1|int]     (--repeat)	number of repeations
  -a [0|1]       (--array)	0 for Matlab array, 1 for C array
  -z [0|1]       (--srcfrom0)    src/detector coordinates start from 0, otherwise from 1
  -g [1|int]     (--gategroup)	number of time gates per run
  -b [1|0]       (--reflect)	1 to reflect the photons at the boundary, 0 to exit
  -B [0|1]       (--reflect3)	1 to consider maximum 3 reflections, 0 consider only 2
  -e [0.|float]  (--minenergy)	minimum energy level to propagate a photon
  -R [0.|float]  (--skipradius)  minimum distance to source to start accumulation
  -U [1|0]       (--normalize)	1 to normailze the fluence to unitary, 0 to save raw fluence
  -d [1|0]       (--savedet)	1 to save photon info at detectors, 0 not to save
  -S [1|0]       (--save2pt)	1 to save the fluence field, 0 do not save
  -s sessionid   (--session)	a string to identify this specific simulation (and output files)
  -p [0|int]     (--printlen)	number of threads to print (debug)
  -h             (--help)	print this message
  -l             (--log) 	print messages to a log file instead
  -L             (--listgpu)	print GPU information only
  -I             (--printgpu)	print GPU information and run program
  -c             (--cpu) 	use CPU as the platform for OpenCL backend
  -k mcx_core.cl (--kernel)      specify path to OpenCL kernel source file
  -G '0111'      (--devicelist)  specify the active OpenCL devices (1 enable, 0 disable)
  -W '50,30,20'  (--workload)    specify relative workload for each device; total is the sum
  -J '-D MCX'    (--compileropt) specify additional JIT compiler options
 example:
  mcxcl -t 1024 -T 64 -n 1e7 -f input.inp -s test -r 1 -b 0 -G 1010 -W '50,50' -k ../../src/mcx_core.cl

 the above command will launch 1024 GPU threads (-t) with every 64 threads
 a block (-T); for each thread, it will simulate 1e7 photons (-n) and
 repeat only once (-r); the media/source configuration will be read from 
 input.inp (-f) and the output will be labeled with the session id "test" (-s); 
 the simulation will utilize the 1st and 3rd Compute Units among the 4 total 
 devices present in the system (-G 1010); the list of CU can be found by mcxcl -L; 
 the workload partition between the two selected devices is 50:50 (-W); the simulation
 requires the relative/full path to the kernel source file mcx_core.cl (-k).

Currently, MCX supports a modified version of the input file format used 
for tMCimg. (The difference is that MCX allows comments)
A typical MCX input file looks like this:

1000000              # total photon, use -n to overwrite in the command line
29012392             # RNG seed, negative to generate
30.0 30.0 0.0 1      # source position (in grid unit), the last num sets srcfrom0 (-z)
0 0 1                # initial directional vector
0.e+00 1.e-09 1.e-10 # time-gates(s): start, end, step
semi60x60x60.bin     # volume ('unsigned char' format)
1 60 1 60            # x voxel size in mm (isotropic only), dim, start/end indices
1 60 1 60            # y voxel size, must be same as x, dim, start/end indices 
1 60 1 60            # y voxel size, must be same as x, dim, start/end indices
1                    # num of media
1.010101 0.01 0.005 1.37  # scat. mus (1/mm), g, mua (1/mm), n
4       1.0          # detector number and default radius (in grid unit)
30.0  20.0  0.0  2.0 # detector 1 position (real numbers in grid unit) and radius if different
30.0  40.0  0.0      # ..., if radius is ignored, MCX will use the default radius
20.0  30.0  0.0      #
40.0  30.0  0.0      # 

Note that the scattering coefficient mus=musp/(1-g).

The volume file (semi60x60x60.bin in the above example),
can be read in two ways by MCXCL: row-major[3] or column-major
depending on the value of the user parameter "-a". If the volume file
was saved using matlab or fortran, the byte order is column-major,
and you should use "-a 0" or leave it out of the command line. 
If it was saved using the fwrite() in C, the order is row-major, 
and you can either use "-a 1".

The time gate parameter is specified by three numbers:
start time, end time and time step size (in seconds). In 
the above example, the configuration specifies a total time 
window of [0 1] ns, with a 0.1 ns resolution. That means the 
total number of time gates is 10. 

MCXCL provides an advanced option, -g, to run simulations when 
the GPU memory is limited. It specifies how many time gates to simulate 
concurrently. Users may want to limit that number to less than 
the total number specified in the input file - and by default 
it runs one gate at a time in a single simulation. But if there's 
enough memory based on the memory requirement in Section II, you can 
simulate all 10 time gates (from the above example) concurrently by using 
"-g 10" in which case you have to make sure the video card has at least  
60*60*60*10*5=10MB of free memory.   If you do not include the -g, 
MCX will assume you want to simulate just 1 time gate at a time.. 
If you specify a time-gate number greater than the total number in the 
input file, (e.g, "-g 20") MCX will stop when the 10 time-gates are 
completed. If you use the autopilot mode (-A), then the time-gates
are automatically estimated for you.


---------------------------------------------------------------------------
IV. Using MCX Studio GUI

MCX Studio is a graphics user interface (GUI) for MCX. It gives users
a straightforward way to set the command line options and simulation
parameters. It also allows users to create different simulation tasks 
and organize them into a project and save for later use.
MCX Studio can be run on many platforms such as Windows,
GNU Linux and Mac OS.

To use MCX Studio, it is suggested to put the mcxstudio binary
in the same directory as the mcx command; alternatively, you can
also add the path to mcx command to your PATH environment variable.

When launching MCX Studio, it will automatically check if mcx
binary is in the search path, if so, the "GPU" button in the 
toolbar will be enabled. It is suggested to click on this button
once, and see if you can see a list of GPUs and their parameters 
printed in the output field at the bottom part of the window. 
If you are able to see this information, your system is ready
to run MCX simulations. If you get error messages or not able
to see any usable GPU, please check the following:

* are you running MCX Studio/MCX on a computer with a supported card?
* have you installed the CUDA/NVIDIA drivers correctly?
* did you put mcx in the same folder as mcxstudio or add its path to PATH?

If your system is properly configured, you can now add new simulations 
by clicking the "New" button. MCX Studio will ask you to give a session
id string for this task. Then you should be able to adjust the parameters
based on your needs. Once you finish the adjustment, you should click the 
"Verify" button to see if there are obvious mistakes. If everything is
fine, the "Run" button will be activated. Click on it once will start your
simulation. If you want to abort the current simulation, you can click
the "Stop" button.

You can create multiple tasks with MCX Studio by hitting the "New"
button multiple times. The information for all of the sessions can
be saved as a project file (with .mcxp extension) by clicking the
"Save" button. You can load a previously saved project file back
to MCX Studio by clicking the "Load" button.


---------------------------------------------------------------------------
V. Interpreting Output

MCX output consists of two parts, the fluence volume 
file and messages printed on the screen.

5.1 Output files

An mc2 file contains the flux distribution from the simulation in 
the given medium. By default, this flux is a normalized solution 
(as opposed to the raw probability) therefore, one can compare this directly 
to the analytical solutions (i.e. Green's function). The order of storage in the 
mc2 files is the same as the input file: i.e., if the input is row-major, the 
output is row-major, and so on. The dimensions of the file are Nx, Ny, Nz, and Ng
where Ng is the total number of time gates.

By default, MCX produces the '''Green's function''' of the 
'''fluence rate''' (or '''flux''') for the given domain and 
source. Sometime it is also known as the time-domain "two-point" 
function. If you run MCX with the following command

  mcxcl -f input.inp -s output ....

the flux data will be saved in a file named "output.dat" under
the current folder. If you run MCX without "-s output", the
output file will be named as "input.inp.dat".

To understand this further, you need to know that a '''flux''' is
measured by number of particles passing through an infinitesimal 
spherical surface per <em>unit time</em> at <em>a given location</em>.
The unit of MCX output flux is "1/(mm<sup>2</sup>s)", if the flux is interpreted as the 
"particle flux" [6], or "J/(mm<sup>2</sup>s)", if it is interpreted as the 
"energy flux" [6].

The Green's function of the flux simply means that the flux is produced
by a '''unitary source'''. In simple terms, this represents the 
fraction of particles/energy that arrives a location per second 
under <em>the radiation of 1 unit (packet or J) of particle or energy 
at time t=0</em>. The Green's function is calculated by a process referred
to as the "normalization" in the MCX code and is detailed in the 
MCX paper [6] (MCX and MMC outputs share the same meanings).

Please be aware that the output flux is calculated at each time-window 
defined in the input file. For example, if you type 

 0.e+00 5.e-09 1e-10  # time-gates(s): start, end, step

in the 5th row in the input file, MCX will produce 50 flux
distributions, corresponding to the time-windows at [0 0.1] ns, 
[0.1 0.2]ns ... and [4.9,5.0] ns. To convert the flux distributions
to the fluence distributions for each time-window, you just need to
multiply each solution by the width of the window, 0.1 ns in this case. To convert the time-domain flux
to the continuous-wave (CW) fluence, you need to integrate the
flux in t=[0,inf]. Assuming the flux after 5 ns is negligible, then the CW
fluence is simply sum(flux_i*0.1 ns, i=1,50). You can read 
<tt>mcx/examples/validation/plotsimudata.m</tt>
and <tt>mcx/examples/sphbox/plotresults.m</tt> for examples 
to compare an MCX output with the analytical flux/fluence solutions.

One can load an mc2 output file into Matlab or Octave using the
loadmc2 function in the <mcx root>/utils folder. 

To get a continuous-wave solution, run a simulation with a sufficiently 
long time window, and sum the flux along the time dimension, for 
example

   mcx=loadmc2('output.mc2',[60 60 60 10],'float');
   cw_mcx=sum(mcx,4);

Note that for time-resolved simulations, the corresponding solution
in the results approximates the flux at the center point
of each time window. For example, if the simulation time window 
setting is [t0,t0+dt,t0+2dt,t0+3dt...,t1], the time points for the 
snapshots stored in the solution file is located at 
[t0+dt/2, t0+3*dt/2, t0+5*dt/2, ... ,t1-dt/2]

A more detailed interpretation of the output data can be found at 
http://mcx.sf.net/cgi-bin/index.cgi?MMC/Doc/FAQ#How_do_I_interpret_MMC_s_output_data


5.2 Console Print messages

Timing information is printed on the screen (stdout). The 
clock starts (at time T0) right before the initialization data is copied 
from CPU to GPU. For each simulation, the elapsed time from T0
is printed (in ms). Also the accumulated elapsed time is printed for 
all memory transaction from GPU to CPU.


---------------------------------------------------------------------------
VI. Reference

[1] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon \
Migration in 3D Turbid Media Accelerated by Graphics Processing Units,"
Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009).

If you used MCX in your research, the authors of this software would like
you to cite the above paper in your related publications.

Links: 

[1] http://www.nvidia.com/object/cuda_get.html
[2] http://www.nvidia.com/object/cuda_learn_products.html
[3] http://en.wikipedia.org/wiki/Row-major_order
