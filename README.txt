---------------------------------------------------------------------
=              Monte Carlo eXtreme for OpenCL (MCX-CL)              =
---------------------------------------------------------------------

*Author: Qianqian Fang (q.fang at neu.edu)
*License: GNU General Public License version 3 (GPLv3)
*Version: 3.0 (v2026, Juxtaposition)
*Website: https://mcx.space/#mcxcl
*Download: https://mcx.space/wiki/?Get
*Discussion: https://groups.google.com/g/mcx-users
---------------------------------------------------------------------

Table of Content:

<toc>

---------------------------------------------------------------------

MCX-CL is the OpenCL version of [https://github.com/fangq/mcx Monte Carlo eXtreme (MCX)],
a GPU-accelerated Monte Carlo simulator of light transport in 3D turbid media such as
biological tissues. Where MCX needs an NVIDIA (CUDA) or AMD (HIP) GPU, MCX-CL runs on
almost any processor with an OpenCL driver: NVIDIA, AMD and Intel GPUs, integrated
graphics, Apple silicon, and multi-core CPUs. It reads the same input files, accepts
the same MATLAB/Python inputs and writes the same outputs as MCX, and for a given seed
it traces the same photon paths.

MCX-CL can be used in four ways:

<pre>
`mcxcl` command line
    What it is: a standalone executable reading a JSON input file
    Best for: batch jobs, clusters, scripting, reproducible inputs
MCXLAB-CL (MATLAB/Octave)
    What it is: `mcxlabcl(cfg)`, a mex function with in-memory inputs/outputs
    Best for: interactive analysis, image reconstruction, plotting
PMCXCL (Python)
    What it is: `pmcxcl.run(cfg)`, a Python module installed with `pip`
    Best for: Python/NumPy workflows, notebooks, macOS
MCX Studio (GUI)
    What it is: a graphical editor and viewer that runs mcx, mcxcl and mmc
    Best for: learning MC, designing and inspecting simulations
</pre>

== # What's New ==

MCX-CL v2026 (3.0) is a milestone release that brings MCX-CL to near feature
parity with CUDA-based MCX, and makes its ray-tracing results verifiable
step-by-step against MCX. Notable features ported to MCX-CL in this release
include

* split-voxel MC (SVMC) for hybrid voxel/surface domains (MCX v2021.2)
* polarized light simulations (MCX v2023)
* mua/mus/g/n 4-float media format (MCX v2024.2)
* RF replay, RF forward and RF/CW adjoint Jacobians, including the
  adjoint_dcoeff/mus/musp and dual Jacobian outputs, saving both forward
  and Jacobian in the adjoint mode
* wedge/fan line sources and slit sources with Gaussian broadening
* international language support (`--lang`)
* 1x1x1 volumes, and `.jdt`/`.jdb` detected photon and trajectory files
  with separate `srcid` records
* per-thread random number seeds derived exactly as in MCX, so that MCX and
  MCX-CL trace the same photon paths for the same seed
* exact output sums: `make fixed` accumulates the output as 64-bit fixed-point
  integers instead of a float with a second float buffer (MCX v2026); the sums
  do not depend on the order of the additions, so a run with `-A 0` is
  bit-reproducible. Each sum is a pair of 32-bit words, low and high words in
  separate planes, so only 32-bit atomics are needed and a deposit touches
  4 bytes per element, as the float buffers do. The speed is about that of the
  default build: 1.06x on an RTX 4090, 1.01x on an A100, 0.99x on a TITAN V and
  0.97x on a CPU (POCL), from 0.89x to 1.44x over 13 built-in examples

In addition, this release contains the following fixes and improvements

* 1.2-1.5x faster simulations on OpenCL devices, from dynamic per-block photon
  dispatch, 1-byte label volumes and `-cl-unsafe-math-optimizations`; `-A 0`
  keeps the static photon split and strict math for reproducible trajectories
* fixed an off-by-one photon index in wp/dcs/wptof/rfmus replay that used the
  time-of-flight and weight of the next photon; wp/dcs/jacobian/wltof replay
  outputs now agree with MCX to ~1e-3 (relative L2)
* ported the MCX replay normalization fixes (`-U 2`, time-gate clamping,
  SVMC mus division, ns time-of-flight accumulation, double-buffering)
* fixed all-zero RF replay (`-O r`, `-O s`) output; RF and rfmus replay now
  agree with MCX (relative L2 4e-5 and 2e-3)
* fixed adjoint Jacobian outputs (`-O A/D/U/V/W/Q`) on the command line and a
  segfault reading JSON detectors with a `Dir` field; adjoint detector sources
  are now launched as disks so that `detdir(4)` shapes the detector beam
* pattern/pattern3d sources now work in wp/dcs/wptof/rfmus replay and carry
  per-pattern weights in the RF imaginary parts; fixed a segfault with
  multi-pattern pattern3d sources
* fixed a GPU memory leak on NVIDIA devices (#66), forcing g=1 in continuous
  media (#65), a macOS OpenCL runtime error (#61) and a double-free in
  `mcx_clearfluence`
* zmat is upgraded to 1.0.0 (miniz 3.1.1, lz4 1.10.0)
* new step-by-step trajectory tests (`test/testtraj.sh`) compare MCX-CL
  against the MCX golden trajectories, and run, together with `testmcx.sh`,
  on a CPU OpenCL device (POCL) in the CI with `-A 0 -t <photons> -T 1`
* polarized simulations now match MCX: the Mie scattering matrix is tabulated at
  1000 angles as in MCX instead of 181 (the coarse table over-weighted forward
  scattering), the source Stokes vector (`Optode.Source.IQUV`) is read from JSON
  input, and Stokes trajectories (`--trajstokes 1`) are sized and saved correctly
* `--maxvoidstep` and `--trajstokes` are accepted on the command line as in MCX
* the host-generated RNG seeds had ~75% of their bits set on Linux and macOS, which
  biased the first photon of each thread; seeds are now uniform (the random streams
  for a given seed change, replay of saved seeds is unaffected)
* pmcxcl wheels are now built for Linux aarch64 (#67), and the macOS wheels are tagged
  with their actual architecture (`arm64` or `x86_64`) instead of `universal2`, which
  made pip install an x86_64 binary on Apple silicon running macOS 15+ (#60);
  `pmcxcl.mcxlab()` runs the simulation with pmcxcl, and a failed import reports the
  actual error (#60)
* the released Linux binaries and mex files are built in a manylinux2014 container and
  run on glibc 2.17 or newer; Octave mex files for Windows and macOS are built against
  Octave 10 (`liboctmex`)
* unit tests and coverage: host-side unit tests (`make unittest`, 570 checks) raise the
  line coverage of the host code to about 95%, and the 141 built-in example tests of MCX
  run with MCX-CL against golden values; the bugs these tests found were fixed
* fuzzing: four rounds of JSON-aware mutation fuzzing of the input files and of the
  command-line options, run on AddressSanitizer builds of mcx, mcxcl and mmc; every
  crash, hang and memory error they found was fixed; the fuzzer is now part of MCX-CL
  (`test/fuzz/dfuzz.py`, seeded by MCX's built-in inputs, now in `test/builtin`), and six
  more rounds of 300 cases on the GPU found the five bugs listed below
* hardening: JSON arrays are length-checked before use; JData arrays must carry valid
  annotations and decode to their declared size; optical properties, time gates, SrcNum,
  the voxel size (`-u`), the roulette threshold (`-e`), device workloads (`-W`), media
  formats (`-K`) and compression methods (`-Z`) are validated; invalid input now stops
  with an error message instead of crashing or hanging
* fixes found by fuzzing: the step length is no longer divided by a zero effective
  scattering coefficient (mus=0, or g=1 once `--gscatter` is reached), which hung the
  kernel; `-e` of 1 or more no longer relaunches photons forever; RF forward output now
  keeps its imaginary part in .jnii/.bnii files; a number given for a string field no
  longer crashes the reader; a source pattern smaller than Param1/Param2 is no longer read
  past its end, and one with no non-zero pixel in its sampled area no longer hangs the
  kernel; svmc labels beyond the media are rejected instead of reading past the property
  table; a run that detects no photon no longer frees the detected-photon buffer
* `Session.OutputType` in JSON inputs accepts the names mcxlabcl and pmcxcl use ("flux",
  ...); only the first letter used to be read, so "flux" gave fluence
* `-r` now repeats or splits the photons as documented; a time step of 0 is treated as
  a single time gate; `--dumpjson`, `mcx2json` and `json2mcx` round-trip all settings
* fixed the svmc mismatch with MCX, a hang with interior label-0 regions, and the
  `cl_ext_float_atomics` kernel build on Intel Arc GPUs
* pmcxcl is bumped to 0.7.5

We would like to thank @hanno-gmg (#60) and @ptrhbt (#67) for reporting issues.

The most important changes since v2025.10 are summarized below; `ChangeLog.txt`
lists all of them, with their commit hashes.

'''Features ported from MCX'''
* [7c7c6d5] split-voxel MC (SVMC); [00fcf09] split-voxel sides and exits tracked as in MCX
* [68f3211] polarized light simulations; [b3af30b] Mie matrix tabulated at 1000 angles as in MCX
* [8b05259] the mua/mus/g/n 4-float (`asgn_float`) media format
* [5756a54, 7e92747] RF replay; [00bdfae, bcd8067] RF forward and RF/CW adjoint Jacobians, including
  the dcoeff/mus/musp and dual outputs; [0f692be] forward fluence and Jacobians both returned
* [4d53ce3] wedge/fan line sources and slit sources with Gaussian broadening
* [5c2816e] international language support (`--lang`)
* [e0b526a] 1x1x1 volumes
* [0b17a15] detected photons and trajectories saved as `.jdt`/`.jdb`, with separate srcid records
* [95b9dd1] per-thread RNG seeds derived exactly as in MCX: the same seed gives the same photon paths
* [1d2af50] `make fixed`: exact, order-independent 64-bit fixed-point output sums, as in MCX
* [2a14c42] pattern sources in wp/dcs/wptof/rfmus replay; [a2bc0a4] per-pattern RF imaginary parts
* [4fa20b7] `--maxvoidstep` and `--trajstokes`; [c06cdf0] `issave2pt`, `flog` and `lang` in
  mcxlabcl/pmcxcl, as in mcxlab/pmcx

'''Performance'''
* [8680d3f] dynamic per-block photon dispatch, 1-byte label volumes and the unsafe-math JIT
  option in autopilot mode: 1.2-1.5x faster
* [2964165] parallel zlib/gzip compression of JData outputs, with a block index for parallel decoding

'''Bug fixes'''
* [7bb12a9] off-by-one photon index in wp/dcs/wptof/rfmus replay; replay now agrees with MCX
* [7e92747] all-zero RF replay (`-O r`, `-O s`) output
* [57545b0] adjoint outputs (`-O A/D/U/V/W/Q`) on the command line; detector JSON with `Dir`
* [d7ea87b] `-U 2` skips only the detected-weight normalization in replay; [b5d3b0a] double
  preprocessing with `-j`
* [8c9c3f0] GPU memory leak on NVIDIA devices (#66)
* [0ad93bb] host RNG seeds had ~75% of their bits set; now uniform
* [5acc84f] photons entering label 0 end as in MCX, fixing a hang with interior label-0 regions
* [5e139f0] `cl_ext_float_atomics` kernels built as OpenCL C 3.0, fixing Intel Arc GPUs
* [7b8f5f2] strict math is used when saving or replaying seeds
* [00d55fc] `-r` repeats or splits the photons as documented; [25ea487] a time step of 0 is one gate
* [614e647] RF forward output keeps its imaginary part in `.jnii`/`.bnii` files
* [04a7734] `Session.OutputType` names read in full, not just their first letter
* [e14dbef, ef4ec6b, aca5a92] `--dumpjson`, mcx2json and json2mcx round-trip all settings

'''Testing and robustness'''
* [253b22f] step-by-step trajectory tests against MCX's golden data, run on a CPU (POCL) in CI
* [7fe8bea] host unit tests and short GPU feature tests; [c9de165] host coverage about 95%
* [55da8c8] four rounds of fuzzing of inputs and options: all crashes, hangs and memory errors
  fixed [70ef956, ef650f2]; input values now validated
* [73a7aea] `test/fuzz/dfuzz.py` and the built-in inputs of MCX in `test/builtin`; six more rounds
  of fuzzing found five bugs shared with MCX, fixed in [7c1fcba]

'''Packaging'''
* [a3aefe2] Linux releases built in manylinux2014 (glibc 2.17+); Octave 10 mex files for macOS/Windows
* pmcxcl wheels for Linux aarch64 (#67); [ee9125f] macOS wheels tagged with their real
  architecture (#60); [b429887] `pmcxcl.mcxlab` runs with pmcxcl
* [65dfa31] bundled zmat updated to 1.2.0; [87dd73e] cJSON updated to 1.7.19

== # Introduction ==

Monte Carlo eXtreme simulates how light propagates in 3D heterogeneous, scattering
media by tracing a large number of photon packets through a voxelated domain. Each
packet is attenuated by absorption, randomly redirected by scattering, and
reflected or refracted at interfaces between media; accumulating their weights gives
a statistically exact solution of the radiative transfer equation (RTE), the model
behind diffuse optics, fluorescence imaging, photobiomodulation dosimetry and many
other biophotonics applications. Tracing the photons in parallel on a GPU makes this
hundreds to thousands of times faster than a single-threaded CPU code.

MCX-CL implements the MCX algorithm in OpenCL [Yu2018], so the same simulation runs on
GPUs and CPUs from any vendor. Since v2026, MCX-CL supports nearly all MCX features and
traces the same photon paths as MCX for the same seed; the differences are listed in
MCX-CL and MCX: differences.

'''What MCX-CL can do'''

* arbitrary 3D heterogeneous media given as label volumes, continuously varying
    optical properties, or built from simple shapes; 2D simulations
* time-resolved, continuous-wave (CW) and frequency-domain (RF) simulations
* 18 source types, from pencil beams to wide-field patterns and spatial-frequency
    sources; many sources in one run
* disk detectors and bounding-box faces as detectors; per-photon partial path
    lengths, scattering counts, momentum transfer, exit positions/directions
* Fresnel reflection, total absorption, mirror and cyclic boundary conditions
* Jacobians by photon replay or by the adjoint method, in CW, time and frequency
    domain
* photon sharing, polarized light, split-voxel MC (SVMC), custom phase functions
* several devices - even from different vendors - in one simulation
* JSON-based inputs and JData/NeuroJSON-based outputs

'''The MCX family'''

<pre>
MCX (https://github.com/fangq/mcx)
    Description: voxel-based MC in CUDA (NVIDIA) or HIP (AMD); MCXLAB, PMCX and MCX Studio
MCX-CL (https://github.com/fangq/mcxcl)
    Description: this package: the OpenCL version of MCX; MCXLAB-CL (`mcxlabcl`) and PMCXCL
        (`pmcxcl`)
MMC (https://github.com/fangq/mmc)
    Description: mesh-based MC on tetrahedral meshes, on CPUs, OpenCL, CUDA and OptiX;
        MMCLAB and PMMC
MCX Cloud (https://mcx.space/cloud)
    Description: run MCX and MMC in a web browser on cloud GPUs
Redbird (https://github.com/fangq/redbird) / Redbirdpy (https://github.com/fangq/redbirdpy)
    Description: diffuse optical tomography solvers that can use MCX/MMC as the forward
        model
</pre>

Scripts move between MCX and MCX-CL by changing `mcxlab` to `mcxlabcl`, or `pmcx` to
`pmcxcl`; in MATLAB, `mcxlab(cfg, 'opencl')` also runs MCX-CL.

== # Installation and configuration ==

=== What you need ===

MCX-CL needs an '''OpenCL runtime''', the part of a graphics or CPU driver that runs
OpenCL programs (`OpenCL.dll` on Windows, `libOpenCL.so` on Linux, built into macOS).
No SDK or compiler is needed: MCX-CL compiles its kernel for your device when it
starts.

<pre>
NVIDIA GPU
    OpenCL runtime: included in the NVIDIA driver; CUDA is not needed
AMD GPU
    OpenCL runtime: included in the AMD Adrenalin driver (Windows); the ROCm OpenCL runtime
        (Linux)
Intel GPU (integrated or Arc)
    OpenCL runtime: included in the Intel graphics driver (Windows); `intel-opencl-icd`
        (Linux)
Apple silicon / Intel Mac
    OpenCL runtime: built into macOS
CPU
    OpenCL runtime: POCL (`pocl-opencl-icd` on Debian/Ubuntu, works on any x86_64 or ARM
        CPU), or Intel's CPU runtime
</pre>

On Linux, an ICD loader (`ocl-icd-libopencl1` on Debian/Ubuntu) dispatches OpenCL
calls to the installed runtimes, and `clinfo` lists the devices they provide. Several
runtimes can be installed together, for example a GPU driver and POCL.

As a rough guide, high-end dedicated GPUs are 20-60x faster than a multi-core CPU, and
integrated GPUs 3-4x; see https://mcx.space/computebench/ for measured speeds.

=== Getting MCX-CL ===

<pre>
MCX Studio all-in-one package (mcx, mcxcl, mmc, the GUI and the MATLAB/Octave toolboxes)
    How to get it: https://mcx.space/wiki/?Get or the MCX releases
        (https://github.com/fangq/mcx/releases); on Windows also the installer or `winget
        install coti.mcxstudio`
mcxcl command line or MCXLAB-CL alone
    How to get it: https://mcx.space/wiki/?Get or https://github.com/fangq/mcxcl/releases
PMCXCL (Python)
    How to get it: `python3 -m pip install pmcxcl pmcx`
nightly builds
    How to get it: https://mcx.space/nightly/
source code
    How to get it: `git clone https://github.com/fangq/mcxcl.git`
</pre>

PMCXCL wheels are built for Linux (x86_64 and aarch64), macOS (arm64 and x86_64) and
Windows; `pmcx` provides the utility functions PMCXCL uses. The released Linux binaries
and mex files run on glibc 2.17 or newer; the Octave mex files for Windows and macOS
need Octave 10 or newer.

=== The MCX Studio package ===

The all-in-one package unzips into one folder:

<pre>
MCXStudio/
├── mcxstudio            the GUI (MCXStudio.app on macOS)
├── mcxsuite_addpath.m   adds all the MATLAB/Octave toolboxes to the path
├── MATLAB/              mcxlab, mcxlabcl, mmclab, iso2mesh and redbird
└── MCXSuite/            mcx, mcxcl and mmc, each with its bin/ folder and helper scripts
</pre>

Add `MCXSuite/mcxcl/bin` (and `mcx/bin`, `mmc/bin`) to your `PATH`. In MATLAB or
Octave, `cd` into the `MCXStudio` folder, run `mcxsuite_addpath`, and `savepath` to keep
the paths for later sessions. On macOS, unzip the package under your home folder.

=== Checking your devices ===

Unzip the package, add its `bin/` folder to your `PATH`, and run

<pre>
mcxcl -L              # list the OpenCL devices
mcxcl -Q cube60       # run the "cube60" benchmark on the first device
</pre>

`mcxcl -L` prints each platform (runtime) and its devices, numbered from 1 across all
platforms:

<pre>
Platform [0] Name NVIDIA CUDA
============ GPU device ID 0 [1 of 2]: NVIDIA TITAN V  ============
 Device 1 of 2:		NVIDIA TITAN V
 Compute units   :	80 core(s)
 Global memory   :	12625903616 B
 ...
 Auto-thread    :	327680
 Auto-block     :	64
</pre>

Use the device number with `-G`: `-G 2` runs on the 2nd device, and `-G 0110` on the
2nd and 3rd together. If no device is listed, the OpenCL runtime is missing or broken:
install or update the driver (or POCL for a CPU) and check with `clinfo`. Without
`clinfo`, you can check that the runtime library exists:

* Windows: `OpenCL.dll` in `C:\Windows\System32`;
* macOS: `/System/Library/Frameworks/OpenCL.framework` exists;
* Linux: `ldconfig -p | grep libOpenCL` lists `libOpenCL.so.1`; if the library is in
    a non-standard folder, add that folder to `LD_LIBRARY_PATH`, and check that the
    vendor's `.icd` file is in `/etc/OpenCL/vendors`.

Intel CPUs older than Ivy Bridge (3rd generation Core, 2012) are not well supported by
Intel's OpenCL runtimes; POCL is an alternative for them.

=== Setting up MCXLAB-CL and PMCXCL ===

<pre>
addpath('/path/to/mcxlabcl');     % mcxcl.mex*, mcxlabcl.m
addpath('/path/to/mcxcl/utils');  % helper functions
info = mcxlabcl('gpuinfo')        % list the devices
</pre>

<pre>
python3 -m pip install pmcxcl pmcx jdata bjdata matplotlib
python3 -c "import pmcxcl; print(pmcxcl.gpuinfo())"
</pre>

The MATLAB helpers in `mcxcl/utils` are a subset of those of MCX; for the full set
(`mcxpreview`, `mcxplotshapes`, `mcxsvmc`, ...), also add MCX's `utils` folder, which
works with MCXLAB-CL's outputs.

=== Operating system notes ===

'''Windows''': Windows resets a GPU that is busy for more than a few seconds. The MCX
Studio installer raises this limit; with the zip packages, right-click
`setup/win64/apply_timeout_registry_fix.bat`, choose '''Run as administrator''', and
reboot.

'''Linux laptops with hybrid NVIDIA graphics''': if the screen freezes when MCX-CL runs,
use the NVIDIA GPU as the primary GPU (`sudo prime-select nvidia`). If simulations fail
after the laptop wakes up from suspension, close MATLAB/Python and run
`sudo rmmod nvidia-uvm; sudo modprobe nvidia-uvm`.

'''macOS''': Apple's OpenCL runs MCX-CL on Apple silicon and Intel Macs; PMCXCL is the
way to run simulations from Python on a Mac. MCX Studio is provided as a universal
`MCXStudio.app`.

=== Building from source ===

<pre>
git clone https://github.com/fangq/mcxcl.git
cd mcxcl/src
make                  # the mcxcl binary in ../bin
make mex              # MCXLAB-CL for MATLAB: ../mcxlabcl/mcxcl.mex*
make oct              # MCXLAB-CL for Octave: ../mcxlabcl/mcxcl.mex
make unittest         # host unit tests
make fixed            # exact 64-bit fixed-point output sums (-DUSE_FIXEDPOINT)
cd ../pmcxcl && python3 -m pip install .   # PMCXCL
</pre>

Building needs a C/C++ compiler and the OpenCL headers and loader (`opencl-headers`
and `ocl-icd-opencl-dev` on Debian/Ubuntu); `make oct` needs `liboctave-dev`, and the
Python module needs `cmake`. The kernel source, `src/mcx_core.cl`, is embedded in the
binary.

== # Quick start: one simulation, four ways ==

The same simulation - a pencil beam entering a 60x60x60 mm homogeneous cube with four
detectors - in each mode.

'''Command line.''' Save this as `qtest.json`:

<pre>
{
    "Domain": {
        "Dim": [60, 60, 60],
        "OriginType": 1,
        "LengthUnit": 1,
        "Media": [
            {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
            {"mua": 0.005, "mus": 1.0, "g": 0.01, "n": 1.37}
        ]
    },
    "Shapes": [
        {"Grid": {"Tag": 1, "Size": [60, 60, 60]}}
    ],
    "Session": {
        "Photons": 1000000,
        "RNGSeed": 29012392,
        "ID": "qtest"
    },
    "Forward": {
        "T0": 0.0e+00,
        "T1": 5.0e-09,
        "Dt": 5.0e-10
    },
    "Optode": {
        "Source": {
            "Type": "pencil",
            "Pos": [29.0, 29.0, 0.0],
            "Dir": [0.0, 0.0, 1.0]
        },
        "Detector": [
            {"Pos": [29.0, 19.0, 0.0], "R": 1.0},
            {"Pos": [29.0, 39.0, 0.0], "R": 1.0},
            {"Pos": [19.0, 29.0, 0.0], "R": 1.0},
            {"Pos": [39.0, 29.0, 0.0], "R": 1.0}
        ]
    }
}
</pre>

and run

<pre>
mcxcl -f qtest.json -G 1
</pre>

MCX-CL writes `qtest.jnii` (the fluence rate, 60x60x60 voxels x 10 time gates) and
`qtest_detp.jdt` (the detected photons).

'''MATLAB/Octave:'''

<pre>
cfg.nphoton = 1e6;
cfg.vol = uint8(ones(60, 60, 60));
cfg.prop = [0 0 1 1; 0.005 1 0.01 1.37];   % [mua(1/mm), mus(1/mm), g, n] per label
cfg.issrcfrom0 = 1;
cfg.srcpos = [29 29 0];
cfg.srcdir = [0 0 1];
cfg.detpos = [29 19 0 1; 29 39 0 1; 19 29 0 1; 39 29 0 1];
cfg.tstart = 0; cfg.tend = 5e-9; cfg.tstep = 5e-10;
cfg.gpuid = 1;                              % device number from mcxlabcl('gpuinfo')
[flux, detp] = mcxlabcl(cfg);
imagesc(log10(squeeze(sum(flux.data(:, 30, :, :), 4))));   % CW fluence rate at y=30
</pre>

'''Python:'''

<pre>
import numpy as np, pmcxcl, matplotlib.pyplot as plt
cfg = {
    'nphoton': 1000000,
    'vol': np.ones([60, 60, 60], dtype='uint8'),
    'prop': [[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]],
    'issrcfrom0': 1,
    'srcpos': [29, 29, 0], 'srcdir': [0, 0, 1],
    'detpos': [[29, 19, 0, 1], [29, 39, 0, 1], [19, 29, 0, 1], [39, 29, 0, 1]],
    'tstart': 0, 'tend': 5e-9, 'tstep': 5e-10,
    'gpuid': 1,
}
res = pmcxcl.mcxlab(cfg)
plt.imshow(np.log10(res['flux'][:, 29, :, :].sum(axis=2)))
plt.show()
</pre>

'''GUI:''' start `mcxstudio`, open `qtest.json`, choose the `intel`, `amd` or `cpu`
processor (which run mcxcl), and press '''Run'''.

== # Key concepts ==

'''Grid and units.''' The domain is a grid of Nx*Ny*Nz cubic voxels of edge length
`LengthUnit` (JSON), `-u` (command line) or `cfg.unitinmm` (MATLAB/Python), in mm.
Positions of sources and detectors are given in ''voxel units''. With
`OriginType: 1` (`-z 1`, `cfg.issrcfrom0=1`), the corner of the first voxel is at
[0,0,0]; otherwise (the default in the command line and MCXLAB-CL) it is at [1,1,1],
matching MATLAB's 1-based indices. Absorption (mua) and scattering (mus)
coefficients are in 1/mm regardless of the voxel size.

'''Media.''' Each voxel holds a medium label. The optical properties of label i are
the (i+1)-th row of `Media` (`cfg.prop`): `[mua, mus, g, n]`, with mus the scattering
(not reduced scattering) coefficient, mus=mus'/(1-g). Label 0 is the background
(outside of the domain): a photon moving from a non-zero voxel into a 0 voxel
leaves the domain. Instead of labels, a voxel can also store its own optical
properties, see Volume and media formats.

'''Time gates.''' A simulation covers the time window [T0, T1], divided into gates of
width Dt; the output has one volume per gate. Photons are tracked until T1, so for
a CW solution choose T1 long enough for the light to decay (a few ns in tissue),
and sum the gates.

'''Photons and statistics.''' Results are averages over the simulated photon
packets; their noise falls as 1/sqrt(number of photons). 10^6-10^8 photons are
typical. Russian roulette (`-e`) terminates packets of low weight without biasing
the result.

'''Normalization.''' By default the output is normalized to a unit source, so MCX
produces the Green's function of the fluence rate (1/(mm^2 s), or W/mm^2 per watt
of source power), directly comparable with analytical solutions. `-U 0`
(`cfg.isnormalized=0`) returns the raw accumulated weights.

'''Boundaries.''' At the domain boundary, photons are reflected according to the
Fresnel equations when the refractive indices differ (`-b 1`, the default), or
leave the domain (`-b 0`). `-B`/`cfg.bc` sets each face separately, including
mirror and cyclic boundaries.

== # Running the mcxcl command line ==

A simulation needs a JSON input file (`-f`) or a built-in benchmark (`-Q`).
Command-line options override the corresponding settings in the file:

<pre>
mcxcl -f input.json [options]
mcxcl -Q cube60 -n 1e7 -G 1
</pre>

`mcxcl` (or `mcxcl -h`) prints all options. They are the same as those of `mcx`,
except for the device options; the most useful ones are below.

'''Input and simulation control'''

<pre>
`-f file`, `--input`
    Meaning: read a JSON input file; a string starting with `{` is read as inline JSON, `-f
        -` reads stdin
`-Q name`, `--bench`
    Meaning: run a built-in benchmark (`-Q` alone lists them): cube60, cube60b,
        cube60planar, cubesph60b, onecube, twocube, skinvessel, sphshells, spherebox,
        colin27
`-N name`, `--net`
    Meaning: download and run a simulation shared on NeuroJSON.io
        (https://neurojson.io/db/mcx)
`-j '{...}'`, `--json`
    Meaning: modify any setting of the input with a JSON fragment
`-P '{...}'`, `--shapes`
    Meaning: add shapes to the domain, see Shape constructs
`-n num`, `--photon`
    Meaning: number of photons
`-r num`, `--repeat`
    Meaning: repeat the simulation num times; negative: split the photons into |num| runs
`-E seed`, `--seed`
    Meaning: RNG seed; a `_detp.jdt` file name replays its detected photons
`-u mm`, `--unitinmm`
    Meaning: voxel edge length in mm
`-z 0/1`, `--srcfrom0`
    Meaning: 1: the first voxel's corner is [0,0,0]; 0: [1,1,1]
`-K fmt`, `--mediabyte`
    Meaning: volume data format, see Volume and media formats
`--gscatter n`
    Meaning: after n scattering events, scatter isotropically (faster)
</pre>

'''Physics, boundaries and output'''

<pre>
`-b 1/0`, `--reflect`
    Meaning: Fresnel reflection at the domain boundary (1) or none (0)
`-B '______'`, `--bc`
    Meaning: per-face boundary conditions: `r` Fresnel, `a` absorbing, `m` mirror, `c`
        cyclic, `_` use `-b`; 6 more digits mark faces as detectors
`-V 0/1`, `--specular`
    Meaning: include the specular reflection when the source is outside the domain
`--voidtime 1/0`
    Meaning: for a source outside the domain, start the clock at launch (1) or at entry (0)
`-e w`, `--minenergy`
    Meaning: Russian roulette threshold
`-U 1/0/2`, `--normalize`
    Meaning: normalize the output (1), raw (0), or in replay skip the detected-weight
        normalization (2)
`-s name`, `--session`
    Meaning: name of the output files
`-O X`, `--outputtype`
    Meaning: the output quantity, see Output quantities
`-F fmt`, `--outputformat`
    Meaning: `jnii` (default), `bnii`, `nii`, `hdr`, `mc2`, `tx3`
`-Z zlib`, `--zip`
    Meaning: compression for jnii/bnii: `zlib`, `gzip`, `base64`, `lzip`, `lzma`, `lz4`,
        `lz4hc`
`-d 1/0/3`, `--savedet`
    Meaning: save detected photons (1), not (0), or stop when `-H` photons are detected (3)
`-w DP`, `--savedetflag`
    Meaning: detected photon data to save, see Detectors
`-H num`, `--maxdetphoton`
    Meaning: the most detected photons kept
`-x 1` / `-X 1` / `-m 1` / `-q 1`
    Meaning: save exit positions and directions / diffuse reflectance / momentum transfer /
        RNG seeds
`-S 1/0`, `--save2pt`
    Meaning: save the volume output (1) or skip it (0)
`--dumpjson [file]`
    Meaning: print (or save) the complete input, volume included, as one JSON file
`-D P`, `--debug`
    Meaning: `P` progress bar, `M` save trajectories, `T` trajectories only, `R` RNG test
`-y zh_CN`, `--lang`
    Meaning: print messages in another language
</pre>

'''Devices and the OpenCL kernel'''

<pre>
`-L`, `--listgpu`
    Meaning: list the OpenCL devices and exit
`-G n` or `-G '0110'`, `--gpu`
    Meaning: use device n (from 1, as numbered by `-L`), or several devices given as a 0/1
        mask
`-W '50,30,20'`, `--workload`
    Meaning: relative share of photons for each selected device
`-A 1/0`, `--autopilot`
    Meaning: choose the thread count, dispatch photons dynamically and use fast math (1,
        default), or use `-t`/`-T` with strict math (0)
`-t num`, `--thread` / `-T num`, `--blocksize`
    Meaning: total threads and threads per work group when `-A 0`
`-o 3`, `--optlevel`
    Meaning: kernel optimization level, 0 (none) to 3
`-J '-D...'`, `--compileropt`
    Meaning: extra OpenCL compiler options, e.g. `-DMCX_GPU_DEBUG` (print each step) or
        `-cl-nv-verbose` (register counts on NVIDIA)
`-k file.cl`, `--kernel`
    Meaning: run your own copy of the OpenCL kernel instead of the built-in one
`--showkernel 1`
    Meaning: print the built-in (or `-k`) kernel source
`--atomic 1/0`
    Meaning: accumulate the output with atomic operations (1, default)
`--internalsrc 1`
    Meaning: the source is inside the domain: skip the search for the entry point
`-I`, `--printgpu`
    Meaning: print the device information, then run
</pre>

'''Recipes'''

<pre>
mcxcl -L                                          # which devices do I have?
mcxcl -Q cube60b -n 1e8 -G 0111 -W 40,40,20       # three devices, unequal shares
mcxcl -f input.json -s run1 -F bnii -Z lz4        # compressed binary output
mcxcl -Q cube60 -A 0 -t 4096 -T 64 -E 1234        # bit-wise reproducible run
mcxcl -Q cube60 --showkernel 1 > mykernel.cl      # get the kernel, edit it, then
mcxcl -Q cube60 -k mykernel.cl                    # ... run the edited kernel
mcxcl -f input.json -E run1_detp.jdt -O J         # replay run1's detected photons for a Jacobian
</pre>

== # JSON input file format ==

An MCX/MCX-CL input file is a JSON object with four required sections, `Domain`,
`Session`, `Forward` and `Optode`, and an optional `Shapes` section. Field names are
case sensitive; the order of fields does not matter, and unknown sections (such as
`Help`) are ignored. A field can also be written in the root object with a dotted
name, e.g. `"Session.Photons": 1e7`. Command-line options take precedence over the
file, and `-j` edits any field without touching the file.

The fields MCX-CL reads:

'''`Domain`''' - the grid and its media

<pre>
`VolumeFile`
    Meaning: a binary volume file (format set by `MediaFormat` or `-K`), a JSON/JNIfTI
        (`.json`, `.jnii`, `.bnii`) volume, or a shape file; omit it when `Shapes` builds
        the volume
`Dim`
    Meaning: [Nx, Ny, Nz]
`MediaFormat`
    Meaning: voxel format of the volume, `"byte"` by default (see below)
`Media`
    Meaning: array of `{"mua":..., "mus":..., "g":..., "n":...}`, the first for label 0
`LengthUnit`
    Meaning: voxel size in mm (`-u`)
`OriginType`
    Meaning: 1: first voxel's corner at [0,0,0]; 0: at [1,1,1] (`-z`)
`MieScatter`
    Meaning: polarized MC: array of `{"mua", "radius" (micron), "rho" (1/micron^3), "nsph",
        "nmed"}`, one per label from 1
`InverseCDF`
    Meaning: a user-defined scattering phase function, the inverse CDF of cos(theta)
`AngleInverseCDF`
    Meaning: a user-defined launch-angle distribution (also accepted under `Optode.Source`)
</pre>

'''`Session`''' - what to compute and save

<pre>
`ID`
    Meaning: name of the output files (`-s`)
`Photons`
    Meaning: number of photons (`-n`)
`RNGSeed`
    Meaning: RNG seed (`-E`)
`OutputType`
    Meaning: output quantity: `"flux"`, `"fluence"`, `"energy"`, `"jacobian"`, ... or the
        `-O` letter
`OutputFormat`
    Meaning: `"jnii"`, `"bnii"`, `"nii"`, `"hdr"`, `"mc2"`, `"tx3"` (`-F`)
`DoMismatch`
    Meaning: Fresnel reflection at the boundary (`-b`)
`BCFlags`
    Meaning: per-face boundary conditions (`-B`)
`DoNormalize`
    Meaning: normalize the output (`-U`)
`DoSaveVolume`
    Meaning: save the volume output (`-S`)
`DoPartialPath`
    Meaning: save detected photons (`-d`)
`SaveDataMask`
    Meaning: detected photon data to save, e.g. `"DPX"` (`-w`)
`DoSaveExit`, `DoSaveRef`, `DoSaveSeed`, `DoDCS`
    Meaning: as `-x`, `-X`, `-q` and `-m`
`DoSpecular`
    Meaning: specular reflection at entry (`-V`)
`DoAutoThread`
    Meaning: autopilot (`-A`)
`MaxDetPhoton`
    Meaning: the most detected photons kept (`-H`)
`MinEnergy`
    Meaning: Russian roulette threshold (`-e`)
`DebugFlag`
    Meaning: debug flags such as `"P"` (`-D`)
`RootPath`
    Meaning: folder of the input files (`--root`)
</pre>

'''`Forward`''' - time window: `T0` (start), `T1` (end) and `Dt` (gate width), in seconds.

'''`Optode`''' - sources and detectors

<pre>
`Source.Type`
    Meaning: source type, see Sources
`Source.Pos`
    Meaning: [x, y, z] in voxel units
`Source.Dir`
    Meaning: [vx, vy, vz] unit direction, optionally a 4th value: focal length (`nan`
        isotropic, `-inf` Lambertian)
`Source.Param1`, `Source.Param2`
    Meaning: 4 numbers each, depending on `Type`
`Source.Pattern`
    Meaning: the pattern of `pattern`/`pattern3d` sources: `{"Nx", "Ny", "Nz", "Data"}`, or
        a JData array
`Source.SrcNum`
    Meaning: number of patterns simulated together (photon sharing)
`Source.Weight`
    Meaning: initial photon weight
`Source.Frequency`
    Meaning: modulation frequency (Hz) for frequency-domain (RF) outputs
`Source.IQUV`, `Source.WaveLength`
    Meaning: Stokes vector and wavelength (nm) for polarized MC
`Detector`
    Meaning: array of `{"Pos": [x,y,z], "R": radius}`, and `"Dir"` for the adjoint outputs
</pre>

`Optode.Source` can also be an array of sources, each with its own `Type`, `Pos`,
`Dir` and parameters. Several examples are in the `example/` folder, for example
`example/quicktest`, `example/skinvessel` and `example/multisrc`. `mcxcl --dumpjson`
prints the complete input of any simulation, including the defaults, which is a
good template.

Legacy files: `mcx2json.m` and `json2mcx.m` (MATLAB) or `pmcxcl.mcx2json` /
`pmcxcl.json2mcx` convert between an MCXLAB-CL/PMCXCL `cfg` and a JSON input file, so a
simulation set up in MATLAB or Python can be run with the command line, and vice
versa.

== # Shape constructs ==

Instead of supplying a volume file, the domain can be built from simple shapes. The
`Shapes` array is processed in order, each shape overwriting the voxels it covers,
and can be placed in the input file, in a separate `.json` file named by
`Domain.VolumeFile`, or on the command line with `-P`:

<pre>
"Shapes": [
    {"Name": "head model"},
    {"Origin": [0, 0, 0]},
    {"Grid": {"Tag": 1, "Size": [60, 60, 60]}},
    {"ZLayers": [[1, 5, 1], [6, 8, 2], [9, 60, 3]]},
    {"Sphere": {"Tag": 4, "O": [30, 30, 30], "R": 10}},
    {"Box": {"Tag": 0, "O": [10, 10, 10], "Size": [10, 10, 10]}},
    {"Cylinder": {"Tag": 2, "C0": [0, 30, 20], "C1": [60, 30, 20], "R": 3}}
]
</pre>

<pre>
`Grid`
    Parameters: `Tag`, `Size: [Nx,Ny,Nz]`
    Result: (re)creates the grid, filled with `Tag`
`Origin`
    Parameters: `[x,y,z]`
    Result: origin of the coordinates of the shapes that follow
`Sphere`
    Parameters: `Tag`, `O` (center), `R`
    Result: a sphere
`Box`
    Parameters: `Tag`, `O` (lower corner), `Size`
    Result: a box, floating-point coordinates
`Subgrid`
    Parameters: `Tag`, `O`, `Size`
    Result: a box of whole voxels, integer indices, both ends included
`Cylinder`
    Parameters: `Tag`, `C0`, `C1` (axis ends), `R`
    Result: a finite cylinder
`XLayers`, `YLayers`, `ZLayers`
    Parameters: `[[start, end, tag], ...]`
    Result: layers normal to an axis, 1-based voxel indices, ends included
`XSlabs`, `YSlabs`, `ZSlabs`
    Parameters: `Tag`, `Bound: [[start, end], ...]`
    Result: slabs normal to an axis, floating-point bounds
`UpperSpace`
    Parameters: `Tag`, `Coef: [A,B,C,D]`
    Result: the half space A*x+B*y+C*z>D
`Name`
    Parameters: a string
    Result: a comment
</pre>

`Tag` is the medium label, and must be smaller than the number of media. Coordinates
start from the corner of the first voxel, at [0,0,0] or [1,1,1] depending on
`OriginType`, unless an `Origin` is given. In MATLAB/Python, the same JSON string can
be passed as `cfg.shapes`; `mcxplotshapes.m` (MCX's `utils`) and `pmcxcl.plotshapes` draw it.
MCX's `example/shapetest` has more examples.

== # Volume and media formats ==

The domain is usually a label volume, but MCX-CL accepts several voxel formats, set by
`Domain.MediaFormat` in JSON or `-K` on the command line. In MCXLAB-CL and PMCXCL, the
format follows from the type and shape of `cfg.vol`:

<pre>
1
    `MediaFormat`: `byte`
    MCXLAB-CL/PMCXCL `cfg.vol`: 3D uint8
    Each voxel stores: a label, 0-255
2
    `MediaFormat`: `short`
    MCXLAB-CL/PMCXCL `cfg.vol`: 3D uint16
    Each voxel stores: a label, 0-65535
4
    `MediaFormat`: `integer`
    MCXLAB-CL/PMCXCL `cfg.vol`: 3D uint32 (or single/double)
    Each voxel stores: a label
101
    `MediaFormat`: `mua_float`
    MCXLAB-CL/PMCXCL `cfg.vol`: 1xNxxNyxNz single
    Each voxel stores: mua; mus, g, n from label 1
100
    `MediaFormat`: `muamus_float`
    MCXLAB-CL/PMCXCL `cfg.vol`: -
    Each voxel stores: mua and mus; g, n from label 1
102
    `MediaFormat`: `muamus_half`
    MCXLAB-CL/PMCXCL `cfg.vol`: 2xNxxNyxNz single
    Each voxel stores: mua and mus, stored as half floats
104
    `MediaFormat`: `muamus_short`
    MCXLAB-CL/PMCXCL `cfg.vol`: 2xNxxNyxNz uint16
    Each voxel stores: mua and mus as 0-65535 levels between labels 1 and 2
103
    `MediaFormat`: `asgn_byte`
    MCXLAB-CL/PMCXCL `cfg.vol`: 4xNxxNyxNz uint8
    Each voxel stores: mua, mus, g, n as 0-255 levels between labels 1 and 2
96
    `MediaFormat`: `asgn_float`
    MCXLAB-CL/PMCXCL `cfg.vol`: 4xNxxNyxNz single
    Each voxel stores: mua, mus, g, n
99
    `MediaFormat`: `labelplus`
    MCXLAB-CL/PMCXCL `cfg.vol`: 3xNxxNyxNz single
    Each voxel stores: a label, plus one property of it replaced by a value
98
    `MediaFormat`: `mixlabel`
    MCXLAB-CL/PMCXCL `cfg.vol`: 3xNxxNyxNz uint16
    Each voxel stores: two labels and the fraction of the first
97
    `MediaFormat`: `svmc`
    MCXLAB-CL/PMCXCL `cfg.vol`: 8xNxxNyxNz uint8
    Each voxel stores: split-voxel MC: two labels and the interface plane in the voxel
</pre>

Volumes with per-voxel properties (all but the label and `svmc` formats) carry no
per-medium partial path data. Binary volume files are read in column-major (MATLAB)
order unless `-a 1` is given. Examples: `demo_continuous_mua_mus.m` and the
`demo_svmc_*.m` scripts.

== # Sources ==

The source is set by its type, position (`Pos`/`srcpos`), direction (`Dir`/`srcdir`)
and two 4-element parameter vectors (`Param1`/`srcparam1`, `Param2`/`srcparam2`).
The 4th element of the direction is a focal length for the sources marked [*]:
0 for a collimated beam, positive to converge to a focus that far along the
direction, negative to diverge from a virtual point behind the source, `nan` for an
isotropic launch and `-inf` for a Lambertian (cosine) launch (3D only).

<pre>
`pencil`
    Description and parameters: a pencil beam (default)
`isotropic`
    Description and parameters: an isotropic point source
`arcsine`
    Description and parameters: like isotropic, with a uniformly distributed zenith angle
`cone`
    Description and parameters: a uniform cone; Param1[0]: half angle (rad)
`zgaussian`
    Description and parameters: a Gaussian distribution of the zenith angle; Param1[0]:
        variance
`gaussian` [*]
    Description and parameters: a collimated Gaussian beam; Param1[0]: waist radius
`hyperboloid` [*]
    Description and parameters: a hyperboloid Gaussian beam; Param1: [waist radius, distance
        to focus, Rayleigh range]
`disk` [*]
    Description and parameters: a uniform disk normal to Dir; Param1[0]: radius, Param1[1]:
        inner radius of an annulus
`ring` [*]
    Description and parameters: a ring or ring sector; Param1: [outer radius, inner radius,
        start angle, end angle]
`planar` [*]
    Description and parameters: a uniform quadrilateral with corners Pos, Pos+Param1[0:3],
        Pos+Param2[0:3]
`pattern` [*]
    Description and parameters: the `planar` area with intensities from `Pattern` (Param1[3]
        x Param2[3] pixels, values 0-1)
`pattern3d` [*]
    Description and parameters: a 3D intensity volume of size Param1[0:3] starting at Pos
`fourier` [*]
    Description and parameters: a spatial-frequency (SFDI) pattern
        0.5*[1+M*cos(2\pi\(fx*x+fy*y)+phi0)] over the `planar` area; fx, fy and phi0, M from
        Param1[3], Param2[3]
`fourierx` [*]
    Description and parameters: a general Fourier source; Param1: [v1, |v2|], Param2: [kx,
        ky, phi0, M]
`fourierx2d` [*]
    Description and parameters: a general 2D Fourier basis; Param1: [v1, |v2|], Param2: [kx,
        ky, phix, phiy]
`pencilarray`
    Description and parameters: an array of pencil beams over the `planar` area; Param1[3] x
        Param2[3] beams
`line`
    Description and parameters: a line source from Pos to Pos+Param1[0:3], emitting normal
        to the line; Param2[0]: half angle of a wedge (fan) beam
`slit` [*]
    Description and parameters: a collimated slit beam along Dir from the same segment;
        Param2[0:1]: Gaussian broadening
</pre>

'''Several sources''' are given as an array of `Optode.Source` objects, or as one row
per source in `srcpos`/`srcdir`/`srcparam1`/`srcparam2`. By default they are
simulated together into one output; `--srcid -1` (`cfg.srcid=-1`) gives each source
its own output, and `--srcid N` simulates only the N-th source. Example:
`demo_multisrc.m`.

'''Initial weight.''' A 4th element of `srcpos` (or `Source.Weight`) sets the initial
photon weight, which may be negative; the output scales linearly with it.

'''Photon sharing.''' A `pattern` source with `SrcNum`/`cfg.srcnum` > 1 simulates
many illumination patterns in a single run, each photon carrying a weight for
every pattern; the output gains a pattern dimension. Example:
`demo_photon_sharing.m` [Yan2020-OL].

'''Custom angular distributions.''' `Source.AngleInverseCDF` (`cfg.angleinvcdf`)
gives the launch angle distribution as the inverse CDF of theta/pi.
Example: `demo_mcxlab_launchangle.m`.

`mcxpreview(cfg)` (MATLAB, from MCX's `utils`) and `pmcxcl.preview(cfg)` (Python) draw the domain,
sources and detectors, which is the quickest way to check a source setup;
`demo_mcxlab_srctype.m` shows every source type.

== # Detectors and detected photon data ==

Detectors are disks given by a center and a radius (`Optode.Detector`,
`cfg.detpos = [x y z r]`); a photon leaving the domain within a detector is
recorded. Faces of the bounding box can also act as detectors, through the 7th-12th
characters of `-B`/`cfg.bc` (see `demo_bc_det.m`).

For each detected photon, MCX-CL can save the fields selected by `-w`
(`Session.SaveDataMask`, `cfg.savedetflag`), default `DP`:

<pre>
Letter  Field                                             Columns
------  ------------------------------------------------  ---------------
`D`     detector ID                                       1
`S`     scattering events in each medium                  number of media
`P`     partial path length in each medium (voxel units)  number of media
`M`     momentum transfer in each medium (for DCS)        number of media
`X`     exit position                                     3
`V`     exit direction                                    3
`W`     initial weight                                    1
`I`     Stokes vector (polarized MC)                      4
</pre>

These data make it possible to compute detector readings for ''different'' optical
properties without rerunning the simulation (perturbation Monte Carlo): the
weight of a detected photon is w0*exp(-sum(mua_i*L_i)). The helper functions do
this for you: `mcxdetweight`/`pmcxcl.detweight` (weights), `mcxmeanpath`/
`pmcxcl.meanpath` (mean partial path lengths), `mcxdettpsf`/`pmcxcl.dettpsf` (temporal
point spread function), `mcxcwdref`/`pmcxcl.cwdref` (CW reflectance) and
`mcxdcsg1`/`pmcxcl.dcsg1` (DCS autocorrelation g1).

Saving the RNG seed of each detected photon (`-q 1`, or requesting the 4th output in
MCXLAB-CL) enables photon replay, see Advanced features.

== # Output quantities, files and formats ==

=== Output quantities ===

`-O` (`Session.OutputType`, `cfg.outputtype`) selects what the volume output holds:

<pre>
`X`
    Name: `flux`
    Quantity: fluence rate, 1/(mm^2 s) per unit source (default)
`F`
    Name: `fluence`
    Quantity: fluence, 1/mm^2, integrated over each time gate
`E`
    Name: `energy`
    Quantity: energy deposited per voxel
`L`
    Name: `length`
    Quantity: total path length per voxel
`J`
    Name: `jacobian` (`wl`)
    Quantity: replay: mua Jacobian
`P`
    Name: `nscat` (`wp`)
    Quantity: replay: weighted scattering counts, for the mus Jacobian
`M`
    Name: `wm`
    Quantity: replay: weighted momentum transfer
`R`
    Name: `rf`
    Quantity: replay: frequency-domain mua Jacobian; without replay: complex fluence rate
`S`
    Name: `rfmus`
    Quantity: replay: frequency-domain mus Jacobian
`T`
    Name: `wltof`
    Quantity: replay: time of flight x path length
`B`
    Name: `wptof`
    Quantity: replay: time of flight x scattering count
`A`
    Name: `adjoint`
    Quantity: adjoint mua Jacobian
`D`
    Name: `adjoint_dcoeff`
    Quantity: adjoint diffusion coefficient Jacobian
`U`
    Name: `adjoint_mus`
    Quantity: adjoint mus Jacobian
`V`
    Name: `adjoint_musp`
    Quantity: adjoint mus' Jacobian
`W`
    Name: `adjoint_mua_d`
    Quantity: adjoint mua and D Jacobians together
`Q`
    Name: `adjoint_mua_musp`
    Quantity: adjoint mua and mus' Jacobians together
</pre>

=== Interpreting the fluence ===

The volume output is an Nx*Ny*Nz*Ng array of the chosen quantity (with an extra
dimension for photon-sharing patterns, separate sources or replayed detectors). The
default fluence rate is the Green's function for a unit-energy impulse at t=0: the
fraction of the source energy (or photons) passing through a unit area per unit
time. Each gate stores the average over its time window, i.e. approximately the
value at the center of the window, t0+(k-1/2)*Dt. To obtain

* the fluence of each gate: multiply the fluence rate by Dt (or use `-O F`);
* the CW fluence: sum the fluence rate over all gates and multiply by Dt, with
    T1 long enough for the light to decay;
* the energy deposition: use `-O E`, or multiply the fluence by mua
    (`mcxfluence2energy.m`).

The diffuse reflectance can be obtained from the detected photons (the detected
weight divided by the detector area gives the current density), or with `-X 1`
(`cfg.issaveref=1`), which records the escaping weight in the background voxels
next to the domain (stored as negative values in the raw volume; MCXLAB-CL/PMCXCL
return it separately as `dref`). Pad the volume with a layer of 0 voxels where you
want the reflectance. The volume output keeps the voxel order of the input volume
(column-major, or row-major with `-a 1`). `example/validation` compares MCX-CL with
analytical solutions (`cwdiffusion.m`, `tddiffusion.m`, `pmcxcl.cwdiffusion`, ...).

=== Units of the output ===

With the default normalization (`-U 1`), the output is the response to a source that
emits one unit of light (one joule, or one photon) in an impulse at t=0, so each
quantity can be read in energy or in photon units:

<pre>
fluence rate (`flux`, default)
    Energy units: W/mm^2 per J emitted
    Photon units: 1/(mm^2 s) per photon emitted
fluence (`fluence`)
    Energy units: J/mm^2 per J emitted
    Photon units: 1/mm^2 per photon emitted
energy (`energy`)
    Energy units: J absorbed in the voxel per J emitted
    Photon units: the fraction of the photons absorbed there
</pre>

To obtain absolute values, multiply by the energy (or number of photons) your source
delivers; for a continuous source of power P, the CW fluence rate is P times the
time-integrated (summed over gates and multiplied by Dt) fluence rate. Divide the
`energy` output by the voxel volume (`LengthUnit`^3) for the absorbed energy density;
`mcxfluence2energy.m` converts a fluence output into energy deposition. See [Fang2009] and Lihong Wang's *Biomedical Optics: Principles
and Imaging*, chapter 5, for the definitions.

'''Fluence rate and current density.''' The fluence rate is a scalar: the light passing
through a small sphere around a point from all directions. The current density J is a
vector: the net light flowing through a surface in a given direction; it has the same
units. What a detector or the diffuse reflectance measures is the current density
flowing out of the domain: the total weight of the detected photons divided by the
detector area gives J at the detector, as a function of time when the photons are
binned by their time of flight.

=== Output files of the command line ===

<pre>
`session.jnii` (`-F jnii`, default)
    Content: the volume output, JSON-based JNIfTI with compressed binary data
`session.bnii` (`-F bnii`)
    Content: the same in binary JSON (BJData), smaller and faster to read
`session.nii` / `.hdr`+`.img` (`-F nii`/`hdr`)
    Content: NIfTI-1 / Analyze 7.5
`session.mc2` (`-F mc2`)
    Content: raw float32 array, column-major, no header (legacy)
`session_detp.jdt` / `.jdb`
    Content: detected photons (and seeds with `-q 1`), with `jnii`/`bnii`
`session.mch`
    Content: detected photons in the legacy binary format, with `mc2`/`nii`/`hdr`
`session_traj.jdt` / `.jdb` (or `.mct`)
    Content: photon trajectories, with `-D M`
`session.log`
    Content: the messages, with `-l`
</pre>

JNIfTI (`.jnii`, `.bnii`) and JData (`.jdt`, `.jdb`) files follow the
[https://neurojson.org NeuroJSON] specifications: they are plain JSON or binary
JSON, readable in any language, with the array size, type and compression stored
alongside the data. Detected photon files have the structure

<pre>
{"MCXData": {
    "Info":       {"Version", "MediaNum", "DetNum", "Media", ...},
    "PhotonData": {"detid", "srcid", "nscat", "ppath", "mom", "p", "v", "w0"},
    "Seed":       [...]
}}
</pre>

with only the fields selected by `-w` (`srcid` only with several sources, `Seed`
only with `-q 1`). Trajectory files hold a `Trajectory` object with `photonid`,
`p`, `w0`, `srcid` and, with `--trajstokes 1`, `iquv`. Files with the older `.jdat`
suffix are still accepted for replay.

The legacy `.mch`/`.mct` files have a 256-byte header (`MCXHistoryHeader`: magic
`MCXH`, version, number of media, detectors, columns, photons simulated, detected
and saved, voxel size, seed bytes, normalizer, respin, srcnum, savedetflag, total
sources) followed by a float32 array of saved photons x columns, and with `-q 1`
the seeds; `loadmch.m` reads them.

=== Messages ===

MCX-CL prints its settings, the device(s) used, the time spent in each phase (ms) and,
at the end, the simulation speed (photons/ms) and the absorbed fraction. `-D P`
(`cfg.debuglevel='P'`) adds a progress bar.

== # Reading MCX-CL data in MATLAB and Python ==

'''MATLAB/Octave''' - install [https://neurojson.org/download/jsonlab JSONLab] (or
`octave-jsonlab` on Debian/Ubuntu/Fedora) for JNIfTI/JData files; the other loaders
are in MCX's `utils` folder:

<pre>
nii  = loadjd('qtest.jnii');              % JSONLab: .jnii, .bnii, .jdt, .jdb, ...
flux = nii.NIFTIData;                     % Nx x Ny x Nz x Ng
detp = loadjd('qtest_detp.jdt');
ppath = detp.MCXData.PhotonData.ppath;    % one row per detected photon
traj = loadjd('qtest_traj.jdt');

flux = loadmc2('qtest.mc2', [60 60 60 10]);   % utils: .mc2 needs the dimensions
[data, header] = loadmch('qtest.mch');    % utils: legacy detected photons
nii  = mcxloadnii('qtest.nii');           % utils (MATLAB), or niftiread
</pre>

'''Python''' - install `jdata` (`pip install jdata bjdata`, or `python3-jdata` on
Debian/Ubuntu):

<pre>
import jdata as jd
nii  = jd.load('qtest.jnii')              # also .bnii, .jdt, .jdb (needs bjdata)
flux = nii['NIFTIData']                   # numpy array, Nx x Ny x Nz x Ng x 1
detp = jd.load('qtest_detp.jdt')['MCXData']['PhotonData']   # dict of arrays

import pmcxcl
flux = pmcxcl.loadmc2('qtest.mc2', [60, 60, 60, 10])
data, header = pmcxcl.loadmch('qtest.mch')[:2]   # a 3rd value holds the seeds, if saved
</pre>

`.nii` files can be read with `nibabel`.

== # Using MCXLAB-CL in MATLAB and Octave ==

MCXLAB-CL runs MCX-CL inside MATLAB or GNU Octave. It takes the same `cfg` struct as
MCXLAB and returns the same outputs:

<pre>
[flux, detp, vol, seeds, traj] = mcxlabcl(cfg);
</pre>

Request only the outputs you need. `help mcxlabcl` documents every `cfg` field, its
default and the outputs; the OpenCL-specific fields are `cfg.gpuid` (a device number,
or a 0/1 string selecting several), `cfg.workload`, `cfg.optlevel`,
`cfg.compileropt`, `cfg.kernelfile`, `cfg.isatomic` and `cfg.internalsrc`.

<pre>
`flux.data`
    Content: the volume output, Nx x Ny x Nz x Ng (plus extra dimensions for patterns,
        sources, detectors or RF parts)
`flux.dref`
    Content: diffuse reflectance, with `cfg.issaveref=1`
`flux.stat`
    Content: runtime, photons, launched and absorbed energy, normalizer
`flux.jmua`, `.jd`, `.jmus`, `.jmusp`
    Content: adjoint Jacobians
`detp`
    Content: detected photons: `detid`, `ppath`, `nscat`, `mom`, `p`, `v`, `w0`, ...
`vol`
    Content: the preprocessed volume
`seeds`
    Content: RNG seeds of the detected photons, for replay
`traj`
    Content: photon trajectories: `id`, `pos`, `data`
</pre>

Other calls: `mcxlabcl('gpuinfo')` lists the devices, `mcxlabcl('version')` returns the
version, `mcxlabcl(cfg, 'preview')` draws the domain, and `mcxlabcl(cfg, 'cuda')` runs
the simulation with MCX (`mcx.mex`) instead. A variable `USE_MCXCL` in the base
workspace switches both `mcxlab` and `mcxlabcl`: 1 uses MCX-CL, 0 uses MCX, and an
integer N>1 (or a device string such as `'0101'`) also selects that device.

'''Helper functions''' in `utils`: `mcxdetphoton`, `mcxdetweight`, `mcxmeanpath`,
`mcxmeanscat`, `mcxdettpsf`, `mcxdettime`, `mcxdcsg1` (detected photons);
`cwdiffusion`, `tddiffusion` (analytical solutions); `mcxrfreplay`; `mcxplotphotons`,
`mcxplotvol`, `islicer`, `slice3i` (plotting); `mcx2json`, `json2mcx`, `loadmc2`,
`loadmch`, `mcxloadnii`, `mcxfluence2energy` (files). MCX's `utils` folder adds
`mcxpreview`, `mcxplotshapes`, `mcxsvmc`, `mcxcwdref`, `mcxcreate` and more.

'''Examples''': about 40 scripts in `mcxlabcl/examples`, from `demo_mcxlabcl_basic.m` to
atlas head models, SFDI, replay and adjoint Jacobians, polarized light and SVMC; they
are the MCXLAB examples running with MCX-CL.

== # Using PMCXCL in Python ==

PMCXCL brings MCX-CL to Python 3.6+, with the same interface as PMCX:

<pre>
import pmcxcl
res = pmcxcl.run(cfg)                        # cfg: a dict with the MCXLAB field names
res = pmcxcl.run(nphoton=1e6, vol=vol, ...)  # or keyword arguments
res = pmcxcl.mcxlab(cfg)                     # run(), then decode detp/traj into dicts
print(pmcxcl.gpuinfo())                      # the OpenCL devices
</pre>

`res` is a dict with `flux`, `stat` and, when produced, `detp`, `seeds`, `traj`,
`dref`, `vol` and the adjoint Jacobians. The utility functions - `detweight`,
`meanpath`, `dettpsf`, `cwdref`, `dcsg1`, `cwdiffusion`, `tddiffusion`, `rfreplay`,
`loadmc2`, `loadmch`, `mcx2json`, `json2mcx`, `preview`, `plotshapes`, `plotphotons`,
`plotvol`, `svmc` and more - come from `pmcx` and are available as `pmcxcl.<name>`.

All MCXLAB examples have Python versions in MCX's
[https://github.com/fangq/mcx/tree/master/pmcx/examples `pmcx/examples`]; they run with
MCX-CL after replacing `import pmcx` by `import pmcxcl as pmcx`. The
[https://colab.research.google.com/github/fangq/mcx/blob/master/pmcx/tutorials/pmcx_getting_started.ipynb PMCX tutorial notebook]
applies to PMCXCL as well. See `pmcxcl/README.md` for building from source.

== # Using MCX Studio ==

MCX Studio 2, included in the MCX Studio packages, is the graphical front end for
MCX, MCX-CL and MMC (source code in MCX's `mcxstudio2/` folder). A simulation is the
same `.json` file `mcxcl` reads; the settings are grouped in the order a simulation is
described, with F1 help for each; a 3-D view shows the domain, sources, detectors,
fluence and photon paths. Choosing the `intel`, `amd` or `cpu` processor runs mcxcl:

<pre>
mcxstudio qtest.json                              # edit and run
mcxstudio --run --processor intel qtest.json      # run with mcxcl
</pre>

See the MCX README and `mcxstudio2/README.md` in MCX for details.

== # Advanced features ==

MCX-CL provides the same advanced features as MCX; the MCX README describes them in
detail, and the MCXLAB-CL examples show them in use.

* '''Photon replay and Jacobians''' [Yao2018]: save the seeds of the detected photons
    (`-q 1`, or the 4th output of `mcxlabcl`), then replay them (`-E file_detp.jdt`, or
    `cfg.seed=seeds.data` with `cfg.detphotons`) with `-O J` (mua), `P` (mus), `R`
    (frequency domain), `S`, `T` or `B`, for one detector (`-Y n`) or all of them.
    Examples: `demo_mcxlab_replay.m`, `demo_replay_timedomain.m`,
    `demo_replay_frequencydomain.m`; `example/rfreplay`.
* '''Adjoint Jacobians''': give each detector a direction (`Optode.Detector.Dir`,
    `cfg.detdir`) and use output type `adjoint`, `adjoint_dcoeff`, `adjoint_mus`,
    `adjoint_musp`, `adjoint_mua_d` or `adjoint_mua_musp`, in CW or the frequency
    domain. Examples: `demo_validate_mcx_adjoint_*.m`.
* '''Frequency-domain (RF) simulations''': set the modulation frequency
    (`Optode.Source.Frequency` in Hz, `cfg.omega` in rad/s) with output type `rf` to get
    the complex fluence rate. Example: `demo_validate_mcx_rf_redbird.m`.
* '''Photon sharing''' [Yan2020-OL]: a `pattern` source with `SrcNum`/`cfg.srcnum` > 1
    simulates many illumination patterns at once. Example: `demo_photon_sharing.m`.
* '''Polarized light''' [Yan2022-pol]: `Domain.MieScatter`/`cfg.polprop`. Example:
    `demo_polarized_photon.m`.
* '''Split-voxel MC''' [Yan2020] for curved boundaries, and per-voxel optical
    properties. Examples: `demo_svmc_*.m`, `demo_continuous_mua_mus.m`.
* '''Custom phase functions and launch angles''': `cfg.invcdf`, `cfg.angleinvcdf`.
    Examples: `demo_mcxlab_phasefun.m`, `demo_mcxlab_launchangle.m`.
* '''Several devices''': `-G 0111 -W 40,40,20` splits one simulation across devices,
    which may come from different vendors; give a faster device a larger share.
* '''Custom kernels''': `--showkernel 1` prints the kernel source; edit it and run it
    with `-k mykernel.cl` to experiment with the algorithm without recompiling mcxcl.

== # MCX-CL and MCX: differences ==

MCX-CL and MCX share their input format, `cfg` fields, outputs and helper functions,
and are tested against each other step by step. The differences are:

<pre>
hardware
    MCX: NVIDIA GPUs (CUDA), AMD GPUs (HIP)
    MCX-CL: any OpenCL device: NVIDIA, AMD, Intel GPUs, Apple silicon, CPUs
kernel
    MCX: compiled ahead of time
    MCX-CL: compiled for your device at startup (`-o`, `-J`, `-k`)
`-k` option
    MCX: `--voidtime`
    MCX-CL: `--kernel` (use `--voidtime` for the void timer)
`--atomic`
    MCX: ignored (always atomic)
    MCX-CL: selects atomic (1) or non-atomic (0) accumulation
2D disk sources, Lambertian launch in 2D
    MCX: supported
    MCX-CL: not yet (Lambertian launch is 3D only)
MATLAB
    MCX: `mcxlab`
    MCX-CL: `mcxlabcl`, or `mcxlab(cfg, 'opencl')`
Python
    MCX: `pmcx`
    MCX-CL: `pmcxcl` (utility functions from `pmcx`)
</pre>

== # Performance and best practices ==

* '''Pick the right device.''' Run `mcxcl -L` and choose a dedicated GPU with `-G`; the
    first device listed is not always the fastest. A discrete GPU is typically tens of
    times faster than a CPU, and several devices can work together (`-G`, `-W`).
* '''Keep the autopilot on''' (`-A 1`, the default): it sizes the thread count for the
    device, hands out photons dynamically, and compiles the kernel with fast math
    (`-cl-unsafe-math-optimizations`).
* '''Use `-A 0` to reproduce a run exactly.''' With dynamic dispatch and fast math, two
    runs with the same seed agree statistically but not bit by bit. For identical
    results, use `-A 0` and keep the seed, `-t` and `-T` unchanged; saving or replaying
    seeds always uses strict math and static dispatch.
* '''Expect a short startup delay.''' The kernel is compiled for the device each time
    mcxcl starts; this takes from a fraction of a second to a few seconds.
* '''Avoid GPU timeouts.''' A GPU that also drives a display may be reset by the OS
    after a few seconds of computing (see Operating system notes); use a dedicated GPU,
    raise the timeout, or split the photons into shorter runs with `-r -N`.
* '''CPU runs''' (POCL or Intel's runtime) use all CPU cores and are useful for testing
    without a GPU.
* '''Use 1-byte labels''', save only the outputs you need, and remember that the noise
    falls as 1/sqrt(photons).

== # Testing ==

<pre>
host unit tests
    What it checks: parsing, JData, shapes, Mie and output writers, without a device
    How to run: `make unittest` in `src`
regression tests
    What it checks: the command-line options and outputs
    How to run: `test/testmcx.sh`
built-in examples
    What it checks: MCX's 141 example cases, against golden values
    How to run: `test/testbuiltin.sh`
trajectory tests
    What it checks: step-by-step photon trajectories against MCX's golden data
    How to run: `test/testtraj.sh`
PMCXCL tests
    What it checks: the Python module
    How to run: `python3 -m unittest discover -s test` in `pmcxcl/`
fuzzing
    What it checks: mutated built-in inputs and options: crashes, hangs, sanitizer and OpenCL kernel errors
    How to run: `test/fuzz/dfuzz.py`
</pre>

The continuous integration runs all but the fuzzer on a CPU OpenCL device (POCL) with `-A 0`.

== # Getting help ==

* documentation, FAQ and tutorials: https://mcx.space/wiki/
* `help mcxlabcl`, `pmcxcl/README.md`, and the MCX README for features shared with MCX
* questions and discussion: the [https://groups.google.com/g/mcx-users mcx-users mailing list]
* bug reports and feature requests: https://github.com/fangq/mcxcl/issues
* release announcements: the [https://groups.google.com/g/mcx-news mcx-news mailing list]

== # Contributing ==

MCX-CL is open-source software under the GPLv3 license: you may use, modify and share
it, as long as derived work is also released under the GPLv3. Bug fixes and new
features are welcome:

* fork https://github.com/fangq/mcxcl, make your change in your fork, keep it small
    and local, and open a pull request; describe what it fixes and how you tested it;
* or send the change as a `git diff` to the mailing list or a GitHub issue.

Please format C/C++ code with `make pretty` (astyle) in `src`, MATLAB code with
`mh_style --fix` (miss_hit) and Python code with `black` before submitting, and add a
test when you can. Contributions are acknowledged in the What's New section and the
ChangeLog.

== # Publications and how to cite ==

If you use MCX-CL in your research, please cite [Yu2018], which describes MCX-CL, and
[Fang2009], the original MCX paper, as well as the papers of the specific features
you use. The complete
list is maintained at https://mcx.space/#publications.

'''Core papers'''

* [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon
  Migration in 3D Turbid Media Accelerated by Graphics Processing Units," Opt.
  Express, 17(22), 20178-20190 (2009).
  https://opg.optica.org/oe/abstract.cfm?uri=oe-17-22-20178
* [Yu2018] Leiming Yu, Fanny Nina-Paravecino, David Kaeli, Qianqian Fang, "Scalable
  and massively parallel Monte Carlo photon transport simulations for
  heterogeneous computing platforms," J. Biomed. Opt. 23(1), 010504 (2018).
  https://doi.org/10.1117/1.JBO.23.1.010504
* [Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte Carlo
  algorithm for accurate and efficient photon transport modeling in complex
  bio-tissues," Biomed. Opt. Express, 11(11), 6262-6270 (2020).
  https://opg.optica.org/boe/abstract.cfm?uri=boe-11-11-6262

'''Features'''

* [Yao2018] Ruoyang Yao, Xavier Intes, Qianqian Fang, "A direct approach to compute
  Jacobians for diffuse optical tomography using perturbation Monte Carlo-based
  photon 'replay'," Biomed. Opt. Express, 9(10), 4588-4603 (2018).
  https://opg.optica.org/boe/abstract.cfm?uri=boe-9-10-4588
* [Yan2020-OL] Shijie Yan, Ruoyang Yao, Xavier Intes, Qianqian Fang, "Accelerating
  Monte Carlo modeling of structured-light-based diffuse optical imaging via
  'photon sharing'," Opt. Lett., 45, 2842-2845 (2020).
* [Yan2022-pol] Shijie Yan, Steven L. Jacques, Jessica C. Ramella-Roman, Qianqian
  Fang, "Graphics processing unit-accelerated Monte Carlo simulation of polarized
  light in complex three-dimensional media," J. Biomed. Opt. 27(8), 083015 (2022).
  https://doi.org/10.1117/1.JBO.27.8.083015
* [Fang2022-cloud] Qianqian Fang and Shijie Yan, "MCX Cloud - a modern, scalable,
  high-performance and in-browser Monte Carlo simulation platform with cloud
  computing," J. Biomed. Opt. 27(8), 083008 (2022).
  https://doi.org/10.1117/1.JBO.27.8.083008
* [Yuan2018] Yaoshen Yuan, Leiming Yu, Zafer Doğan, Qianqian Fang, "Graphics
  processing units-accelerated adaptive nonlocal means filter for denoising
  three-dimensional Monte Carlo photon transport simulations," J. Biomed. Opt.
  23(12), 121618 (2018).
* Matin Raayai Ardakani, Leiming Yu, David R. Kaeli, Qianqian Fang, "Framework for
  denoising Monte Carlo photon transport simulations using deep learning," J.
  Biomed. Opt. 27(8), 083019 (2022). https://doi.org/10.1117/1.JBO.27.8.083019

'''Mesh-based Monte Carlo (MMC)''', for the related mesh-based simulator

* Qianqian Fang, "Mesh-based Monte Carlo method using fast ray-tracing in Plücker
  coordinates," Biomed. Opt. Express, 1(1), 165-175 (2010).
  https://opg.optica.org/boe/abstract.cfm?uri=boe-1-1-165
* Qianqian Fang and Shijie Yan, "Graphics processing unit-accelerated mesh-based
  Monte Carlo photon transport simulations," J. Biomed. Opt. 24(11), 115002 (2019).
  https://doi.org/10.1117/1.JBO.24.11.115002
* Shijie Yan, Anh Phong Tran, Qianqian Fang, "A dual-grid mesh-based Monte Carlo
  algorithm for efficient photon transport simulations in complex 3D media," J.
  Biomed. Opt. 24(2), 020503 (2019). https://doi.org/10.1117/1.JBO.24.2.020503
* Yaoshen Yuan, Shijie Yan, Qianqian Fang, "Light transport modeling in highly
  complex tissues using the implicit mesh-based Monte Carlo algorithm," Biomed.
  Opt. Express, 12(1), 147-161 (2021).
  https://opg.optica.org/boe/fulltext.cfm?uri=boe-12-1-147
* Ruoyang Yao, Xavier Intes, Qianqian Fang, "Generalized mesh-based Monte Carlo for
  wide-field illumination and detection via mesh retessellation," Biomed. Opt.
  Express, 7(1), 171-184 (2016).
  https://opg.optica.org/boe/abstract.cfm?uri=boe-7-1-171
* Yuxuan Zhang and Qianqian Fang, "BlenderPhotonics - an integrated open-source
  software environment for three-dimensional meshing and photon simulations in
  complex tissues," J. Biomed. Opt. 27(8), 083014 (2022).
  https://doi.org/10.1117/1.JBO.27.8.083014

'''Applications''': MCX has been used to model light in the brain (fNIRS and
photobiomodulation dosimetry), neonatal heads, bone, and many other tissues; see
https://mcx.space/#publications for examples, and Google Scholar for the many
works citing MCX.

== # Acknowledgement ==

The MCX project, including MCX-CL, is funded by the US National Institutes of Health (NIH) / National
Institute of General Medical Sciences (NIGMS) under grant R01-GM114365. We thank
the users and contributors who reported bugs, suggested features and contributed
code; they are credited in the What's New section and the ChangeLog.

MCX-CL contains modified versions of the below source codes from other open-source
projects (with a compatible license).

=== cJSON library by Dave Gamble ===

* Files: src/cjson folder
* Copyright (c) 2009 Dave Gamble
* URL: https://github.com/DaveGamble/cJSON
* License: MIT License, https://github.com/DaveGamble/cJSON/blob/master/LICENSE

=== ZMat data compression unit ===

* Files: src/zmat/*
* Copyright: 2019-2026 Qianqian Fang
* URL: https://github.com/fangq/zmat
* License: GPL version 3 or later, https://github.com/fangq/zmat/blob/master/LICENSE.txt

=== miniz data compression library ===

* Files: src/zmat/miniz/*
* Copyright: 2010-2014 Rich Geldreich and Tenacious Software LLC, 2013-2014 RAD Game Tools and Valve Software
* URL: https://github.com/richgel999/miniz
* License: MIT License, src/zmat/miniz/LICENSE

=== LZ4 data compression library ===

* Files: src/zmat/lz4/*
* Copyright: 2011-2020, Yann Collet
* URL: https://github.com/lz4/lz4
* License: BSD-2-clause, https://github.com/lz4/lz4/blob/dev/lib/LICENSE

=== LZMA SDK and Easylzma data compression libraries ===

* Files: src/zmat/easylzma/*
* Copyright: 2009, Lloyd Hilaiel, 2008, Igor Pavlov
* License: public-domain
* Comment:
 All the cruft you find here is public domain.  You don't have to
 credit anyone to use this code, but my personal request is that you mention
 Igor Pavlov for his hard, high quality work.

=== UBJ (UBJSON) library by Steven Braeger ===

* Files: src/ubj/*
* Copyright (c) 2014 Steven Braeger
* URL: https://github.com/Steve132/ubj
* License: MIT License, src/ubj/LICENSE

=== Pybind11 ===

* Files: src/pybind11/*
* Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
* URL: https://github.com/pybind/pybind11/
* License: BSD-style license, https://github.com/pybind/pybind11/blob/master/LICENSE

=== myslicer toolbox by Anders Brun ===

* Files: utils/{islicer.m, slice3i.m, image3i.m}
* Copyright (c) 2009 Anders Brun, anders@cb.uu.se
* URL: https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume
* License: BSD-3-clause License, https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume#license_modal
