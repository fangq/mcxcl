= README for examples =

In this folder, you will find a number of examples to validate, test, or
demonstrate the functionalities of MCX-CL. Each one is run by the run_*.sh
(Linux and macOS) or run_*.bat (Windows) script in its folder, which calls the
mcxcl binary in ../../bin; folders with a README.txt explain the procedures and
how to interpret the results. Add "-G 1" (or another device number, see
"mcxcl -L") to select the OpenCL device.

* quicktest       - a homogeneous cube to check that mcxcl runs (also lists the devices)
* validation      - a semi-infinite medium compared with the diffusion solution
* benchmark       - the standard benchmarks: a homogeneous cube, and with a sphere
* multisrc        - several sources in one simulation (arrays of Pos/Dir in Optode.Source)
* svmcsphshells   - spherical shells with split-voxel MC (SVMC)
* polarized       - polarized light in a homogeneous slab, with cyclic boundaries
* rfreplay        - RF (frequency-domain) Jacobians by replaying detected photons
* skinvessel      - the skin-vessel benchmark of mcxyz
* colin27         - the Colin27 adult brain atlas
* digimouse       - the Digimouse mouse atlas
* usc19-5         - the USC 19.5-year-old brain atlas
* mcxcl_demo.mcxp - a project file for MCX Studio

MCX-CL reads the same input files as MCX, so the examples of MCX (mcx/example)
also run with mcxcl. More examples, as MATLAB/Octave scripts, are in
../mcxlabcl/examples; the Python examples of pmcx (mcx/pmcx/examples) also run
with pmcxcl.
