# Copyright (c) 2022-2023 Matin Raayai Ardakani <raayaiardakani.m at northeastern.edu>. All rights reserved.
# Copyright (c) 2022-2024 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""PMCXCL - Python bindings of MCX-CL, the OpenCL edition of Monte Carlo eXtreme.

MCX-CL runs on NVIDIA, AMD, Intel and Apple GPUs, and on CPUs through an OpenCL runtime
such as POCL.

Simulator (compiled module ``_pmcxcl``)

* ``run(cfg)`` or ``run(**cfg)``: run a simulation; see ``help(pmcxcl.run)`` for the
  ``cfg`` keys and the outputs
* ``gpuinfo()``: list the OpenCL devices
* ``version()``: the MCX-CL version string

Utilities

When pmcx is installed (``pip install pmcx``), its pure-Python utilities are re-exported:
:func:`mcxlab` (which runs pmcxcl), ``mcxcreate``, ``detweight``, ``meanpath``,
``meanscat``, ``dettpsf``, ``dettime``, ``detphoton``, ``getdistance``, ``rfreplay``,
``rfmusreplay``, the diffusion solutions, ``svmc``, the file I/O functions and the
plotting functions; ``bench`` holds the built-in benchmarks. See the pmcx documentation.

Examples
--------
>>> import numpy as np
>>> import pmcxcl
>>> pmcxcl.gpuinfo()
>>> res = pmcxcl.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype="uint8"),
...                  tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30, 30, 0],
...                  srcdir=[0, 0, 1], prop=np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]))
"""

try:
    from _pmcxcl import gpuinfo, run, version
except ImportError as e:  # pragma: no cover
    # show why: a present but unloadable _pmcxcl (e.g. an arm64 build in an x86_64 python)
    # used to be reported as "not compiled"
    print("the pmcxcl binary extension (_pmcxcl) can not be loaded: %s" % e)

try:
    from pmcx import (
        detweight,
        cwdref,
        meanpath,
        meanscat,
        dettpsf,
        dettime,
        tddiffusion,
        getdistance,
        detphoton,
        mcxlab as _pmcx_mcxlab,
        cwdiffusion,
        cwfluxdiffusion,
        cwfluencediffusion,
        dcsg1,
        mcxcreate,
        rfreplay,
        rfmusreplay,
        loadmc2,
        loadmch,
        loadfile,
        mcx2json,
        json2mcx,
        loadnii,
        preview,
        plotshapes,
        plotphotons,
        plotvol,
    )
except ImportError:  # pragma: no cover
    print(
        "please first install pmcx module to use utility functions such as 'detweight', 'meanpath' etc"
    )
else:

    def mcxlab(*args):
        """pmcx.mcxlab() running the simulation with pmcxcl (OpenCL) instead of pmcx (CUDA)"""
        if len(args) == 1 and isinstance(args[0], dict):
            args = (args[0], "opencl")
        return _pmcx_mcxlab(*args)

    mcxlab.__doc__ += "\n" + (_pmcx_mcxlab.__doc__ or "")

    try:
        from pmcx import (
            svmc,
        )  # newer than pmcx 0.7.3; with an older pmcx, only pmcxcl.svmc is missing
    except ImportError:  # pragma: no cover
        pass

from .bench import bench

__version__ = "0.8.0"

__all__ = (
    "gpuinfo",
    "run",
    "version",
    "bench",
    "detweight",
    "cwdref",
    "meanpath",
    "meanscat",
    "dettpsf",
    "dettime",
    "tddiffusion",
    "getdistance",
    "detphoton",
    "mcxlab",
    "cwdiffusion",
    "cwfluxdiffusion",
    "cwfluencediffusion",
    "dcsg1",
    "mcxcreate",
    "rfreplay",
    "rfmusreplay",
    "loadmc2",
    "loadmch",
    "loadfile",
    "mcx2json",
    "json2mcx",
    "loadnii",
    "preview",
    "plotshapes",
    "plotphotons",
    "plotvol",
)

if "svmc" in globals():
    __all__ += ("svmc",)
