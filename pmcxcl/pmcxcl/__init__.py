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

"""PMCXCL - Python bindings for Monte Carlo eXtreme (OpenCL) photon transport simulator

Example usage:

# To list available GPUs
import pmcxcl
pmcxcl.gpuinfo()

# To run a simulation
res = pmcxcl.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype='uint8'),
               tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30,30,0],
               srcdir=[0,0,1], prop=np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]))
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

from .bench import bench

__version__ = "0.7.5"

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
