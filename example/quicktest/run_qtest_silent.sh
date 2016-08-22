#!/bin/sh

../../bin/mcxcl -A -n 1e7 -f qtest.inp -k ../../src/mcx_core.cl -l "$@"

