#!/bin/sh

../../bin/mcxcl -A -f benchmark1.json -k ../../src/mcx_core.cl -b 0 "$@"
