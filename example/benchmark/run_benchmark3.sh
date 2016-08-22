#!/bin/sh

../../bin/mcxcl -A -f benchmark3.json -b 1 -s benchmark3 -k ../../src/mcx_core.cl "$@"
