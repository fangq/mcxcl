#!/bin/sh
time ../../bin/mcxcl -A -n 1e7 -f validation.inp -s semi_infinite -a 0 -b 0 -d 0 $@
