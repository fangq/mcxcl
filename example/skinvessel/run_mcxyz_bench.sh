#!/bin/sh

../../bin/mcxcl --bench skinvessel -n 1e7 -e 0.01 -d 0 -F bnii "$@"
