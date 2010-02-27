#!/bin/sh
if [ ! -e semi60x60x60.bin ]; then
  dd if=/dev/zero of=semi60x60x60.bin bs=1000 count=216
  perl -pi -e 's/\x0/\x1/g' semi60x60x60.bin
fi

# use -T 256 gives huge speed boost for ATI cards
#time ../../bin/mcxcl -t 2048 -T 128 -g 10 -m 1000000 -f qtest.inp -s qtest -r 1 -a 0 -b 0 -k ../../src/mcx_core.cl
time ../../bin/mcxcl -t 1024 -T 128 -g 10 -m 100000 -f qtest.inp -s qtest -r 1 -a 0 -b 0 -k ../../src/mcx_core.cl

# use CPU backend, set CPU_MAX_COMPUTE_UNITS=n to specify number of CPU cores
#time ../../bin/mcxcl -t 2048 -T 256 -g 10 -m 1000000 -f qtest.inp -s qtest -r 1 -a 0 -b 0 -k ../../src/mcx_core.cl 
