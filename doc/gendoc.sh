#!/bin/sh

# commands to update the document pages from homepage

ROOTURL="https://mcx.space/wiki/index.cgi?embed=1&keywords"

lynx -dump "$ROOTURL=MCXCL" > Download.txt
lynx -dump "$ROOTURL=Workshop/MCX18Preparation/MethodA" > INSTALL.txt

wget https://mcx.space/wiki/upload/mcxcl_benchmark_0118.png -Omcxcl_benchmark.png

