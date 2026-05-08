#!/bin/bash

SRC=../quickscanx/ters2d/mode_070
DST=ters2d/mode_070

# format: src_tippos  dst_tippos
copies=(
    "000 064"
    "001 065"
    "002 066"
    "003 067"
    "004 068"
    "005 069"
    "006 070"
    "007 071"
    "008 072"
    "009 073"
)

for entry in "${copies[@]}"; do
    read src_idx dst_idx <<< "$entry"
    for disp in positive_displacement negative_displacement; do
        cp "$SRC/tippos_${src_idx}/${disp}/field_on/aims.out" \
           "$DST/tippos_${dst_idx}/${disp}/field_on/aims.out"
        echo "Copied tippos_${src_idx} -> tippos_${dst_idx} (${disp})"
    done
done
