#!/bin/bash

VOXCELEB_ROOT="/home/bsrivast/asr_data/voxceleb2"

find $VOXCELEB_ROOT -iname "*.m4a" | while read f; do
  echo "$f"
  out_f="${f/orig/wav}"
  out_f=/"${out_f/m4a/wav}"
  echo "${out_f}"
  if [ ! -d $(dirname "${out_f}") ]; then
    mkdir -p $(dirname "${out_f}")
  fi
  ffmpeg -i /"$f" -ac 1 -ar 16000 -f "wav" "${out_f}"
done
