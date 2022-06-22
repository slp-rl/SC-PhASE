#!/bin/bash

root_dir=$1
target_dir=$2
target_sr=${3:-16000}

rm -rf target_dir
mkdir -p target_dir
files=`ls $root_dir`
count=0
n_files=`ls $root_dir | wc -l`
echo Started: $root_dir.
echo N_files: $n_files
for file in $files; do
  sox $root_dir/$file $target_dir/$file rate $target_sr;
done
echo Done.
