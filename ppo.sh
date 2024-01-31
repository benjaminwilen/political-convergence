#!/bin/bash


mkdir -p outputs

if [ -e "outputs/log.txt" ]; then
    rm "outputs/log.txt"
fi

python ppo.py > "outputs/log.txt" 2>&1 | tee -a "outputs/log.txt"

exit $?
