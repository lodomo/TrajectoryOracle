#!/bin/sh

for file in train_videos/*; do
    python cli.py -i ${file} -q tables/$1.pkl
done
