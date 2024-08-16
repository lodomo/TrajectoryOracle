#!/bin/bash

dts=(10 25 50 100)
video_dir="./test_videos"
videos=(`ls $video_dir`)

for dt in ${dts[@]}; do
    for video in ${videos[@]}; do
        echo "Processing $video with dt=$dt"
        python test_plot_yolo.py $video_dir/$video $dt
    done
done




