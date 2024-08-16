###############################################################################
#
# Author: Lorenzo D. Moon
# Professor: Dr. Anthony Rhodes
# Course: CS-441
# Assignment:  Final Project: Trajectory Oracle
# Description: Test the PlotYolo class and DetectedObject class
#
###############################################################################

from plot_yolo import PlotYolo
import sys


def main(argv):
    video_filepath = argv[0]

    if len(argv) > 1:
        distance_threshold = int(argv[1])
    else:
        distance_threshold = 50

    plot_yolo = PlotYolo(video_filepath, distance_threshold)
    plot_yolo.plot_video(spinner=True)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
