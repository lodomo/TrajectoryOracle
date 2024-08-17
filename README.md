# TrajectoryOracle

CS 441 Final Project 

Group Members:
Aaron Bernard, Alexander Ho, Lorenzo Moon, Luke Walker, Alan Shirk

## Environment Setup

First, ensure that your current Python version is 3.11 or install it if necessary. Version 3.11 is required to set up and run this program.

Then, to set up our program, clone the repository and enter the project directory:

```
git clone https://github.com/lodomo/TrajectoryOracle.git
cd TrajectoryOracle
```

Next, set up the virtual environment and install the necessary dependencies with the following commands:

```bash
pip install pipenv
pipenv shell
pip install opencv-python
pip install ultralytics
```

## Running the Program
After the virtual environment is activated and the dependencies are installed, the Trajectory Oracle application can then be run with the following command:

```bash
python process_video.py
```

After the program starts it will bring up a dialogue box where the distance threshold and prediction frames parameters can be set.

The distance threshold is the maximum allowable distance between a tracked object's position in one frame of the video and it's position in the next frame in order for it to be recognized as the same object.

The prediction frames parameter refers to the upcoming frames which the Reinforcment Learning agent will use to predict the future position of objects detected in the video.

To proceed, click on the "Select Video" button to select a video file to process. The program will then analyze the video using the RL agent.

## Viewing the Results

After the video is processed, a Q-Table Heatmap will be displayed. The heatmap shows the "Predict Now" or "Wait" actions made by the RL agent. The color corresponds to the Q value.

When the heatmap window is closed, the Predicted vs. Actual Positions Over Time plot is displayed, which shows the accuracy of the RL agent's position predictions.

Lastly, a Trajectory Map of Detected Objects is displayed, showing the position trajectories for the objects detected by the YOLO object detection framework.

## Command Line Interface

Alternatively, instead of using the GUI, the program can be run via a command line interface. A video file can be provided as an input file, and a qtable filename can be specified. The distance threshold and the number of prediction frames can be provided optionally. The distance threshold defaults to 50.0, and the number of prediction frames defaults to 20.

This is the usage statement of the program:

```
usage: cli.py [-h] -i INPUT -q QTABLE [-d DISTANCE] [-p PREDICTION]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file
  -q QTABLE, --qtable QTABLE
                        q-table file
  -d DISTANCE, --distance DISTANCE
                        distance threshold
```