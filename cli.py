from argparse import ArgumentParser
import os.path

from process_video import process_video

if __name__ == "__main__":
    def exit_error(err: str):
        print(err)
        exit(1)

    # parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input file", required=True)
    parser.add_argument("-q", "--qtable", type=str, help="q-table file", required=True)
    parser.add_argument("-d", "--distance", type=float, help="distance threshold", default=50.0)
    parser.add_argument("-p", "--prediction", type=int, help="prediction frame count", default=20)
    args = parser.parse_args()

    # validate command-line arguments
    if not os.path.exists(args.input):
        exit_error(f"input file {args.input} does not exist")
    if args.distance < 0:
        exit_error("distance threshold cannot be less than 0")
    if args.prediction <= 0:
        exit_error("prediction frame count must be greater than 0")

    process_video(args.input, None, args.qtable, args.distance, args.prediction)
