import time
import subprocess
import argparse


def main():
    sec = 0

    parser = argparse.ArgumentParser(
        description="Synchronize Weights and Biases (wandb) for a specified folder."
    )
    parser.add_argument("--foldername", help="The name of the folder to synchronize")
    parser.add_argument(
        "--step",
        type=int,
        default=1800,
        help="The time step in seconds (default: 30 min)",
    )

    args = parser.parse_args()
    foldername = args.foldername
    step = args.step

    while True:
        print("Time step:", sec / 3600, "Hours")
        sec += step
        subprocess.run(["wandb", "sync", foldername])
        time.sleep(step)


if __name__ == "__main__":
    main()
