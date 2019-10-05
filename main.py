import argparse
from training.mslr import run_mslr


def parse_args():
    parser = argparse.ArgumentParser(description='Learning to Rank')
    parser.add_argument('-m', '--model', type=str, default="rank_net",
                        help='model to use')
    parser.add_argument('-t', '--task', type=str, default="mslr",
                        help='task to apply')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "mslr":
        run_mslr(args.model)
    else:
        raise ValueError("Unsupported task {}".format(args.model))

