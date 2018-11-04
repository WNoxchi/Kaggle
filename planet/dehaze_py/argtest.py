import argparse
from pathlib import Path

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", nargs=1, help="path to directory containing files")
    parser.add_argument("-sp", "--sliceprefix", nargs=1, help="TODO")
    parser.add_argument("-ss", "--slicesuffix",  nargs=1, help="TODO")
    parser.add_argument("-v", "--verbose",  default="False", nargs=1, help="verbosity")

    parser.add_argument("-n", "--nruns")

    args = parser.parse_args()

    print(args)

if __name__=="__main__":
    main()