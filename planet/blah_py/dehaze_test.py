# WNixalo | 20181104
####################
import argparse
from time import time
import dehaze

def go(n=10, multi=False):
    t = time()
    for i in range(n):
        dehaze.dehaze('image/foggyHouse.jpg', output=f'{(0,i)[multi]:3d}.jpg')
    print(f'Time for {n} dehazes: {time() - t:.3f}')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nruns", type=int)
    parser.add_argument("-m", "--multi", action="store_true", default=False)
    args = parser.parse_args()
    n = args.nruns
    m = args.multi

    go(n=n, multi=m)

if __name__=="__main__":
    main()