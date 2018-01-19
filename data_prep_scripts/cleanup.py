import os
import shutil
import argparse

def main(args):
    if os.path.exists(args.dir):
        shutil.rmtree(args.dir)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='directory to clean (delete)', type=str)
    main(parser.parse_args())
