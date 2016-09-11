
from cfgParser import Conf
import argparse

def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description=''' Get conf file path''')
    # Add arguments
    parser.add_argument(
        '-conf', '--conf', type=str, nargs='+',
        help='filePath, could be multi-files',required=True)

    # Array for all arguments passed to script
    args = parser.parse_args()

    # Return path
    return args.conf

def main():
    # load the data =>
    # clean in shape
    # =multiple-node
    # =multiple-core
    # =multiple-threading
    # training
    # use multiple-threading to excute different conf training
    pass


if __name__ = '__main__':
    main()
