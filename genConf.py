import argparse
from cfgParser import Conf
import os

def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description=''' Get conf file path''')
    # Add arguments
    parser.add_argument(
        '-conf', '--modelpy', type=str, nargs='+',
        help='filePath, could be multi-files',required=True)

    # Array for all arguments passed to script
    args = parser.parse_args()

    # Return path
    return args.modelpy

def main():
    conf = Conf(get_args())
    if not os.path.isdir(conf['model']):
        os.makedirs(conf['model'])
