
from cfgParser import Conf
import argparse

class GN(object):
    def __init__(self):
        pass



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
