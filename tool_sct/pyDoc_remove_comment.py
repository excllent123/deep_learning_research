#! /usr/bin/env python
# Author : Kent Chiu (Jin-Chun)
# This module is a CLI-tool for compressing py-code to a production-model 

import argparse, os

# Get Args
parse = argparse.ArgumentParser()
parse.add_argument('-i', '--InFileAddress', type=str, required=True,
                  help='input the file name to search')
parse.add_argument('-o', '--target_dir', default=os.getcwd(),
                  help='default is your current working dir, or you could assign one')
parse.add_argument('-n', '--numbers_print',type=int, default=5,
                  help='default is 5 search results')
arg = parse.parse_args()

# Def Function
def _remove_python_command(inputFile):
    assert len(inputFile)>3
    result = []
    for line in inputFile:
        temp = line.replace(' ','')
        if temp[0] =='#' or temp=='\n' or temp =='n':
            continue
        else: result.append(line)
    return result

def _compress_code(inputFile):
    """
    [Warning] 
    if use this function, the compressed code is not able to rnu
    """
    assert len(inputFile)>3
    result = ''
    for line in inputFile:
        result+=line.replace('\n',';')
    return result

if __name__=='__main__':
    try : 
        # read + process
        with open(arg.InFileAddress, 'r') as f :
            out = _remove_python_command(f.readlines())
        
        # define
        tar_address = os.path.join(arg.target_dir, 
            'NC_'+arg.InFileAddress.split('/')[-1])

        # write 
        with open(tar_address, 'w' ) as f:
            f.writelines(out)
        print ('[*] DONE')

    except Exception as e :
        print (e)
