import argparse  
import os 


add_search_dirs = [
"C:/Users/kent/Documents/GitHub/boost_1_67_0", 
"C:/Users/kent/Anaconda3/include"]


PAR = argparse.ArgumentParser()
PAR.add_argument('-f', "--file", required=True, type=str, nargs='+' )
PAR.add_argument('-o', "--output", default=None, type=str )
arg  = PAR.parse_args()

cmd_add_search_dirs = ["-I "+(i) for i in add_search_dirs]
cmd = "g++ "

while cmd_add_search_dirs: cmd +=cmd_add_search_dirs.pop() + ' '

while arg.file: cmd += arg.file.pop() + ' '

if arg.output: cmd += '-o {}'.format(arg.output)


os.system(cmd)
#os.system('''gcc {} ''')