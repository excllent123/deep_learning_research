#!python 
import argparse  
import os 


add_search_dirs = [
"C:/Users/kent/Documents/GitHub/boost_1_67_0", 
"C:/Users/kent/Anaconda3/Library/bin", 
"C:/Users/kent/Anaconda3/DLLs",
"C:/Users/kent/Anaconda3/include"]


PAR = argparse.ArgumentParser()
PAR.add_argument('-f', "--file", required=True, type=str, nargs='+' )
PAR.add_argument('-o', "--output", default=None, type=str )
arg  = PAR.parse_args()

cmd_add_search_dirs = ["-I "+(i) for i in add_search_dirs]
cmd = "g++ "

cmd += ' -std=gnu++03 '
# python pyconfig.h _hypot comflict with wingw-gcc cmath hypot 
# add this to set pirority 

# build as a static library, That way, you will got another import error: 
# DLL load failed: %1 is not a valid Win32 application. So just build as DLL

cmd += ' -c '
# since we only add .o file for python to call 
# if we just compile the function ==> gcc / g++ need main() function 

while cmd_add_search_dirs: cmd +=cmd_add_search_dirs.pop() + ' '

while arg.file: cmd += arg.file.pop() + ' '


cmd += " -march=x86-64 -m64 " # force -march=i686 -m32 (32bit) -march=x86-64 -m64 (64 bit )



os.system(cmd)
#os.system('''gcc {} ''')

print(cmd)

if arg.output: 
    if '.pyd' not in arg.output :
        arg.output+= '.pyd'
    cmd += ' -o {} '.format(arg.output)


os.system(cmd)
print(cmd)

#g++ -c -DBUILDING_EXAMPLE_DLL example_dll.cpp
# g++ -shared -o example_dll.dll example_dll.o -Wl

#g++ -c example_exe.cpp
#g++ -o example_exe.exe example_exe.o -L. -lexample_dll

#For Windows, you go into $(BoostDir)\tools\build\v2\engine and run build.bat which automatically builds bjam (into the bin.ntx86 directory on windows). There is a build.sh file there too, but I've never used a Mac so I don't know if that will work for you. Otherwise, just do a Google search for a precompiled bjam executable for Mac OS X.

# https://stackoverflow.com/questions/44161857/the-dll-created-by-boost-python-cannot-be-imported-following-boost-pythons-qui

# C:\Users\kent\Documents\GitHub\boost_1_67_0\tools\build


# turns out need build boost-jam 
# follwoing the tutal but use different compiler bootstrap.bat gcc
# b2 install --prefix=dir toolset=gcc
# b2 --with-python toolset=gcc


# copy libs/python/example 
# need follow the convention of 
# 1. parent folder with Jamroot 
# 2. sub_folder with your cpp and with a .jam file 
# 3. add include path there is your compiler need -I (search-dir)
# 4. bjam toolset=gcc cxxflags=-std=gnu++0x 
# 4->(if you use mingw-gcc in windows and enconter hypot confict in cmath)