


import argparse, cv2, imageio, os
import pandas as pd
from skimage.io import imsave, imread


parser = argparse.ArgumentParser()
parser.add_argument('-f','--file', help='the labeled file path')
parser.add_argument('-s','--source_folder', help='corresponding  source')
parser.add_argument('-o','--output', help='output folder')
arg = vars(parser.parse_args())
# The source foler must contain a bonch of images and vatic.txt



maplist = ['Rhand', 'ScrewDriver']
# 
# output 



if __name__ == '__main__':
    # make sure output folder is not existed
    try:
        assert os.path.isdir(arg['output'])==False
    except:
        print ('the output folder is existed')
    os.makedirs(arg['output'])

    # Get vid
    vid = imageio.get_reader(arg['vidPath'])

    # Get df from vatic
    df = get_vatic_data(arg['file'])



    # imsave bg
    for _, img in enumerate(gen_bg(vid, df)):
        tarFolder = arg['output']+'\\BG\\'
        if not os.path.isdir(tarFolder):
            os.makedirs(tarFolder)
        imsave(tarFolder+str(_)+'.png',img)

    labels = list(set(df['label_name']))
    for label in labels:
        for _, img in enumerate(gen_pos(vid, df, label)):
            label_folder = label.replace('"',"").split('\n')[0]
            tarFolder = arg['output']+'\\{}\\'.format(label_folder)

            if not os.path.isdir(tarFolder):
                os.makedirs(tarFolder)
            imsave(tarFolder+str(_)+'.png', img)


