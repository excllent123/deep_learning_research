import os
import numpy as np
import argparse
import os


#parser = argparse.ArgumentParser()
#parser.add_argument('-t', '--train_path', type=str, required=True)
#parser.add_argument('-l', '--label_path', type=str, required=True)


class VocPreprocess(object):
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.filelist = os.listdir(image_path)
        # notice 1 to 1 relation

    def gen_foler_train_batch(self, batch_size):
        batch_X = []
        batch_Y = []
        while self.filelist:
            x_file_name = filelist.pop()
            if x_file_name.split(".")[-1] not in ('png','jpq'):
                continue

            img = imread(os.path.join(self.image_path, x_file_name))



