'''
# Description :
  - To augmentating image is important in training DNN.
  - There are multiple ways to perform augmentation 
  - such as in tf.train & keras.preprocess
  - tflearn 

# Functionalities :
  - image data augmentation ops
  - image-tensor = (height, wid, channel)
  
  [serialization : single or multipile batch-per-record with augmentation]
  - from dictionary to tf-record 
  - from dictionary to hdf5

# Image IO Conditions <batch-size>, <io-mode>
  [image-classification ]
  - from dictionary to batch-pair <with balance pumping>
    |-> single - tag - prediction


  [image-detections, image-caption, image-multi-tag]
  - file-file maping (img vs target-file)
    |-> multi  - tag - prediction
    |-> tag-tree(yolo-9000)
    |-> yolo

# Augmentation Functions
  [data-preprocess]
  - sample-wise-normalization
  - rescale
  - filter outlier if any 

  [affine-transformation-base]
  - (1) random_rotation 
  - (2) random_shift
  - (3) random_zoom
  - (4) random_crop
  - (5) random_black
  - (6) random_blur
  - (7) random_color_space
  - (8) random_color_jitter

# Segmentation
  [about 8 ways to segment the image]
  - (1) Histogram Thresholding 
  - (2) Color Clusttering (like slic in skimage) [This](http://sharky93.github.io/docs/dev/auto_examples/plot_segmentations.html)
  - (3) Region growing, region splitting and merging 
  - (4) Edge-base (canny, boble ... etc)
  - (5) Physical Model-base 
  - (6) Fuzzy Approach
  - (7) Neural Network (deep segment)
  - (8) Generic Algorithm

# Segmentation Functions:
  - TBD

'''


import numpy as np
import os
from skimage.io import imread
import random 
try :
    from inspect import signature
except:
    from inspect import getargspec
from img_augment_ops import ImgAugmentator
# random pipe-in 
# balance-propotion


class BatchGenerator():
    def __init__(self):
        self.augment_callbacks = []
        self.regist_ops = ImgAugmentator.__dict__
        self.dataset    = None
        self.train_data = None
        self.valid_data = None
        self.test_data  = None 

    def to_hdf(self):
        '''Perform serialization with augmented data Pair to hdf '''
        raise NotImplementedError()

    def to_tfrecord(self):
        '''Perform serialization with augmented data Pair to tfrecord '''
        raise NotImplementedError()

    def gen_from_dictionary():
        raise NotImplementedError()

    def gen_balance():
        raise NotImplementedError()

    def _get_check_variable(self, func):
        '''Perform check function parameter without default-value

        # Dependence : 
          - inspect.signature  [py3]
          - inspect.getargspec [py2]
        
        '''
        try: # python3 for args without defaults-value
            s = signature(func).parameters.items()
            check_variable = [k for k, v in s if v.default==v.empty] 
        except:
            args, varargs, keywords, default = getargspec(func)
            # return args without defaults-value
            if default:
                check_variable = args[:-len(default)]
            else:
                check_variable = args
        return check_variable

    def add_augment_op(self, func, **kwargs):
        assert (func.__name__ in self.regist_ops) == True 

        for s in self._get_check_variable(func):
            if (s!='x') and (s!='self') and (s not in kwargs.keys()):
                raise ValueError('missing [{}] parameter in {} ''function \n'
                         '[Check] {}'.format(s, func.__name__, func.__doc__))

        self.augment_callbacks.append([func, kwargs])

    def _apply_augment_op(self, img):
        '''store : [func, {}] usage : func(img, **value)'''
        for func, value in self.augment_callbacks:
            img = func(img, **value)
        return img

class FileFileGenerator(BatchGenerator):
    pass

class UnsupervisedGenerator(BatchGenerator):
    pass

class ImgOneTagGenerator(BatchGenerator):
    '''
    # Args:
      - dir_path_list : [class_1_dir, class_2_dir , ... ]

    # Fuctions:
      - gen_balance_batch

    # Usage : 
        ```
        # step 1 : init the augmentation operation
        augment_op = ImgAugmentator()

        # step 2 : init the BatchGenerator with from dictionary
        # [class_1, class_2, ....] 

        data_preprocess_pipe = ImgBatchGenerator([train/type1,train/type2/, train/type3])    

        # step 3 : add augmentation ops
        data_preprocess_pipe.add_augment_op(augment_op.<some_functions> , **kwargs )

        # step 4 : 
        data_generator = data_preprocess_pipe.gen_balance_batch(batch_size)

        ```

    # Notice : 
        - the order of the imput path_list decided the label as the order 0, 1,2 ..
    '''

    def __init__(self, dir_path_list):
        BatchGenerator.__init__(self)
        self.dir_path_list = dir_path_list
        self.img_path_all  = []
        for dir_path in dir_path_list:
            self.img_path_all.append([os.path.join(dir_path, s) \
                                      for s in os.listdir(dir_path) \
                                      if s.split('.')[-1] in ('jpg', 'png')])
        print ('img_path for {} classes'.format(len(self.img_path_all)))
 
    def gen_balance_batch(self, batch_size, max_iter_per_epoch=10e4):
        '''A generator of a balance batch pumping for classification

        # Args 
          - batch_size : batch_size
          - max_iter_per_epoch : default 10e4, 

        # Usage : 
          ```
          # keras 
          # tensorflow 
          # pytorch
          ```
        '''
        assert batch_size % len(self.dir_path_list) ==0

        y_table = np.eye(len(self.dir_path_list))

        iter_num=0
        while iter_num < max_iter_per_epoch:
            batch_y = []
            batch_x = []
            for i in range(len(y_table)):
                img_path_by_class = self.img_path_all[i]
                for j in random.sample(range(0,len(img_path_by_class)), batch_size):
                    batch_y.append(y_table[i])
                    img = self._apply_augment_op(imread(img_path_by_class[j]))
                    batch_x.append(img)
            iter_num+=1
            tmp = list(zip(batch_x, batch_y))
            random.shuffle(tmp)
            batch_x, batch_y = zip(*tmp)
            yield np.array(batch_x), np.array(batch_y)


