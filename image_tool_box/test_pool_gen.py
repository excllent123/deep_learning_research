import unittest as ut 

from batch_generator import ImgOneTagGenerator

from ops_augmentator import ImgAugmentator

from datetime import datetime

from multiprocessing import cpu_count


import multiprocessing
import types

def _reduce_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)



if __name__ == '__main__':

    print ('CPU : ', cpu_count())

    O = ImgAugmentator()

    dir_list = ['/Users/kentchiu/MIT_Vedio/2D_DataSet/RHand','/Users/kentchiu/MIT_Vedio/2D_DataSet/Rhand_v2']

    gen = ImgOneTagGenerator(dir_list)

    gen.add_augment_op(O.resize, size=(10,10))
    gen.add_augment_op(O.normaliza)
    gen.add_augment_op(O.sample_mean_center)

    #############################
    print ('=======!!!====')
    start = datetime.now()
    s = gen.gen_balance_batch(4)

    for i in range(5):
        next(s)
        #gen.gen_balance_batch(4)
        
    print (datetime.now() - start)

    ############################

    start = datetime.now()
    s = gen.gen_balance_batch(4, pool_=True)
    for i in range(5):
        next(s)# => if generator => but generator not able to multiprocssing 
        #gen.gen_balance_batch(4, pool_=True)
    print (datetime.now() - start)
