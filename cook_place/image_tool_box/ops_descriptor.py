'''
# Description 

'''
from random import sample

class ImgDescriptor:
    '''
    # Constrains
      - img must be numpy 3D tensor with  [H, W, Channel]
      - Returns only one-dim feature vector

    # Functions
      - 
    '''
    pass



class OpsPipe:
    '''
    # Constrain 
      - all functions in ops_callbacks should always return img only
      - like in ImgAugmentator or ImgSegmentator/Unsupervisor
    '''
    def __init__(self):
        self.ops_callbacks=[]

    def add_augment_op(self, func, **kwargs):
        assert (func.__name__ in self.regist_ops) == True 

        for s in self._get_check_variable(func):
            if (s!='x') and (s!='self') and (s not in kwargs.keys()):
                raise ValueError('missing [{}] parameter in {} ''function \n'
                         '[Check] {}'.format(s, func.__name__, func.__doc__))

        self.ops_callbacks.append([func, kwargs])

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

    def _apply_augment_op(self, img):
        '''store : [func, {}] usage : func(img, **value)'''
        for func, value in self.ops_callbacks:
            img = func(img, **value)
        return img

    def imshow_many(img_list):
        ''' plot multi-image in jupyter-notebool    
        # Args:
          - img_list : [img1, img2, ... ]
        
        # Scenario:
          - double-comparison
          - show-all
          - show-sampling-data
        '''
        x = int(len(img_list)**(0.5))+1
        fig, axes = plt.subplots(x, x)
        ax = axes.ravel()
        for i in range(len(img_list)):
            ax[i].imshow(img_list[i])
        plt.show()

    def random_subsampling_imshow(population_path, sampling_size=9):
        '''random_subsampling imshow
        
        # Args:
          - callback like some ops that get img only
        
        '''    

        candidates = sample( range(0, len(population_path), sampling_size)    

        before_imgs = [ imread(population_path[i]) for i in candidates]     

        after_imgs = [ self._apply_augment_op(i) for i in before_imgs ]    

        imshow_many(list(zip(before_imgs, after_imgs)))


