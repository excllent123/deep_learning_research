

import cv2

class RGBImage(Descriptor):
    '''This class is basicall for Deep Learning Part'''
    def __init__(self):
        pass
    def describe(self, image):
        try:
            img = image[:,:,:3]
        except:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
