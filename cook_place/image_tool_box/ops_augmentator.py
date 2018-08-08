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
from skimage.io import imread, imshow
from skimage.transform import resize as sk_resize
import random 
import scipy.ndimage as ndi


class ImgAugmentator(object):
    '''Interface for augmentation-ops

    # Constrains
        - img must be numpy 3D tensor with  [H, W, Channel]
        - Returns only img

    # Single Usage : 
        - if you already have a batch flow, you could also use it only on batch_imgs
        ```
        augment_A = ImgAugmentator()
        for x in batch_x:
            x = augment_A.resize(x, size)
            x = augment_A.featurewise_center(x)
            x = augment_A.random_rotation(x, 90)

        augment_B = ImgAugmentator()
        for x in batch_x2:
            x = augment_B.resize(x, size)
            x = augment_B.featurewise_center(x)
            x = augment_B.random_rotation(x, 90)

        ```
    '''

    def two_tail_normalize(self, x):
        '''Performs -=127.5 and *=1/127.
        '''
        x -= 127.5
        x *= 1/127.
        return x

    def normaliza(self, x):
        '''Performs *=1/255.'''
        x *= 1/255.
        return x

    def rescale(self, x, factor):
        '''Performs rescale of the image factor 
        '''
        img = x*factor
        return img

    def resize(self, x, size):
        '''Performs resizing the image like cv2.resize

        # Args:
          - size : the size of img_height and img width, ex (200,200)
        '''
        assert type(size)==tuple
        assert len(size)==2
        img = sk_resize(x, size)
        return img

    def sample_mean_center(self, x):
        '''Performs image_based mean-zero and rescale to [-1,1]'''
        mean = np.sum(x)
        x-=mean
        x*=1/np.max(abs(x))
        return x

    def featurewise_center(self, x):
        '''Performs x -= mean(x)'''
        img = x - np.mean(x)
        return img

    def featurewise_std_normalization(self, x):
        '''Performs x /= std(x)'''
        img = x / (np.std(x)+1e-7)
        return img

    def random_flip_axis(self, x, axis):
        '''Performs random flips of image 

        Args:
          - axis : axis=0 (row-axis) for vertical ; axis=1 for horizontal
        '''
        if np.random.random() > 0.49:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
        return x

    def random_rotation(self, x, rg, row_axis=0, col_axis=1, channel_axis=2,
        fill_mode='nearest', cval=0.):
        """Performs a random rotation of a Numpy image tensor.    

        # Arguments
            x: Input tensor. Must be 3D.
            rg: Rotation range, in degrees.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.    

        # Returns
            Rotated Numpy image tensor.
        """
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])    

        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self._transform_matrix_offset_center(rotation_matrix, h, w)
        x = self._apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x   

    def random_shift(self, x, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2, 
        fill_mode='nearest', cval=0.):
        """Performs a random spatial shift of a Numpy image tensor.    

        # Arguments
            x: Input tensor. Must be 3D.
            wrg: Width shift range, as a float fraction of the width.
            hrg: Height shift range, as a float fraction of the height.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.    

        # Returns
            Shifted Numpy image tensor.
        """
        h, w = x.shape[row_axis], x.shape[col_axis]
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])    

        transform_matrix = translation_matrix  # no need to do offset
        x = self._apply_transform(x, transform_matrix, channel_axis, 
                                 fill_mode, cval)
        return x    

    def random_shear(self, x, intensity, row_axis=0, col_axis=1, channel_axis=2,
        fill_mode='nearest', cval=0.):
        """Performs a random spatial shear of a Numpy image tensor.    

        # Arguments
            x: Input tensor. Must be 3D.
            intensity: Transformation intensity.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.    

        # Returns
            Sheared Numpy image tensor.
        """
        shear = np.random.uniform(-intensity, intensity)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])    

        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self._transform_matrix_offset_center(shear_matrix, h, w)
        x = self._apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x    

    def random_zoom(self, x, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
        fill_mode='nearest', cval=0.):
        """Performs a random spatial zoom of a Numpy image tensor.    

        # Arguments
            x: Input tensor. Must be 3D.
            zoom_range: Tuple of floats; zoom range for width and height.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.    

        # Returns
            Zoomed Numpy image tensor.    

        # Raises
            ValueError: if `zoom_range` isn't a tuple.
        """
        if len(zoom_range) != 2:
            raise ValueError('zoom_range should be a tuple or list of two floats. '
                             'Received arg: ', zoom_range)    

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])    

        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self._transform_matrix_offset_center(zoom_matrix, h, w)
        x = self._apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_channel_shift(self, x, intensity, channel_axis=2):
        '''
        '''
        x = np.rollaxis(x, channel_axis, 0)
        min_x, max_x = np.min(x), np.max(x)
        channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                          for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    def center_crop(self, x, crop_size):
        '''Performs center crop 
        Args :
          - crop_size : is the window with (h, w), assert tuple input

        Note : 
          - this operatoin would change the img size 
        '''
        assert type(crop_size)==tuple
        assert crop_size[0] < x.shape[0] # height 
        assert crop_size[1] < x.shape[1] # width

        centerw, centerh = x.shape[1]//2, x.shape[2]//2
        halfw, halfh = crop_size[0]//2, crop_size[1]//2
        return x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]

    def random_crop(self, x, crop_size, sync_seed=None):
        '''Performs random crop 

        Args:
          - crop_size : is the window with (h, w), assert tuple input

        Note : 
          - this operatoin would change the img size
        '''
        np.random.seed(sync_seed)
        w, h = x.shape[1], x.shape[2]
        rangew = (w - crop_size[0]) // 2
        rangeh = (h - crop_size[1]) // 2
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        return x[:, offsetw:offsetw+crop_size[0], offseth:offseth+crop_size[1]]

    def _transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def _apply_transform(self, x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
        """Apply the image transformation specified by a matrix.    

        # Arguments
            x: 2D numpy array, single image.
            transform_matrix: Numpy array specifying the geometric transformation.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.    

        # Returns
            The transformed version of the input.
        """
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x



def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]

