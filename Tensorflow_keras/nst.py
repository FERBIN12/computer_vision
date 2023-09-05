## neural style transfer

import numpy as np
import tensorflow as tf
import IPython.display as display
import os
import time,functools
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL.Image

# load compresses models from the tf hub
os.environ['TFHUB_MODEL _LOAD_FORMAT']='COMPRESSED'

# now set image plot and axis params
mpl.rcParams['figure.figsize']=(12,12)
mpl.rcParams['axes.grid']=False

def tensor_to_image(tensor):
  tensor=255*tensor   # image pixels vary form 0 to 255 so the tensor is multiplied by 255
  tensor=np.array(tensor,dtype=np.uint8)  # converts the scalar tensor into numpy array and the pixels are generaly take as unssigned 8 bits integers
  if np.ndim(tensor)>3:  # if the dimensions of the tensor is more than 3 the condition activates
    assert tensor.shape[0] ==1  # arrises an assertion error when there are more than 1 image in the batch
    tensor=tensor[0]  # if there are more images only the first one in the batch is taken
  return PIL.Image.fromarray(tensor)  

# load our content and style images
content_path = tf.keras.utils.get_file('labrador.jpeg', 'https://github.com/rajeevratan84/ModernComputerVision/raw/main/labrador.jpeg')
style_path = tf.keras.utils.get_file('the_wave.jpg','https://github.com/rajeevratan84/ModernComputerVision/raw/main/the_wave.jpg')


# mosaic - https://github.com/rajeevratan84/ModernComputerVision/raw/main/mosaic.jpg
# feathers - https://github.com/rajeevratan84/ModernComputerVision/raw/main/feathers.jpg


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

import tensorflow_hub as hub

hub_model=hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

image=hub_model(tf.constant(content_image),tf.constant(style_image))[0]

# then convert the following tensor to img

tensor_to_image(image)
