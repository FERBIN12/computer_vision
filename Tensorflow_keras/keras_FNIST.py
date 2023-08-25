# We load our data directly from the included datasets in tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib


# loads the Fashion-MNIST training and test dataset 
(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()

# Our Class Names, when loading data from .datasets() our classes are integers
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Check to see if we're using the GPU
print(device_lib.list_local_devices())

# Display the number of samples in x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))


# we now reshape our data into no of samples,width,height and color depth

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

# now change the datatype to float32
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

# now we get the no of rows and cols and the shape of the image

img_rows=x_train[0].shape[0]
img_cols=x_train[0].shape[1]

# i/p image shape
input_shape=(img_rows,img_cols,1)
print(input_shape)

# normalize our data between 0/1

x_test /=255.0


