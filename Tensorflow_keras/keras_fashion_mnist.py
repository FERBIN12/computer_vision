# fashion keras mnist data set we will try with and without regularisation

from tensorflow.keras.datasets import fashion_mnist

# load the train and test data
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())