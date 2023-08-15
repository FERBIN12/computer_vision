from tensorflow.keras.datasets import mnist

## load the MNISt training and test dataset

(x_train,y_train) , (x_test,y_test) = mnist.load_data()

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# lets store the num of rows and columns 

img_rows=x_train[0].shape[0]
img_cols=x_train[0].shape[1]

# then we reshape to addd 4th dimension

x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

# we are storing the shape of a single image here

input_shape=(img_rows,img_cols, 1)

# change the datatype

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

# normalizing the range from (0,255) to (0,1)

x_train /= 255.0
x_test /= 255.0

print("The shape of the code is :",x_train.shape)
print(x_train.shape[0],"Train_sample")
print(x_test.shape[0],"Test_samples")

from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# lets count the no of cols in our hot encoded matrix

print("The num of classes is :",str(y_test.shape[1]))

num_classes=y_test.shape[1]
num_pixels=x_train.shape[1]*x_train.shape[2]