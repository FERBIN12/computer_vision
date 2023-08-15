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

## creating the CNN layers for the training

from tensorflow.keras.models import Sequential ## means a model in which each layer is connected to the next 
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import SGD

 
# create a model 
model=Sequential()

# we create our first convoultional layer with filter size 32 and a layer size of 26*26*32 and for activation we use Relu and set our input image

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))

# second layer is has 64 filer size of 64 and the size of the layer would be 24,24,64    ## we have gone with the default value of padding and stride

model.add(Conv2D(64,(3,3),activation='relu'))

# next we add the max pool layer 
model.add(MaxPooling2D(pool_size=(2,2)))

# then we flatten the layer of the CNN 
model.add(Flatten())

# we connect the flattend to a fully connected layer of size 128  using Dense function
model.add(Dense(128,activation='relu'))

# Then we finally connect the output to our final layer with an output for each class ie) 10

model.add(Dense(num_classes,activation='softmax'))

# then we compile the whole model using compile function mention the needed loss functions 
# optimizer and metric ie accuracy and all

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.001),
              metrics=['accuracy'])
print(model.summary())


# training the model

batch_size=128
epochs=25


history=model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss',score[0])
print('Test accuracy',score[1])

