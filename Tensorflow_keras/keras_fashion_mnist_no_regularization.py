# fashion keras mnist data set we will try with and without regularisation

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD


# load the train and test data
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(device_lib.list_local_devices())

# now we start the pre procesing for keras ie) storing the no of rows and cols 

img_rows=x_train[0].shape[0]
img_cols=x_train[0].shape[1]

# we store the shape of the image
input_shape=(img_rows,img_cols,1)
print(input_shape)

# Now we one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Let's count the number columns in our hot encoded matrix 
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]

# we create a model
model=Sequential()

# now we start builing our CNN layers
# we create a convolutional layer using relu activation function of kernel size=(3,3)
model.add(Conv2D(32,kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
# we can add a second conv layer with 64 filtes ,3X3relu activation
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=input_shape))

# now we maxpool them with 2X2 with stride 2
model.add(MaxPooling2D(pool_size=(2,2)))
# flatten the last layer into in default the output will be 12 X 12 X 64 X 1 =9216
model.add(Flatten())

# now we connect this layer to a 128 nodes of our fully connected layer again using relu
model.add(Dense(128,activation='relu'))
# now we create our final fully connected layer using number of nodes as 10 and use softmax as 
# actuivation layer 
model.add(Dense(num_classes,activation='softmax'))

# now we complile our model by mentioning the loss function, optimizer and the metrices
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.001,momentum=0.9),
              metrics=['accuracy'])

# we can use summary function to see the params
print(model.summary())


batch_size=32
epochs=15

history=model.fit(x_train,
                  y_train,
                  batch_size=batch_size, # no of samples used per iteration
                  epochs=epochs, # no of times its trained
                  verbose=2,  # determines the no of information that will be displayed using training
                  validation_data=(x_test,y_test) 
                  )
score=model.evaluate(x_test,y_test,verbose=0)
print('test loss:',score[0])
print('Test accuracy',score[1])


