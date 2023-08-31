
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta

(x_train,y_train),(x_test,y_test)= mnist.load_data()


# lets get the no of img cols and rowsss
img_rows=x_train[0].shape[0]
img_cols=x_train[1].shape[0]

# Then we reashape the imagesss
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
print(x_train.shape[0])
# then we get the input shape
input_shape=(img_rows,img_cols,1)
print(input_shape)

# then we change the input dattype to float32 and normalize them '
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train /= 255
x_test /= 255

# then we perform hot encode on the labels
y_train=to_categorical(x_train)
y_test=to_categorical(x_test)

num_classes=y_test.shape[1]
num_pixels=x_train.shape[1] * x_train.shape[2]

# building the model

model=Sequential()

# building the layers

model.add(Conv2D(6,(5,5),
                 padding="same",
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# second conv layer

model.add(Conv2D(16,(5,5),
                 padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# third conv layer
model.add(Conv2D(120,(5,5),
                 padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# now built the fc layers
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# now we compile them with the loss function,optimizer and metrics

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])

print(model.summary())

# train the model

batch_size=128
epochs=50

history=model.fit(x_train,y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test,y_test),
                  shuffle=True)

model.save('mnits_Lenet.h5')

# get the score for getting the accuracy and loss

scores=model.evaluate(x_test,y_test,verbose=1)

print('Test Loss:',scores[0])
print('Accuracy :',scores[1])