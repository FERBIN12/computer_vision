## now we are gonna test Alexnet in CFIR10 dataset

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Activation
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# print out the shape of the datasets

print("Train shape :",x_train.shape)
print(x_train.shape[0],"Train samples")
print(x_test.shape[0],"Test samples")

# now we perform one hot encode the labels

num_classes=10
print(num_classes)

y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)

l2_reg=0.001

# Now we build the model

model=Sequential()

# 1st conv layer 
model.add(Conv2D(96,(11,11),
                 input_shape=x_train.shape[1:],
                 padding="same",kernel_regularizer=l2(l2_reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd conv layer
model.add(Conv2D(256,(5,5),padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 3rd conv layer
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),
                 padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# 4th conv layer
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024,(3,3),padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5th conv layer 
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024,(3,3),padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# first fc layer 

model.add(Flatten())
model.add(Dense(3072))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 2nd fc layer

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 3rd fc layer
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

print(model.summary())  

#Train the model '

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])

batch_size=64
epochs=25

history=model.fit(x_train,y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test,y_test),
                  shuffle=True)

scores=model.evaluate(x_test,y_test,verbose=1)
print('Test Loss :',scores[0])
print('Accuracy :',scores[1])