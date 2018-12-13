import keras
import time 
import numpy as np
import pandas as pd 
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from MyFunctions import MyFunctions as mf

# Load the data 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of y_train: {y_train.shape}\n")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_test: {y_test.shape}\n")


img_row, img_col = 32, 32
batch_size = 128
num_class = 10 
epochs = 15

if K.image_data_format() == 'channels_first':
	print('Channel first')
	x_train = x_train.reshape(x_train.shape[0],3,img_row,img_col)
	x_test = x_test.reshape(x_test.shape[0],3,img_row,img_col)
	input_shape = (3, img_row, img_col)
else:
	print('Channel last')
	x_train = x_train.reshape(x_train.shape[0],img_row,img_col,3)
	x_test = x_test.reshape(x_test.shape[0],img_row,img_col,3)
	input_shape = (img_row, img_col, 3)

print(x_train.shape)
# Normalize the input to be between 0 and 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /=255
print(f"x_train shape: {x_train.shape}")
print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)

# Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

start_time = time.time()
model.fit(x_train, y_train, validation_data=(x_test,y_test), verbose=1, epochs=epochs, batch_size=batch_size)
score = model.evaluate(x_test, y_test)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")
elapsed_time = time.time() - start_time
print(f'Elapsed Time: {mf.hms_string(elapsed_time)}')
model.save('MyModel_Cifar10.h5')
