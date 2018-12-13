import numpy as np
import pandas as pd 
from keras import backend as K
from keras.datasets import cifar10
from keras.models import load_model

#Load and prep the data to test 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of y_train: {y_train.shape}\n")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_test: {y_test.shape}\n")

img_row, img_col = 32, 32 # size of tensor 
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

# Load the model 
print('Loading model...')
model = load_model('MyModel_Cifar10.h5')

# Predictions key = {0: 'Ariplane',}
key = ['Ariplane', 'Automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
df = pd.DataFrame(columns=['Predicted', 'Correct'])
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
for x in pred:
	df = df.append({'Predicted': key[x]}, ignore_index=True)
actu = []
for x in y_test:
	index = x[0]
	actu.append(key[index])

df.insert(0, 'Actual', actu)
df['Correct'] = df.apply(lambda x: 'True' if x['Predicted'] == x['Actual'] else 'False', axis=1)
print(df[:10])
df.to_csv('CIFAR_out.csv', index=False)