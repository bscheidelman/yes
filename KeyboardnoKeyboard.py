import numpy as np
#numbers/math help

import matplotlib as plt
%matplotlib inline
#helping with the graphs/visualization of data

import keras
from keras.models import Sequential
from keras.layers import Dense
#for helping with the neural network

import pandas as pd
#tablular format

from google.colab import files

import time
import itertools

from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

#import libraries for simplifying workload of code
data1 = pd.read_csv('answers.txt')
#data2 = np.loadtxt("voltage.txt", delimiter=',')
#data2 = pd.read_csv('voltage.txt', header = None, names = ['f1', 'f2', 'f3', 'f4', 'f5']).to_numpy
data2 = pd.read_csv('voltage.txt')
print(data2)
#visualize data for voltage
print(data2.shape)
#prep data for training
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["y", "h", "n", "u", "j", "m", "i", "k", "o", "l", "p", "nokey"])
letter_classes = list(le.classes_)
print(letter_classes)
le.transform(["y", "h", "n", "u", "j", "m", "i", "k", "o", "l", "p", "nokey"]) 
list(le.inverse_transform([ 1, 0,  5,  2, 10,  1,  3,  4,  8]))
data1
np.unique(data1, return_counts=True )
#define variables
#variables f1 f2 f3 f4 f5
def normalizer(val, min_val, max_val):
  return (val - min_val)/ (max_val - min_val)
X = data2
print(X)
X = normalizer(X, np.min(X), np.max(X))
print(X)
y = data1.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_classes = 12
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
np.unique(data1, return_counts = True)
#wide nn for multible keys for variables
#density layers 5-?-?-1
model = Sequential()
model.add(Dense(5, kernel_initializer='normal', activation='relu', input_dim=5))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(units=12, activation='softmax'))
#compile model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
print(X_train[0:20])
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
#fit the model
history = y_pred = model.fit(X_train, y_train, epochs=35, batch_size=5, validation_split = 0.8, verbose=True)
#epochs vs improv
loss_train, acc_train  = model.evaluate(X_train, y_train, verbose=False)
loss_test, acc_test  = model.evaluate(X_test, y_test, verbose=False)
print(f'Train acc/loss: {acc_train:.3}, {loss_train:.3}')
print(f'Test acc/loss: {acc_test:.3}, {loss_test:.3}')
y_pred_train = model.predict(X_train, verbose=True)
print(np.shape(y_pred_train))
y_pred_test = model.predict(X_test,verbose=True)
print(np.shape(y_pred_test))
#epochs vs improv
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
f = figure(num=None, figsize=(12,5), dpi=80, facecolor='w', edgecolor='k')
f.add_subplot(1,2, 1)

# plot accuracy as a function of epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')

# plot loss as a function of epoch
f.add_subplot(1,2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show(block=True)
from sklearn.metrics import confusion_matrix
plt.plot(y_test, y_pred_test)
print(np.shape(y_test))
print(np.shape(y_train))
print(type(y_test))
print(type(y_pred))
y_predt_array=np.asarray(y_pred)