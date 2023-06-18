from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import shutil

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')



print('geldi')
model =tf.keras.models.load_model(r"C:\Users\abdul\trainedmodel.h5")#yolunda turkce karakter olmamasi icin burayta tasidim
print('yuklendi')
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)