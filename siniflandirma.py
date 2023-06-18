from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import shutil
import matplotlib.pyplot as plt



data = np.load(r"C:\Users\abdul\OneDrive\Masaüstü\pythonproje\imagearrays\224x224_images.npy")
labels = np.load(r"C:\Users\abdul\OneDrive\Masaüstü\pythonproje\imagearrays\224x224_labels.npy")
data.shape
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


labelEn = LabelEncoder() #string olan etiketleri 0 1 2 şeklinde kodla
labels = labelEn.fit_transform(labels)
labels = to_categorical(labels)

print(labels)

# physical_devices = tf.config.list_physical_devices('GPU')
# print("Available GPUs:", len(physical_devices))
# #yüzde 10 validasyon ve yüzde 90 test verisi olacak şekilde 2. bir ayırma yapıyoruz
# x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = .10, shuffle = True,random_state=42)

# device_name = tf.test.gpu_device_name()
# if device_name != '/DESKTOP-3JKP16V:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# train -test split
#%10 test %90 eğitimv ve valıdasyon seti olacak şekilde böl
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = .10, shuffle = True)


print(
"""
x_train shape: {}
x_test shape: {}
y_train shape: {}
y_test shape: {}

""".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = .10, shuffle = True,random_state=42)
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
np.save('x_validate.npy', x_validate)
np.save('y_validate.npy', y_validate)



#modelimin kurgusu
def createModel():
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(224, 224, 1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))


    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.25))
   
    model.add(tf.keras.layers.Conv2D(64, (3, 3),padding='same',  activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),  activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
   
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3),padding='same',  activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),  activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
   
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
   
  
    model.add(tf.keras.layers.Flatten())
   
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    return model


model1 = createModel()
batch_size =24
epochs =25
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1,  factor=0.5, min_lr=0.00001)
history = model1.fit(x_train , y_train , batch_size=batch_size , epochs = epochs , validation_data = (x_validate,y_validate) , verbose = 1 , callbacks=[learning_rate_reduction])



 

model1.save('trainedmodel.h5')
# Plot accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



