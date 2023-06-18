
import numpy as np 
import random
import math
import csv
import cv2
import os
import tensorflow as tf


inputbasepath = r"C:\Users\abdul\OneDrive\Masaüstü\pythonproje\images"
outputbasepath = r"C:\Users\abdul\OneDrive\Masaüstü\pythonproje\imagesarrays"



image_width = 224
image_height = 224

classes = ['epidural','intraparanchymal','intraparanthukal','nodata','subarachnoid','subdural']

os.chdir(inputbasepath) #chdir -> change directory, inputBasePath yoluyla verilen dizine git

X = [] # resimleri yani girdileri yani X değerlerini tutmak için dizi
Y = [] # etiketleri yani Y değerlerini tutmak için dizi. her bir resmin etiketi içinde yer aldığı klasörün adı zaten

i = 0
for class1 in classes:
  os.chdir(class1) #base yoldan sonra sıradaki sınıfı gösteren klasöre konumlan
  print('=> '+class1) #o an üzerinde bulunulan sınıfı (klasör adını) yaz
  for files in os.listdir('./'): # nokta mevcut dizini gösteriyor. ./ mevcut dizin altındakiler
    img = cv2.imread(files,0) #dosya yolundan resmi binary array olarak okuma.resmi grayscale almak için ikinci parametreye 0 yazılır cv2.imread(files,0)
    img = cv2.resize(img, (image_width,image_height)) #isteğe bağlı olarak resize edilebilir
    X.append(img) #resmi oluşturan bit dizisini X'e ekle
    Y.append(class1) # bu resmin sınıfı içinde bulunduğu klasör adı. resmin etiketi olarak bunu Y'ye ekle
    i = i + 1
  os.chdir('..') #bir üst dizine çık. bu sınıfla ve bunu içeren klasörler işimiz bitti
  
print("X : ",len(X))
print("Y : ",len(Y))


X = np.array(X).reshape(-1,image_width,image_height,1) #-1 ile verilen ilk değerin yerinde toplam resim adedi var; 
                #bu aynı kalacak. diğer parametreler verilen width ve height'e göre ve resmin renkli olduğunu 
                #belirten 3 ile yeniden şekillendirilecek
Y = np.array(Y) #etiket adlarını içeren Y'yi reshape etmeye gerek yok. 

print("X : ",X.shape)
print("Y : ",Y.shape)

print("X : ",len(X))
print("Y : ",len(Y))

os.chdir('..') #bir üst dizine daha çıkıp sonra imagearrays klasörüne gidersek zaten outputBasePath'e ulaşmış olacağız
os.chdir("imagearrays")
# üstteki iki satır yerine bunu direkt chdir(outputBasePath) olarak da yapabilirdik
np.save(str(image_width)+'x'+str(image_height)+'_images', X) #diziyi kaydederken dosya adını en x boy_images olarak adlandır.
                                                            #'224x224_images' gibi
np.save(str(image_width)+'x'+str(image_height)+'_labels', Y) #diziyi kaydederken dosya adını en x boy_labels olarak adlandır.

print("[ INFO - STAGE1 ]  NUMPY ARRAY CREATION COMPLETED \n ")

