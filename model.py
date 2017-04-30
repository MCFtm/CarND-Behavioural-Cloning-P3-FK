
# coding: utf-8

# In[ ]:

import csv
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.convolutional import Cropping2D


# In[ ]:

lines = []
#with open('data_as_shipped/driving_log.csv') as csvfile:
with open('./data_FK/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# In[ ]:

images=[]
measurements=[]
for line in lines:
    source_path=line[0]
    filename = source_path.split('/')[-1]
    current_path = './data_FK/IMG/'+filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement=float(line[3])
    measurements.append(measurement)
    image_flipped=np.fliplr(image)
    images.append(image_flipped)
    measurements.append(-measurement) 


# In[ ]:

X_train=np.array(images)
y_train=np.array(measurements)


# In[ ]:

model=Sequential()
model.add(Lambda (lambda x: x/255-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# In[ ]:

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)
model.save('model.h5')
exit()

