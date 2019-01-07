import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import cv2, os, random
import numpy as np

dir = 'C:/Users/akash/OneDrive/Desktop/Practice/tf/training_set/training_set'
categories = ['dogs', 'cats']
img_size = 50
train_data = []
X = []
y = []

def create_train_set():    
    for category in categories:
        path = os.path.join(dir, category)
        print(path)
        class_num = categories.index(category)        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size), interpolation=cv2.INTER_AREA)
                train_data.append([new_array, class_num])
            except Exception as e:
                print(e)

create_train_set()
random.shuffle(train_data)

for features, labels in train_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:] ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3) ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X, y, epochs=3, batch_size=32)


img_to_predict = cv2.imread('babdu.jpg', cv2.IMREAD_GRAYSCALE)
new_img_to_predict = cv2.resize(img_to_predict, (img_size, img_size), interpolation=cv2.INTER_AREA)
predict_this = new_img_to_predict.reshape(-1, img_size, img_size, 1)


predict = model.predict([predict_this])
print(categories[int(predict[0][0])])