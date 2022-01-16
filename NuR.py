import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split  # it is model to split Our Data
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense ,Dropout,Flatten
from keras.optimizers import adam_v2
from keras.layers.convolutional import Conv2D,MaxPooling2D
import tensorflow as tf
from tensorflow.keras import layers
########################################
path = 'MyData'
testRatio = 0.2
valRatio = 0.2
imageDimention=(32,32,3)
#########################################
images = []
classNo = []
myList = os.listdir(path)
print("Total Number of Classes Detected ", len(myList))
noOfClasses = len(myList)

print("Importing Classes .....")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))  # First iteration for reading folders
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)  # Second iteration for reading images inside each folder
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)  # add images to list
        classNo.append(x)
    print(x, end=" ")
print(" ")

############# Convert to Numpy List###################

images = np.array(images)
classNo = np.array(classNo)
print("Print Shapes ....")
print(images.shape)
print(classNo.shape)

print("Shape After Splitting ....")
############################## Splitting Data ####################

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

noOfSamples = []
for x in range(0, noOfClasses):
    # print(len(np.where(y_train==x)[0])) # number of images in training mode for each ID
    noOfSamples.append(len(np.where(y_train == x)[0]))
print("Number Of samples in each ID....")
print(noOfSamples)

print("Show Figure for our Data")
plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), noOfSamples)
plt.title("No For Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("No OF images")
plt.show()

print("print before a preprocessing .... in 3 channel .. RGB")
print(X_train[30].shape)


################### Pre Processing######

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


# img = preprocessing (X_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed",img)
# cv2.waitKey(0)

print("print as after preprocessing in one channel.....Gray Scale ")
X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))
# print(X_train[30].shape)

# Add Depth for reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2])

# Data Generation


dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.2, shear_range=0.2,
                             rotation_range=10)

dataGen.fit(X_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

##################################### Training Model Confguration #################

def myModel():
    noOfFilters=60
    sizeOfFilters1=(5,5)
    sizeOfFilter2 =(3,3)
    sizeOfPool=(2,2)
    noOfNode=500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilters1,
                      input_shape=(imageDimention[0],imageDimention[0],1),
                      activation='relu'
               )))
    model.add((Conv2D(noOfFilters, sizeOfFilters1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

model=myModel()
print(model.summary())

