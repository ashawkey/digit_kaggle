import os, time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

np.random.seed(42)

train = pd.read_csv("./digit_recognizer/train.csv")
train_X = train.values[:,1:]/255.0
train_X = train_X.reshape([-1,28,28,1])
train_y = train.values[:,0]

test_X = pd.read_csv("./digit_recognizer/test.csv").values/255.0
test_X = test_X.reshape([-1,28,28,1])

n_classes = len(set(train_y))
n_train = train_X.shape[0]
n_test = test_X.shape[0]

train_y = to_categorical(train_y, n_classes)

train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1)

# keras model

model = Sequential()
model.add(Conv2D(input_shape=[28,28,1], filters=32, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(filters=32, kernel_size=5, activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=n_classes, activation='softmax'))
optimizer = RMSprop(lr=0.001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

### training -----

epochs = 30 
batch_size = 86

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(train_X)

start = time.time()
model.fit_generator(datagen.flow(train_X, train_y, batch_size=batch_size),
                    epochs = epochs,
                    validation_data = (val_X,val_y),
                    steps_per_epoch=train_X.shape[0] // batch_size
                    )

print("training finished in {} seconds".format(time.time()-start))
model.save("mnist_model.h5")
    
pred = model.predict(test_X)
pred = np.argmax(pred, axis=1)

print("testing finished in {} seconds".format(time.time()-start))

results = pd.Series(pred, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
