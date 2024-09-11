import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from itertools import cycle


import scipy
import skimage
import sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from tensorflow.keras.callbacks import CSVLogger

import tensorflow as tf


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D,Reshape, multiply
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization


from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate, Activation
import matplotlib
import time

matplotlib.use('TkAgg')

print("Device: \n", tf.config.experimental.list_physical_devices())
print(tf.__version__)
print(tf.test.is_built_with_cuda())

images = np.load("D:/MN/gastro three stage/SaveFileForStage1/updated/Stage1X.npy")
y = np.load("D:/MN/gastro three stage/SaveFileForStage1/updated/Stage1Y.npy")

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=2)


################ PSE-CNN ####################

input_shape = images.shape[1:]
inp = Input(shape=input_shape)


# Define the Squeeze and Excite block
def squeeze_excite_block(input_layer, ratio=16):
    filters = input_layer.shape[-1]

    se = GlobalAveragePooling2D()(input_layer)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input_layer, se])
    return se

convs = []
parrallel_kernels = [11, 9, 7, 5, 3]
for k in range(len(parrallel_kernels)):
    conv = SeparableConv2D(256, parrallel_kernels[k], padding='same', activation='relu', input_shape=input_shape)(inp)
    conv = squeeze_excite_block(conv) # Integrate SE block
    convs.append(conv)

out = Concatenate()(convs)
conv_model = Model(inputs=inp, outputs=out)

model = Sequential()
model.add(conv_model)

model.add(SeparableConv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(16, (3, 3), padding='same', name='lastconv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(200, activation='relu', name='DenseLastPL'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(3, input_dim=124, activation='softmax'))
adam = tf.optimizers.Adam(lr=0.0001)

model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.summary()

mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/PSECNN.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

csv_logger = CSVLogger("D:/MN/gastro three stage/SaveFileForStage1/updated/PSECNN_history.csv", separator=',', append=True)
# Add the CSVLogger callback to the list of callbacks
callbacks = [mc, csv_logger]

start = time.time()

with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks)

end = time.time()

elapsed = end - start
print("Time:", elapsed)

# Load the training history from the CSV file
training_history = pd.read_csv("D:/MN/gastro three stage/SaveFileForStage1/updated/PSECNN_history.csv")

# Plot the loss curve
plt.figure(1)
plt.plot(training_history['loss'])
plt.plot(training_history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of PSECNN')
plt.xlabel('epoch')
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of PSECNN', dpi=600)

# Plot the accuracy curve
plt.figure(2)
plt.plot(training_history['acc'])
plt.plot(training_history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of PSECNN')
plt.xlabel('epoch')
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/AAccuracy of PSECNN')

# Show the plots
plt.show()

model.save("D:/MN/gastro three stage/SaveFileForStage1/updated/LastPSECNN.h5")


################ Pr-CNN ####################


input_shape = images.shape[1:]
inp = Input(shape=input_shape)

convs = []
parrallel_kernels = [11, 9, 7, 5, 3]
for k in range(len(parrallel_kernels)):
    conv = Conv2D(256, parrallel_kernels[k], padding='same', activation='relu', input_shape=input_shape)(inp)
    convs.append(conv)

out = Concatenate()(convs)
conv_model = Model(inputs=inp, outputs=out)

model = Sequential()
model.add(conv_model)

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), name='lastconv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(200, activation='relu', name='DenseLastPL'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(3, input_dim=124, activation='softmax'))
adam = tf.optimizers.Adam(lr=0.001)

model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.summary()

mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/PRCNN.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

csv_logger = CSVLogger("D:/MN/gastro three stage/SaveFileForStage1/updated/PRCNN_history.csv", separator=',', append=True)

# Add the CSVLogger callback to the list of callbacks
callbacks = [mc, csv_logger]

start = time.time()

with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks)

end = time.time()

elapsed = end - start
print("Time:", elapsed)

# Load the training history from the CSV file
training_history = pd.read_csv("D:/MN/gastro three stage/SaveFileForStage1/updated/PRCNN_history.csv")

# Plot the loss curve
plt.figure(3)
plt.plot(training_history['loss'])
plt.plot(training_history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of PRCNN')
plt.xlabel('epoch')
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of PRCNN')

# Plot the accuracy curve
plt.figure(4)
plt.plot(training_history['acc'])
plt.plot(training_history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of PRCNN')
plt.xlabel('epoch')
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Accuracy of PRCNN')

# Show the plots
plt.show()

model.save("D:/MN/gastro three stage/SaveFileForStage1/updated/LastPRCNN.h5")



################################ DenseNet201 ##############################################


from tensorflow.python.keras.applications.densenet import DenseNet201


densenet = DenseNet201(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in densenet.layers:
    layer.trainable = False
x = Flatten()(densenet.output)

x = Dense(1024, activation='relu')(x)
x = Dense(200, activation='relu', name='LastDenseNet201')(x)
prediction = Dense(3, activation='softmax')(x)

# create a model object
model_densenet = Model(inputs=densenet.input, outputs=prediction)
model_densenet.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')


model_densenet.summary()
mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/TDenseNet201.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

start = time.time()

with tf.device('/GPU:0'):
    history = model_densenet.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=[mc])

end = time.time()

elapsed = end - start
print("Time:", elapsed)


plt.figure(15)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of DenseNet201')
plt.xlabel('epoch')
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of DenseNet201')

plt.figure(16)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of DenseNet201')
plt.xlabel('epoch')
plt.show()
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Accuracy of DenseNet201')



###########################InceptionResNetV2################################

from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2


densenet = InceptionResNetV2(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in densenet.layers:
    layer.trainable = False
x = Flatten()(densenet.output)
x = Dense(1024, activation='relu')(x)
x = Dense(200, activation='relu', name='LastInceptionResNetV2')(x)
prediction = Dense(3, activation='softmax')(x)

# create a model object
model_densenet = Model(inputs=densenet.input, outputs=prediction)
model_densenet.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')


model_densenet.summary()
mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/TInceptionResNetV2.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

start = time.time()

with tf.device('/GPU:0'):
    history = model_densenet.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=[mc])

end = time.time()

elapsed = end - start
print("Time:", elapsed)


plt.figure(13)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of InceptionResNetV2')
plt.xlabel('epoch')
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of InceptionResNetV2')

plt.figure(14)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of InceptionResNetV2')
plt.xlabel('epoch')
plt.show()
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Accuracy of InceptionResNetV2')



###########################Xception################################

from tensorflow.python.keras.applications.xception import Xception


densenet = Xception(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in densenet.layers:
    layer.trainable = False
x = Flatten()(densenet.output)
x = Dense(1024, activation='relu')(x)
x = Dense(200, activation='relu', name='LastXception')(x)
prediction = Dense(3, activation='softmax')(x)

# create a model object
model_densenet = Model(inputs=densenet.input, outputs=prediction)
model_densenet.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')


model_densenet.summary()
mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/TXception.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

start = time.time()

with tf.device('/GPU:0'):
    history = model_densenet.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=[mc])

end = time.time()

elapsed = end - start
print("Time:", elapsed)


plt.figure(11)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of Xception')
plt.xlabel('epoch')
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of Xception')

plt.figure(12)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of Xception')
plt.xlabel('epoch')
plt.show()
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Accuracy of Xception')


###########################MobileNetV2################################

from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2


densenet = MobileNetV2(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in densenet.layers:
    layer.trainable = False
x = Flatten()(densenet.output)
x = Dense(1024, activation='relu')(x)
x = Dense(200, activation='relu', name='LastMobileNetV2')(x)
prediction = Dense(3, activation='softmax')(x)

# create a model object
model_densenet = Model(inputs=densenet.input, outputs=prediction)
model_densenet.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')


model_densenet.summary()
mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/TMobileNetV2.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

start = time.time()

with tf.device('/GPU:0'):
    history = model_densenet.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=[mc])

end = time.time()

elapsed = end - start
print("Time:", elapsed)


plt.figure(9)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of MobileNetV2')
plt.xlabel('epoch')
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of MobileNetV2')

plt.figure(10)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of MobileNetV2')
plt.xlabel('epoch')
plt.show()
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Accuracy of MobileNetV2')


####################  vgg 16 #######################

from tensorflow.python.keras.applications.vgg16 import VGG16


densenet = VGG16(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in densenet.layers:
    layer.trainable = False
x = Flatten()(densenet.output)
x = Dense(1024, activation='relu')(x)
x = Dense(200, activation='relu', name='LastVGG16')(x)
prediction = Dense(3, activation='softmax')(x)

# create a model object
model_densenet = Model(inputs=densenet.input, outputs=prediction)
model_densenet.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')


model_densenet.summary()
mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/TVGG16.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

start = time.time()

with tf.device('/GPU:0'):
    history = model_densenet.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=[mc])

end = time.time()

elapsed = end - start
print("Time:", elapsed)


plt.figure(23)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of VGG16')
plt.xlabel('epoch')
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of VGG16')

plt.figure(24)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of VGG16')
plt.xlabel('epoch')
plt.show()
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Accuracy of VGG16')



########## ResNet152V2 #####################

from tensorflow.python.keras.applications.resnet_v2 import ResNet152V2


densenet = ResNet152V2(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in densenet.layers:
    layer.trainable = False
x = Flatten()(densenet.output)
x = Dense(1024, activation='relu')(x)
x = Dense(200, activation='relu', name='LastResNet152V2')(x)
prediction = Dense(3, activation='softmax')(x)

# create a model object
model_densenet = Model(inputs=densenet.input, outputs=prediction)
model_densenet.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')


model_densenet.summary()
mc = ModelCheckpoint("D:/MN/gastro three stage/SaveFileForStage1/updated/TResNet152V2.h5", monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

start = time.time()

with tf.device('/GPU:0'):
    history = model_densenet.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=[mc])

end = time.time()

elapsed = end - start
print("Time:", elapsed)


plt.figure(21)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss of ResNet152V2')
plt.xlabel('epoch')
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Loss of ResNet152V2')

plt.figure(22)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy of ResNet152V2')
plt.xlabel('epoch')
plt.show()
plt.pause(1)
plt.savefig('D:/MN/gastro three stage/SaveFileForStage1/updated/Accuracy of ResNet152V2')