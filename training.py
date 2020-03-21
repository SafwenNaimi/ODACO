import datetime
import os, sys, shutil
import numpy as np
from numpy import loadtxt
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers, applications
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input,Flatten
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard
import time




train_dir = 'new/train/'
validation_dir = 'new/validation/'


bacth_size = 16
epochs = 5
warmup_epocks = 2
learning_rate = 0.00001
warmup_learning_rate = 0.0001
height = 128
width = 128
colors = 3
n_classes = 2
es_patience = 18
rlrop_patience = 3
decay_drop = 0.5
based_model_last_block_layer_number = 0 #100

#NAME="SBS-{}".format(int(time.time()))

#tensorboard=TensorBoard(log_dir='logs\{}'.format(NAME))


train_datagen = ImageDataGenerator(
      rescale=1/255,
      rotation_range=10,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.5,
      brightness_range=[0.7,1.3],
      horizontal_flip=True,
      fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        batch_size= bacth_size,
        shuffle = True,
        #color_mode='grayscale',
        class_mode= 'categorical')

val_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size = bacth_size,
        shuffle=True,
#color_mode='grayscale',
        class_mode= 'categorical')

#test_generator = ImageDataGenerator(rescale=1/255).flow_from_directory(
#        test_dir,
#        target_size=(HEIGHT, WIDTH),
#        batch_size = 1,
#        class_mode= 'categorical',
#        shuffle = False)

import efficientnet.tfkeras as eft
from keras.layers import Flatten,GlobalMaxPooling2D
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization

mod = "Xception"
###################
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = applications.Xception(weights='imagenet', #à modifier
                                       include_top=False,
                                       input_tensor=input_tensor)
    x= GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    output = Dense(512, activation='relu', name='output')(x)
    output = Dropout(0.5)(output)
    model_prim = Model(input_tensor, output)
    final_output = Dense(n_out, activation='softmax', name='final_output')(model_prim.output)
    model = Model(input_tensor, final_output)

    return model

model = create_model(input_shape=(height, width, colors), n_out=n_classes)
##############


#baseModel = applications.Xception(weights="imagenet", include_top=False,
#	input_tensor=Input(shape=(224, 224, 3)))

#headModel = baseModel.output
#headModel = GlobalAveragePooling2D()(headModel)
#headModel = Flatten(name="flatten")(headModel)
#headModel = Dense(64, activation="relu")(headModel)
#headModel = Dropout(0.5)(headModel)
#headModel = Dense(2, activation="softmax")(headModel)

#model = Model(inputs=baseModel.input, outputs=headModel)

#for layer in baseModel.layers:
	#layer.trainable = False



from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform


###############
for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True
##############

#new_model = tf.keras.models.load_model("fin.h5",custom_objects={'GlorotUniform': glorot_uniform()})


import tensorflow_model_optimization as tfmot


#pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
#                        initial_sparsity=0.0, final_sparsity=0.5, #0.5-->92.3
#                        begin_step=100, end_step=5000)

#pruned_model=Sequential()

#for layer in model.layers:

#    if (layer.name=='output') or (layer.name=='final_output') or (layer.name=='conv2d') or (layer.name=='conv2d_1') :

#        pruned_model.add(tfmot.sparsity.keras.prune_low_magnitude(layer,pruning_schedule=pruning_schedule))
#    else:
#        try:
#            pruned_model.add(layer)

#        except ValueError:
#            pass

#model=pruned_model







metric_list = ["accuracy"]
optimizer = optimizers.Adam(lr=warmup_learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
print(model.summary())



es = EarlyStopping(monitor='val_loss', mode='min', patience=es_patience, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=rlrop_patience, factor=decay_drop, min_lr=1e-6, verbose=1)

optimizer = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metric_list)




step_train = train_generator.n//train_generator.batch_size
step_validation = (val_generator.n//val_generator.batch_size)

print(train_generator.n)

checkpointer = ModelCheckpoint(filepath='new.h5',monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')


callback_list = [es, rlrop,checkpointer]#,tfmot.sparsity.keras.UpdatePruningStep()]


history_warmup = model.fit_generator(generator=train_generator,
                              steps_per_epoch=150,
                              validation_data=val_generator,
                              validation_steps=100,
                              callbacks=[checkpointer],#,tfmot.sparsity.keras.UpdatePruningStep()],
                              epochs=1,
                              verbose=1).history



history = model.fit_generator(generator=train_generator,
                             steps_per_epoch=150,
                             validation_data=val_generator,
                              validation_steps=100,
                              epochs=epochs,# revert to: epochs=EPOCHS
                              callbacks=callback_list,
                              verbose=1).history

#exported_model = tfmot.sparsity.keras.strip_pruning(pruned_model)


model.save('news.h5')

import matplotlib.pyplot as plt
import numpy as np

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history["val_accuracy"], label="val_acc")
plt.title("Training COVID-19 Dataset with " + mod)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("result.png")

