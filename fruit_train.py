import numpy as np
import sys
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 4

#test_ds=tf.keras.preprocessing.image_dataset_from_directory(sys.argv[2],image_size=(224,224),batch_size=batch_size)

#train_ds=tf.keras.preprocessing.image_dataset_from_directory(sys.argv[1],image_size=(224,224),batch_size=batch_size)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(sys.argv[1],validation_split=0.2,seed=123,image_size=(224,224),subset="training",batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(sys.argv[1],validation_split=0.2,seed=123,image_size=(224,224),subset="validation",batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
#test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomFlip(),
	layers.experimental.preprocessing.RandomRotation(0.2),
	layers.experimental.preprocessing.RandomZoom(0.1),
	layers.experimental.preprocessing.RandomWidth(0.2),
	layers.experimental.preprocessing.RandomHeight(0.2)
	])

num_classes = 101

preprocess_input = tf.keras.applications.vgg19.preprocess_input
base_model = tf.keras.applications.VGG19(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')

base_model.trainable = False

prediction_layer = tf.keras.layers.Dense(101,activation='softmax')

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(256,activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])

initial_epochs = 5

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath='fruit_modelv2',
	monitor='val_accuracy',
	mode='max',
	save_best_only=True)

history = model.fit(train_ds,epochs=initial_epochs,validation_data=val_ds,callbacks=[model_checkpoint_callback])#val_ds)
