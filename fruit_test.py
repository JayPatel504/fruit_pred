import numpy as np
import sys
import os
import PIL
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 4

test_ds=tf.keras.preprocessing.image_dataset_from_directory(sys.argv[1],image_size=(224,224),shuffle=False,batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.load_model(sys.argv[2])

results=model.evaluate(test_ds)

print("test err: ",1-results[1])
