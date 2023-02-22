import os
import pickle

import numpy as np

# from keras.preprocessing import image
# import tensorflow as tf
# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace

# from tf.keras.preprocessing import image_dataset_from_directory

# from tensorflow.keras.preprocessing import image

actors = os.listdir("dataset")
filenames = []
for actor in actors:
    for file in os.listdir(os.path.join("dataset", actor)):
        filenames.append(os.path.join("dataset", actor, file))

pickle.dump(filenames, open("filenames.pkl", "wb"))

# filenames = pickle.load(open("filenames.pkl", "rb"))
# model = VGGFace(
#     model="resnet50",
#     include_top=False,
#     input_shape=(224, 224, 3),
#     pooling="avg",
# )
# model.summary()
