"""
helper_utils: just helper functions

purpose: help keep the notebook clean and only for analysis
"""


import time, os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import (Image, )
import pprint
import json
from collections import (OrderedDict, )
import copy

batch_size = 10
train_split = 32
img_size = 224 # All images will be resized to 224x224

import warnings
warnings.simplefilter('ignore')

def get_class_names(f_class_names='label_map.json'):
    with open(f_class_names , 'r') as f:
        class_names = json.load(f)
    return class_names

def process_image(img):
    """
    Process image and get it ready ready for training/prediction
    """
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    img /= 155
    return img.numpy()

def predict(img_path, base_model, top_k):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    top_k = max(top_k, 1)
    img = Image.open(img_path)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = process_image(img)
    prb = base_model.predict(img)
    (top_k_val, top_k_ind) = tf.nn.top_k(prb, k=top_k)
    return  (top_k_val.numpy(), top_k_ind.numpy(), img)        
