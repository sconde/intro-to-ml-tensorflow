'''
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

e.g.: python predict.py 'flowers/test/1/image_06760.jpg' 'checkpoint.pth' 5 'gpu'
'''

from time import (time,)
from IPython.display import (display,)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import ( nn, optim, )
from torch.optim import ( lr_scheduler, )
from torch.autograd import ( Variable, )

    
from PIL import ( Image, )

import torchvision

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()
import argparse
import json
import helper_utils

import pdb


parser = argparse.ArgumentParser(
    description="Predict flower name from an image with a pre-trained model: pass a single image, return the flower name and class probability")

parser.add_argument('input',
                    default='./test_images/orchid.jpg',
                    type=str,
                    help='choose the image path')
parser.add_argument('checkpoint',
                    default="./saved_model/base_model",
                    type=str,
                    help='give the checkpoint')
parser.add_argument('--top_k',
                    default=5,
                    dest="top_k",
                    type=int,
                    help='num of top predicted classes')
parser.add_argument('--category_names',
                    default='label_map.json',
                    type=str,
                    help='')
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

print("TensorFlow version: ", tf.__version__)
print("TensorFlow Keras version: ", tf.keras.__version__)
print("Running on GPU: ", tf.config.list_physical_devices('GPU'))

def main():
    args_parsed = parser.parse_args()

    if args_parsed.verbose: pprint.pprint(args_parsed.__dict__)
    args = args_parsed.__dict__


    # get the category name
    category_names = helper_utils.get_class_names(args['category_names'])

    # load the model
    based_model_path = args['checkpoint']
    print(f"input: {args['input']}")

    #pdb.set_trace()
    with tf.device('/cpu:0'):
        #reloaded_model = tf.keras.models.load_model(based_model_path)
        loaded_model = tf.keras.models.load(based_model_path, clear_devices=True)
        if args_parsed.verbose: print(loaded_model.classifier)

        
        # predict
        (top_k_val, top_k_ind, img) = helper_utils.predict(
            args['input'],
            loaded_model,
            args['top_k']
        )
    
    print(f"Likelihood: {likelyhood}")
    print(f"Classes: {classes}")
    print(f"Labled-Classes: {[category_names[cls] for cls in top_k_ind[0]]}")
    



if __name__ == "__main__":
    main()
