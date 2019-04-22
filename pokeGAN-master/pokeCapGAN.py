import numpy as np 
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K 
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

import sys 
import os
import glob
from tqdm import tqdm

BATCH_SIZE = 64
HEIGHT, WIDTH, CHANNEL = 128, 128, 3

def load_data():   
    current_dir = os.getcwd()
    pokemon_dir = os.path.join(current_dir, 'resized_black')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer([all_images])                           
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    images_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return images_batch, num_images

def squash(vectors, axis = -1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def discriminator():
    img = Input(shape = )