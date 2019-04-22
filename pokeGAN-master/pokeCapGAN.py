import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *

slim = tf.contrib.slim
HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000
os.environ['CUDA_VISIBLE_DEVICES'] = '15'
version = 'newPokemon'
newPoke_path = './' + version

def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
 
def process_data():   
    current_dir = os.getcwd()
    pokemon_dir = os.path.join(current_dir, 'resized_black')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
        
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer(
                                        [all_images])                               
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
                                    num_threads = 4, capacity = 200 + 3 * BATCH_SIZE,
                                    min_after_dequeue = 200 + 3 * BATCH_SIZE)
    num_images = len(images)

    return images_batch, num_images

def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # 4*4*512
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        
        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6

class CapsConv(object):
    def __init__(self, num_units, with_routing=True):
        self.num_units = num_units
        self.with_routing = with_routing

    def __call__(self, input, num_outputs, kernel_size=None, stride=None, reuse=False):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        if not self.with_routing:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            capsules = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                self.kernel_size, self.stride, padding="VALID",
                                                activation_fn=tf.nn.relu)
            capsules = tf.reshape(capsules, (BATCH_SIZE, -1, self.num_units, 1))
            capsules = squash(capsules)
            return(capsules)
            
        else:
            input = tf.reshape(input, shape=(BATCH_SIZE, 9*9*32, 8,1))
            b_IJ = tf.zeros(shape=[1,9*9*32,32,1], dtype=np.float32)
            capsules = []
            for j in range(self.num_outputs):
                with tf.variable_scope('caps_' + str(j)):
                    caps_j, b_IJ = capsule(input, b_IJ, j)
                    capsules.append(caps_j)
            
            capsules = tf.concat(capsules, axis=1)
        
        return(capsules)

def squash(vector):
    vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector)))  # a scalar
    scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
    vec_squashed = scalar_factor * tf.divide(vector, vec_abs)  # element-wise
    return(vec_squashed)

def discriminator(input, is_train, reuse=False):
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        # Input shape is 128 x 128 x 3
        # After First Convolutional Layer: 60 x 60 x 128
        conv1 = tf.layers.conv2d(input, 128, kernel_size = [9, 9], strides = [2, 2], padding = 'VALID', 
                                kernel_initializer = tf.truncated_normal_initializer(stddev=0.02), name = 'conv1')
        # Next Convolutional Layer: 26 x 26 x 256
        conv2 = tf.layers.conv2d(input, 256, kernel_size = [9, 9], strides = [2, 2], padding = 'VALID',
                                kernel_initializer = tf.truncated_normal_initializer(stddev=0.02), name = 'conv2')

    
    # First capsule: 8 units per capsule, 9x9 feature map. Number of capsules = 32
    with tf.variable_scope('Primary_Capsule'):
        if reuse:
            scope.reuse_variables()

        primaryCaps = CapsConv(num_units = 8, with_routing = False)
        caps1 = primaryCaps(conv2, num_outputs = 32 * 9, kernel_size = [9, 9], stride = 2, reuse = reuse)

    with tf.variable_scope('Secondary_Capsule'):
        if reuse:
            scope.reuse_variables()

        secondaryCaps = CapsConv(num_units = 16, with_routing = True)
        caps2 = secondaryCaps(caps1, num_outputs = 32, reuse = reuse)

    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        # Fully Connected Layers
        d_w3 = tf.get_variable('d_w3', [16*32, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(caps2, [-1, 16*32])
        d3 = tf.matmul(d3, d_w3) + d_b3
        d3 = tf.nn.relu(d3)
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4
        
    return d4

def train():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    random_dim = 100
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.        
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()
    batch_num = int(samples_num / batch_size)
    total_batch = 0
    saver = tf.train.Saver()
    # continue training
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    if(ckpt is not None):
        saver.restore(sess, ckpt)
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print( 'total training sample num:%d' % samples_num)
    print( 'batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print( 'start training...')
    for i in range(EPOCH):
        for j in range(batch_num):
            d_iters = 5
            g_iters = 1
            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                train_image = sess.run(image_batch)
                sess.run(d_clip)
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})
            
        # save check point every 50 epoch
        if i%50 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)

            saver.save(sess, './model/' +version + '/' + str(i))
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)

            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            save_images(imgtest, [8,8] ,newPoke_path + '/epoch' + str(i) + '.jpg')
            
            print( 'train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))

    coord.request_stop()
    coord.join(threads)

def capsule(input, b_IJ, idx_j):
    with tf.variable_scope('routing'):
        w_initializer = np.random.normal(size=[1, 9*9*32, 8, 16], scale=0.01)
        W_Ij = tf.Variable(w_initializer, dtype=tf.float32)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

        W_Ij = tf.tile(W_Ij, [BATCH_SIZE, 1, 1, 1])
        u_hat = tf.matmul(W_Ij, input, transpose_a=True)
        shape = b_IJ.get_shape().as_list()
        size_splits = [idx_j, 1, shape[2] - idx_j - 1]
        for r_iter in range(3):
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            b_Il, b_Ij, b_Ir = tf.split(b_IJ, size_splits, axis=2)
            c_Il, c_Ij, b_Ir = tf.split(c_IJ, size_splits, axis=2)
            s_j = tf.multiply(c_Ij, u_hat)
            s_j = tf.reduce_sum(tf.multiply(c_Ij, u_hat),
                                axis=1, keep_dims=True)
            v_j = squash(s_j)
            v_j_tiled = tf.tile(v_j, [1, 9*9*32, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)
            b_Ij += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
            b_IJ = tf.concat([b_Il, b_Ij, b_Ir], axis=2)

        return(v_j, b_IJ)

if __name__ == "__main__":
    train()