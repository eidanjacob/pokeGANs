import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 32
EPOCH = 5000
os.environ['CUDA_VISIBLE_DEVICES'] = '15'
version = 'capsPokemon'
newPoke_path = './' + version
sess = tf.Session()

def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
 
def process_data():   
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, 'resized_black')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer(
                                        [all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()
    print("line 50")
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return iamges_batch, num_images

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
        print("line 100")
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        
        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6

class CapsConv(object):
    def __init__(self, num_units, with_routing = True):
        self.num_units = num_units
        self.with_routing = with_routing
    
    def __call__(self, input, num_outputs, kernel_size= None, stride=None):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        batch_size = 32
        if not self.with_routing:
            capsules = tf.contrib.layers.conv2d(input, self.num_outputs, self.kernel_size, self.stride, padding = "VALID", activation_fn=tf.nn.relu)
            capsules = tf.reshape(capsules, (batch_size, -1, self.num_units, 1))
            capsules = squash(capsules)
            return(capsules)
        else:
            input = tf.reshape(input, shape = (batch_size, 120*120, 8, 1))
            b_IJ = tf.zeros(shape=[1, 120*120, 32, 1], dtype = np.float32)
            capsules = []
            for j in range(self.num_outputs):
                with tf.variable_scope('caps_'+str(j)):
                    caps_j, b_IJ = capsule(input, b_IJ, j)
                    capsules.append(caps_j)
            
            capsules = tf.concat(capsules, axis = 1)
        
        print("line 137")
        return(capsules)


def capsule(input, b_IJ, idx_j):
    with tf.variable_scope('routing'):
        w_initializer = np.random.normal(size = [1, 120*120, 8, 16], scale = 0.01)
        W_Ij = tf.Variable(w_initializer, dtype = tf.float32)
        sess.run(tf.global_variables_initializer())
        batch_size = 32
        W_Ij = tf.tile(W_Ij, [batch_size, 1, 1, 1])
        u_hat = tf.matmul(W_Ij, input, transpose_a = True)
        shape = tf.shape(b_IJ)
        print("line 150")
        size_splits = [idx_j, 1, shape[2] - idx_j - 1]
        for r_iter in range (3):
            c_IJ = tf.nn.softmax(b_IJ, dim = 2)
            b_Il, b_Ij, b_Ir = tf.split(b_IJ, size_splits, axis = 2)
            c_Il, c_Ij, c_Ir = tf.split(c_IJ, size_splits, axis = 2)
            s_j = tf.multiply(c_Ij, u_hat)
            s_j = tf.reduce_sum(tf.multiply(c_Ij, u_hat), axis = 1, keepdims = True)
            v_j = squash(s_j)
            v_j_tiled = tf.tile(v_j, [1, 120*120, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_j_tiled, transpose_a = True)
            b_Ij = b_Ij + tf.reduce_sum(u_produce_v, axis = 0, keepdims = True)
            b_Ij = tf.concat([b_Il, b_Ij, b_Ir], axis = 2)

        return(v_j, b_IJ)

def squash(vector):
    vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector)))
    scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
    vec_squashed = scalar_factor * tf.divide(vector, vec_abs)
    return(vec_squashed)

def discriminator(x_image, reuse=False):
    x_image.get_shape()
    if(reuse):
        tf.get_variable_scope().reuse_variables()

    # Discriminator hyperparameters
    n_conv_features = 256
    conv_feature_size = 9

    # Convolutional Layer
    d_w1 = tf.get_variable('d_w1', [conv_feature_size, conv_feature_size, 3, n_conv_features], initializer=tf.truncated_normal_initializer(stddev=0.05))
    d_b1 = tf.get_variable('d_b1', [n_conv_features], initializer = tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='VALID')
    d1 = d1 + d_b1

    # Capsule Layer
    with tf.variable_scope('cap1'):
        primaryCaps = CapsConv(num_units = 8, with_routing = False)
        caps1 = primaryCaps(d1, num_outputs=256, kernel_size = 9, stride = 2)

    with tf.variable_scope('cap2'):
        secondaryCaps = CapsConv(num_units = 16, with_routing = True)
        caps2 = secondaryCaps(caps1, num_outputs = 32, stride = 1)

    # Fully Connected Layers
    d_w3 = tf.get_variable('d_w3', [16*32, 1024], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer = tf.constant_initializer(0))
    d3 = tf.reshape(caps2, [-1, 16*64])
    print("line 200")
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer = tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer = tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4
    return d4

def train():
    random_dim = 100
    print( os.environ['CUDA_VISIBLE_DEVICES'])
    
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    real_result = discriminator(real_image, False)
    fake_result = discriminator(fake_image, False)
    
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.
    
    # # dcgan loss
    # fake_image = generator(random_input, random_dim, is_train)
    # # sample_fake = generator(random_input, random_dim, is_train, reuse = True)
    # real_logits, real_result = discriminator(real_image, is_train)
    # fake_logits, fake_result = discriminator(fake_image, is_train, reuse=True)
    
    # d_loss1 = tf.reduce_mean(
            # tf.nn.sigmoid_cross_entropy_with_logits(
            # logits = real_logits, labels = tf.ones_like(real_logits)))
    # d_loss2 = tf.reduce_mean(
            # tf.nn.sigmoid_cross_entropy_with_logits(
            # logits = fake_logits, labels = tf.zeros_like(fake_logits)))
    
    # d_loss = d_loss1 + d_loss2
    
    # g_loss = tf.reduce_mean(
            # tf.nn.sigmoid_cross_entropy_with_logits(
            # logits = fake_logits, labels = tf.ones_like(fake_logits)))
            

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    print("line 250")
    # test
    print(d_vars)
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    
    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()
    
    batch_num = int(samples_num / batch_size)
    total_batch = 0
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
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
                #wgan clip weights
                sess.run(d_clip)
                
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            # print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)
        print("line 300")    
        # save check point every 500 epoch
        if i%500 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))  
        if i%50 == 0:
            # save images
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            save_images(imgtest, [8,8] ,newPoke_path + '/epoch' + str(i) + '.jpg')
            
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    train()
