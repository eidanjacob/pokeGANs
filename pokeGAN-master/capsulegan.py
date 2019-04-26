import os
import tensorflow as tf 
import numpy as np 
import cv2
import random
from utils import *

HEIGHT, WIDTH, CHANNEL, BATCH_SIZE, EPOCH = 128, 128, 3, 32, 5000
version = 'capsulePokemon'
newPoke_path = "./" + version
config = tf.ConfigProto(device_count = {'GPU':0, 'CPU':8})
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 8
sess = tf.Session(config = config)

def lrelu(x, leak = 0.2):
    return tf.maximum(x, leak*x)

def squash(vector):
    vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector)))
    scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
    vec_squashed = scalar_factor * tf.divide(vector, vec_abs)
    return(vec_squashed)

def process_data():
    print('1')
    current_dir = os.getcwd()
    pokemon_dir = os.path.join(current_dir, 'resized_black')
    images = []
    print('2')
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir, each))

    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer([all_images])
    print('3')
    content = tf.read_file(images_queue[0])
    sess = tf.Session()
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    image = tf.image.random_flip_left_right(image) 
    image = tf.image.random_brightness(image, max_delta = 0.1)
    print('4')
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    print('5')
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    images_dataset = tf.data.Dataset.from_tensor_slices(image)
    images_dataset = images_dataset.shuffle(2000, reshuffle_each_iteration = True)
    images_dataset = images_dataset.batch(BATCH_SIZE, drop_remainder = True)
    print('6')
    images_iter = images_dataset.make_initializable_iterator()
    print('7')
    sess.run(images_iter.initializer, feed_dict = None)
    print('8')
    next_batch = images_iter.get_next()
    print('9')
    return next_batch, len(image)

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

def discriminator(input, is_train, reuse=False):
    # A capsule CNN for discriminating 
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        
        # Convolution: knumber ksize x ksize features
        c1knumber = 256
        c1ksize = 9
        conv1 = tf.layers.conv2d(input, c1knumber, kernel_size = [c1ksize, c1ksize], strides = [2,2], reuse = reuse, padding = 'VALID', activation = lrelu, name = 'conv1')

        # Primary Capsule Layer
        caps1number = 32
        caps1dim = 8
        caps1size = 9
        # print(conv1.shape)
        caps1 = tf.layers.conv2d(conv1, caps1number*caps1dim, kernel_size = [caps1size, caps1size], strides = [2,2], reuse = reuse, padding = 'VALID', activation = lrelu, name = 'caps1')
        caps1 = tf.reshape(caps1, shape = [BATCH_SIZE, 26*26*caps1number, 8, 1])
        caps1 = squash(caps1)

        # Dynamic Routing into Secondary Capsule Layer
        caps2number = 32
        b_IJ = tf.zeros(shape = [1, 26*26*caps1number, caps2number, 1], dtype = tf.float32)
        capsules = []
        for j in range(caps2number):
            with tf.variable_scope('capsule' + str(j)):
                caps_j, b_IJ = capsule(caps1, b_IJ, j)
                capsules.append(caps_j)
        
        caps2 = tf.concat(capsules, axis = 1)
        # BATCH_SIZE x 32 x 16 x 1

        # Fully Connected Layers Follow
        caps2 = tf.reshape(caps2, [-1, 16*32])
        conv2 = tf.layers.dense(inputs = caps2, units = 200, activation = lrelu, kernel_initializer = tf.initializers.truncated_normal, reuse=reuse, name = 'conv2')
        conv3 = tf.layers.dense(inputs = conv2, units = 100, activation = lrelu, kernel_initializer = tf.initializers.truncated_normal, reuse=reuse, name = 'conv3')
        conv4 = tf.layers.dense(inputs = conv3, units = 10, activation = lrelu, kernel_initializer = tf.initializers.truncated_normal, reuse=reuse, name = 'conv4')
        logits = tf.layers.dense(inputs = conv4, units = 1, activation = lrelu, kernel_initializer = tf.initializers.truncated_normal, reuse=reuse, name = 'logit')
        return logits

def capsule(input, b_IJ, idx_j):
    with tf.variable_scope('routing'):
        w_initializer = np.random.normal(size = (1, 26*26*BATCH_SIZE, 8, 16), scale = 0.01)
        w_Ij = tf.Variable(w_initializer, dtype = tf.float32)
        sess.run(w_Ij.initializer)
        # print(w_Ij.shape, input.shape)
        w_Ij = tf.tile(w_Ij, [BATCH_SIZE, 1, 1, 1])
        u_hat = tf.matmul(w_Ij, input, transpose_a = True)
        shape = b_IJ.get_shape().as_list()
        size_splits = [idx_j, 1, shape[2] - idx_j - 1]
        for routing_iteration in range(3):
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            b_Il, b_Ij, b_Ir = tf.split(b_IJ, size_splits, axis=2)
            c_Il, c_Ij, b_Ir = tf.split(c_IJ, size_splits, axis=2)
            s_j = tf.multiply(c_Ij, u_hat)
            s_j = tf.reduce_sum(tf.multiply(c_Ij, u_hat), axis=1, keepdims=True)
            v_j = squash(s_j)
            v_j_tiled = tf.tile(v_j, [1, 26*26*BATCH_SIZE, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)
            b_Ij += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
            b_IJ = tf.concat([b_Il, b_Ij, b_Ir], axis=2)

        return(v_j, b_IJ)

def train():
    random_dim = 100
    print('a')
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        print('b')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        print('c')
        is_train = tf.placeholder(tf.bool, name='is_train')
    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    print('d')
    real_result = discriminator(real_image, is_train)
    print('e')
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        pass

    fake_result = discriminator(fake_image, is_train, reuse=True)
    print('f')
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_result, labels=tf.fill([BATCH_SIZE, 1], np.float32(1))))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_result, labels=tf.fill([BATCH_SIZE, 1], np.float32(0))))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_result, labels = tf.fill([BATCH_SIZE, 1], np.float32(1))))      
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    print('g')
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    print('h')
    
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
    batch_size = BATCH_SIZE
    print('i')
    sess.run(tf.global_variables_initializer())
    print('j')
    saver = tf.train.Saver()
    print('k')
    image_batch, samples_num = process_data()
    print('l')
    batch_num = int(samples_num / batch_size)
    # continue training
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    if(ckpt is not None):
        print('ll')
        saver.restore(sess, ckpt)

    coord = tf.train.Coordinator()
    print('m')
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print( 'total training sample num:%d' % samples_num)
    print( 'batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print( 'start training...')
    for i in range(EPOCH):
        for j in range(batch_num):
            d_iters = 5
            g_iters = 1
            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            print("dis start")
            for k in range(d_iters):
                train_image = sess.run(image_batch)
                print("train_image done")
                sess.run(d_clip)
                print("sess.run done")
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
                print("dis loop done")

            print("dis end")
            # Update the generator
            print("gen start")
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})
                print("gen loop done")

            print("gen end")
        # save check point every 50 epoch
        if i%50 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            save_images(imgtest, [8,8], newPoke_path + '/epoch' + str(i) + '.jpg')
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    train()