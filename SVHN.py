#Created by: Ryan Tran and Thomas Bryant

import numpy as np
import scipy as sp
import scipy.io as spo
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_dir = "data_batch_1.mat"
test_dir = "test_32x32.mat"

########################################################################################
#--------------------------------Load_Data_Functions-----------------------------------#
########################################################################################

def load_training():
    
    train_data = spo.loadmat(train_dir)
    Xtemp = np.asarray(train_data['data'])
    num_image = Xtemp.shape[0]
    y_train = np.asarray(train_data['labels'])
    X_train = []
    for i in range(10000):
        chunk_size = int(Xtemp.shape[1]/3)
        r = Xtemp[i][0:chunk_size]
        g = Xtemp[i][chunk_size:2*chunk_size]
        b = Xtemp[i][2*chunk_size:3*chunk_size]
        rgb = np.vstack((r,g,b)).reshape((-1),order="F")
        Xtemp[i] = np.asarray(rgb)
    X_train = np.reshape(Xtemp, (num_image,
                                 img_size,
                                 img_size,
                                 num_channels))
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    return X_train,y_train

def load_test():
    
    test_data = spo.loadmat(train_dir)
    Xtemp = np.asarray(test_data['data'])
    num_image = Xtemp.shape[0]
    y_test = np.asarray(test_data['labels'])                    
    X_test = []
    for i in range(10000):
        chunk_size = int(Xtemp.shape[1]/3)
        r = Xtemp[i][0:chunk_size]
        g = Xtemp[i][chunk_size:2*chunk_size]
        b = Xtemp[i][2*chunk_size:3*chunk_size]
        rgb = np.vstack((r,g,b)).reshape((-1),order="F")
        Xtemp[i] = np.asarray(rgb)
    X_test = np.reshape(Xtemp, (num_image,
                                img_size,
                                img_size,
                                num_channels))
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return X_test,y_test

########################################################################################
#--------------------------------Hyper Parameters--------------------------------------#
########################################################################################

img_size = 32    #Image is a 32x32
num_channels = 3 #Image has 3 channels: red, green, blue.
num_classes = 10 #10 different kinds of single digits

filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 32

filter_size3 = 5
num_filters3 = 48

filter_size4 = 5
num_filters4 = 64

batch_size = 100
steps = 1000
learn_rate = 0.001

X = tf.placeholder(tf.float32, shape=[None,img_size,img_size,num_channels])
y = tf.placeholder(tf.float32, shape=[None,num_classes])

#Shape of each convoltional layer
conv_shape1 = [filter_size1, filter_size1, num_channels, num_filters1]
conv_shape2 = [filter_size2, filter_size2, num_filters1, num_filters2]
conv_shape3 = [filter_size3, filter_size3, num_filters2, num_filters3]
conv_shape4 = [filter_size4, filter_size4, num_filters3, num_filters4]

#Initialize weights and biases for convoltional layers
conv_weight1 = tf.Variable(tf.truncated_normal(shape=conv_shape1, stddev=0.05))
conv_bias1   = tf.Variable(tf.constant(value=0.05, shape=[num_filters1]))

conv_weight2 = tf.Variable(tf.truncated_normal(shape=conv_shape2, stddev=0.05))
conv_bias2   = tf.Variable(tf.constant(value=0.05, shape=[num_filters2]))

conv_weight3 = tf.Variable(tf.truncated_normal(shape=conv_shape3, stddev=0.05))
conv_bias3   = tf.Variable(tf.constant(value=0.05, shape=[num_filters3]))

conv_weight4 = tf.Variable(tf.truncated_normal(shape=conv_shape4, stddev=0.05))
conv_bias4   = tf.Variable(tf.constant(value=0.05, shape=[num_filters4]))

#Initialize weights and biases for fully connected layers
fc_weight1 = tf.Variable(tf.truncated_normal(shape=[256, 128], stddev=0.05))
fc_bias1   = tf.Variable(tf.constant(value=0.05, shape=[128]))

fc_weight2 = tf.Variable(tf.truncated_normal(shape=[128,num_classes], stddev=0.05))
fc_bias2   = tf.Variable(tf.constant(value=0.05, shape=[num_classes]))

########################################################################################
#--------------------------------------------------------------------------------------#
########################################################################################

def cnn(data):
    #Create convolutional layer 1
    conv1 = tf.nn.conv2d(input=data,
                         filter=conv_weight1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 += conv_bias1
    conv1 = tf.nn.max_pool(value=conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    conv1 = tf.nn.relu(conv1)

    #Create convolutional layer 2
    conv2 = tf.nn.conv2d(input=conv1,
                         filter=conv_weight2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 += conv_bias2
    conv2 = tf.nn.max_pool(value=conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    conv2 = tf.nn.relu(conv2)

    #Create convolutional layer 3
    conv3 = tf.nn.conv2d(input=conv2,
                         filter=conv_weight3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 += conv_bias3
    conv3 = tf.nn.max_pool(value=conv3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    conv3 = tf.nn.relu(conv3)

    #Create convolutional layer 4
    conv4 = tf.nn.conv2d(input=conv3,
                         filter=conv_weight4,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv4 += conv_bias4
    conv4 = tf.nn.max_pool(value=conv4,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    conv4 = tf.nn.relu(conv4)

    #Reduce 4dim to 2dim
    prev_lay_shape = conv4.get_shape()
    prev_num_feat = prev_lay_shape[1:4].num_elements()
    print(prev_num_feat)
    flatten_layer = tf.reshape(conv4,[-1,prev_num_feat])

    #Dropout layer
    drop_layer = tf.nn.dropout(flatten_layer,0.90)

    print("first")
    #First fully connected layer
    fc1 = tf.matmul(drop_layer, fc_weight1) + fc_bias1
    fc1 = tf.nn.relu(fc1)

    print("second")
    #Second fully conected layer
    fc2 = tf.matmul(fc1, fc_weight2) + fc_bias2
    fc2 = tf.nn.relu(fc2)
    return fc2

#------------------------------------------#
X_train,y_train = load_training()
X_test,y_test = load_test()
num_instances = X_train.shape[0]
model = cnn(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,
                                                              labels=y))
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)
y_pred = tf.nn.softmax(model)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tf.__version__)
    for step in range(steps):
        #ensure that our batch circles around our data set
        start = step*batch_size % num_instances
        training_batch = X_train[start:(start+batch_size),:,:,:]
        label_batch = y_train[start:(start+batch_size)]
        sess.run(cnn(training_batch))
##        feed_dict = {X: training_batch,
##                     y: label_batch}
##        prediction = sess.run([optimizer, loss], feed_dict=feed_dict)
        if step%100 == 0:
            print(step)
    print("done")
