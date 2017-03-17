#DISCLAIMER: 
#The code is built upon Magnus Erik Hvass Pedersen's template code on convultional neural networks
#Template code modified to work with SVHN datasets
#Created by: Ryan Tran and Thomas Bryant

import numpy as np
import scipy as sp
import scipy.io as spo
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_dir = "data_batch_1.mat"
test_dir = "test_32x32.mat"

#Training and test data are 4D. First dim encases the images.
#Second dim encases the rgb containers. $Third dim encases the rgb values
#Fourth dim is the rgb value.

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

################################################
#############---HELPER FUNCTIONS---#############
################################################
def create_weights(shape):
    #create a weight with random values
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#conv layers are always 4 dims
#EX: [num_images, img_height, img_width, num_channels]
def create_conv_layer(input,                #The previous layer
                      num_input_channels,   #Num. of filters in prev layer
                      filter_size,          #W and H of each filter
                      num_filters,          #Num. of filters
                      max_pooling=True):    #Use max-pooling?
    
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = create_weights(shape=shape)
    biases = create_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    if max_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

#Conv layer is 4 dims. Reduce to 2 so fully connected
#layer can take layer as input.
#Reduce layer to [num_images,num_features]
def two_dim_reduction(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    #[-1,..] tells the prog to find what the first dim is 
    new_layer = tf.reshape(layer,[-1,num_features])
    return new_layer, num_features

def create_fc_layer(input,          #The previous layer
                    num_inputs,     #Num of inputs in prev layer
                    num_outputs,    #Num of outputs in prev layer
                    relu=True):     #Use relu?
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if relu:
        layer = tf.nn.relu(layer)
    return layer

#Create the network
def cnn(data):
    conv1, weights_conv1 = create_conv_layer(data,
                                             num_channels,
                                             filter_size1,
                                             num_filters1)
    conv2, weights_conv2 = create_conv_layer(conv1,
                                             num_filters1,
                                             filter_size2,
                                             num_filters2)
    conv3, weights_conv3 = create_conv_layer(conv2,
                                             num_filters2,
                                             filter_size3,
                                             num_filters3)
    conv4, weights_conv4 = create_conv_layer(conv3,
                                             num_filters3,
                                             filter_size4,
                                             num_filters4)
    redu_layer, num_feat = two_dim_reduction(conv4)
    drop_layer = tf.nn.dropout(redu_layer,0.90)
    fc1 = create_fc_layer(drop_layer,
                          num_feat,
                          128)
    fc2 = create_fc_layer(fc1,
                          128,
                          num_classes)
    return fc2

#------------LOAD DATA FUNCTIONS------------#
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
    for step in range(steps):
        #ensure that our batch circles around our data set
        start = step*batch_size % num_instances
        training_batch = X_train[start:(start+batch_size),:,:,:]
        label_batch = y_train[start:(start+batch_size)]
        feed_dict = {X: training_batch,
                     y: label_batch}
        prediction = sess.run([optimizer, loss], feed_dict=feed_dict)
        if step%100 == 0:
            print(step)
    print("done")
