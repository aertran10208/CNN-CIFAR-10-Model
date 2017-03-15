#DISCLAIMER: TEMPLATE CODE IS FROM MAGNUS ERIK HAVASS PEDERSEN

import numpy as np
import scipy as sp
import scipy.io as spo
import tensorflow as tf

train_dir = "train_32x32.mat"
test_dir = "test_32x32.mat"

#Training and test data are 4D. First dim holds encases the images.
#Second dim encases the rgb containers. $Third dim encases the rgb values
#Fourth dim is the rgb value.

img_size = 32    #Image is a 32x32
num_channels = 3 #Image has 3 channels: red, green, blue.
num_classes = 10 #10 digits

################################################
#############---HELPER FUNCTIONS---#############
################################################

def create_weight(shape):
    #create a weight with random values
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def create_conv_layer(input,                #The previous layer
                      num_input_channels,   #Num. of channesl in prev layer
                      filter_size,          #W and H of each filter
                      num_filters,          #Num. of filters
                      use_pooling=True):    #2x2 max-pooling
    
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
    
#------------LOAD DATA FUNCTIONS------------#
def load_training():
    
    train_data = spo.loadmat(train_dir)
    Xtemp = np.array(train_data['X'])
    y_train = np.array(train_data['y'])
    X_train = []
    
    for i in range(Xtemp.shape[3]):
        X_train.append(Xtemp[:,:,:,i])
    X_train = np.array(X_train)
    
    return X_train,y_train

def load_test():
    
    test_data = spo.loadmat(test_dir)
    Xtemp = np.array(test_data['X'])
    y_test = np.array(test_data['y'])
    X_test = []
    
    for i in range(Xtemp.shape[3]):
        X_test.append(Xtemp[:,:,:,i])
    X_test = np.array(X_test)
    
    return X_test,y_test
