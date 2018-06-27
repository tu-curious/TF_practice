# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 01:01:19 2018

@author: agarwal.270
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data


# In[Helpe f]

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    return

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.05))

def new_bias(shape):
    return tf.Variable(tf.constant(0.05,shape=[shape]))

def conv_layer(input_layer,n_fil,fil_height,fil_len,use_pool=True):
    # filter shape accepted by tf
    _,_,_,in_channels=input_layer.get_shape().as_list()
    out_channels=n_fil
    shape=[fil_height,fil_len,in_channels,out_channels]
    weights=new_weights(shape)
    bias=new_bias(out_channels)
    layer=tf.nn.conv2d(input=input_layer,filter=weights,strides=[1,1,1,1],padding='SAME')
    layer=layer+bias
    
    if use_pool:
        layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    layer=tf.nn.relu(layer)
    return layer,weights

def flat_layer(input_layer):
    shape=input_layer.get_shape()
    num_features=shape[1:4].num_elements()
    layer_flat=tf.reshape(input_layer,[-1,num_features])
    return layer_flat,num_features


def fc_layer(input_layer,num_out,use_relu=True):
    _,num_in=input_layer.get_shape().as_list()
    shape=[num_in,num_out]
    weights=new_weights(shape)
    bias=new_bias(num_out)
    layer=tf.add(tf.matmul(input_layer,weights),bias)
    if use_relu:
        layer=tf.nn.relu(layer)
    return layer,weights


def train_net(num_iterations,session):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    global total_iter;total_iter=0
    for i in range(total_iter,
                   total_iter + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    #total_iter=total_iter+num_iterations
    total_iter=i+1
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    return

def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    return

def plot_confusion_matrix(cls_pred):

    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    return


# Split the test-set into smaller batches of this size.# Split  
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_classes, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
    return
# In[Intitlzations & Hyper-Params]

#Laod data
data=input_data.read_data_sets('./data/MNIST/',one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

# Image config
img_size = 28
flat_img_len = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.


train_batch_size = 64

# In[Build]

tf.reset_default_graph() #reset tf graph

#placeholders for data
x=tf.placeholder(tf.float32,shape=[None,flat_img_len],name='x')
x_img=tf.reshape(x,shape=[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_classes=tf.argmax(y_true,axis=1)

# layers
L1_C,W1=conv_layer(x_img,num_filters1,filter_size1,filter_size1)
L2_C,W2=conv_layer(L1_C,num_filters2,filter_size2,filter_size2)
L3_F,num_features=flat_layer(L2_C)
L4_FC,W4=fc_layer(L3_F,fc_size)
L5_FC,W5=fc_layer(L4_FC,num_classes,use_relu=False)



#Just for our sake coz loss auto calculates softmax anyways
y_pred=tf.nn.softmax(L5_FC)
y_pred_classes=tf.argmax(y_pred,axis=1)

#Loss
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=L5_FC,labels=y_true)
loss=tf.reduce_mean(cross_entropy)

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

#Accuracy
correct_prediction = tf.equal(y_pred_classes, y_true_classes)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[Run]

session=tf.Session()
session.run(tf.global_variables_initializer())
writer=tf.summary.FileWriter('./graphs',session.graph)

#initial accuracy
print_test_accuracy()

#some training accuracy
train_net(num_iterations=100,session=session)
print_test_accuracy(show_example_errors=True)

#more training accuracy
train_net(num_iterations=900,session=session)
print_test_accuracy(show_example_errors=True)

session.close()
