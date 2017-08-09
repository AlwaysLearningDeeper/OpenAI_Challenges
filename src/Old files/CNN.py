import gym,time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
#1 NO Op, 2-3 Either Right or Left

env = gym.make('Breakout-v0')
env.reset()
goal_steps=500

def stateExploration():
    env.reset()
    for _ in range(1000):
        env.render()
        time.sleep(0.1)
        observation, reward, done, info = env.step(3)
        if _ == 50:
            nobs = np.reshape(observation,(160,210,3))
            print(observation.shape)
            print(nobs.shape)

        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            break

stateExploration()




def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def cnn_model_fn(features,labels,mode):

    #Input layer
    input_layer=tf.reshape(features,[-1,28,28,1])

    #Convolutional Layer 1
    conv1=tf.layers.conv2d(inputs=input_layer,
                          filters=32,
                          kernel_size=5,
                          padding='same',
                          activation=tf.nn.relu)

    #Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=2,
                                    strides=2)
    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,
                            filters=64,
                            kernel_size=5,
                            padding='same',
                            activation=tf.nn.relu)
    #Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=2,
                                    strides=2)

    #Dense layer, 7 pool2 * height * 7 pool2 witdh * 64 pool2channels
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu) # 1024 relu neurons
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    #Logits Layer
    logits=tf.layers.dense(inputs=dropout, units=10)

    loss=None
    train_op=None

    #Calculate loss
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)#Transform values to onehot vectors
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

      # Generate Predictions
    predictions = {
          "classes": tf.argmax(input=logits, axis=1), #Predicted class
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")#% for each class
      }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)
