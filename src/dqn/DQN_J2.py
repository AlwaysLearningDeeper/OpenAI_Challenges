import tensorflow as tf
import numpy as np
from collections import deque
import random,sys

NUM_CHANNELS = 4  # image channels
IMAGE_SIZE = 84  # 84x84 pixel images
SEED = 17  # random initialization seed
NUM_ACTIONS = 4  # number of actions for this game
BATCH_SIZE = 100
INITIAL_EPSILON = 1.0
GAMMA = 0.99
RMS_LEARNING_RATE = 0.00025
RMS_DECAY = 0.99
RMS_MOMENTUM = 0.0
RMS_EPSILON = 1e-6
REPLAY_MEMORY = 15000


def weight_variable(shape, sdev=0.1):
    initial = tf.truncated_normal(shape, stddev=sdev, seed=SEED)
    return tf.Variable(initial)


def bias_variable(shape, constant=0.1):
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)


class QNet:
    def __init__(self, num_actions):
        # the weights and biases will be reassigned during training,
        # so they are instance-specific properties

        # weights
        self.conv1_w = weight_variable([8, 8, NUM_CHANNELS, 32])
        self.conv1_b = bias_variable([32])

        self.conv2_w = weight_variable([4, 4, 32, 64])
        self.conv2_b = bias_variable([64])

        self.conv3_w = weight_variable([3, 3, 64, 64])
        self.conv3_b = bias_variable([64])

        self.fc_w = weight_variable([7744, 512])
        self.fc_b = bias_variable([512])

        self.num_actions = num_actions
        self.out_w = weight_variable([512, num_actions])
        self.out_b = bias_variable([num_actions])

        self.stateInput = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

        # hidden layers
        h_conv1 = tf.nn.conv2d(self.stateInput, self.conv1_w, strides=[1, 4, 4, 1], padding='SAME')
        h_relu1 = tf.nn.relu(tf.nn.bias_add(h_conv1, self.conv1_b))

        h_conv2 = tf.nn.conv2d(h_relu1, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME')
        h_relu2 = tf.nn.relu(tf.nn.bias_add(h_conv2, self.conv2_b))

        h_conv3 = tf.nn.conv2d(h_relu2, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME')
        h_relu3 = tf.nn.relu(tf.nn.bias_add(h_conv3, self.conv3_b))

        # reshape for fully connected layer
        relu_shape = h_relu3.get_shape().as_list()
        print('Output relu shape %s' % relu_shape)
        reshape = tf.reshape(h_relu3,
                             [-1, relu_shape[1] * relu_shape[2] * relu_shape[3]])
        # fully connected and output layers
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc_w) + self.fc_b)

        # calculate the Q value as output
        self.QValue = tf.matmul(hidden, self.out_w) + self.out_b

    def properties(self):
        return (self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b,
                self.conv3_w, self.conv3_b, self.fc_w, self.fc_b,
                self.out_w, self.out_b)


class DQN:
    def __init__(self, actions):
        self.replayMemory = deque()
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        self.currentQNet = QNet(len(actions))
        self.targetQNet = QNet(len(actions))

        self.actionInput = tf.placeholder("float", [None, len(actions)],name="input")
        self.yInput = tf.placeholder("float", [None],name="y")
        self.Q_action = tf.reduce_sum(tf.multiply(self.currentQNet.QValue, self.actionInput), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.yInput - self.Q_action))
        self.trainStep = tf.train.RMSPropOptimizer(RMS_LEARNING_RATE, RMS_DECAY, RMS_MOMENTUM, RMS_EPSILON).minimize(
            self.loss)

    def copyCurrentToTargetOperation(self):
        targetProps = self.targetQNet.properties()
        currentProps = self.currentQNet.properties()
        props = zip(targetProps, currentProps)
        return [targetVar.assign(currVar) for targetVar, currVar in props]

    def selectAction(self, currentState):
        action = np.zeros(len(self.actions))
        if random.random() < self.epsilon:
            actionInd = random.randrange(0, len(self.actions))
        else:
            qOut = self.currentQNet.QValue.eval(feed_dict={self.stateInput: [currentState]})
            actionInd = np.argmax(qOut)
        action[actionInd] = 1.0
        return action

    def storeExperience(self, state, action, reward, newState, terminalState):
        if len(self.replayMemory) < REPLAY_MEMORY:
            self.replayMemory.append((state, action, reward, newState, terminalState))
        else:
            #print('Max replay memory reached')
            self.replayMemory.pop()
            self.replayMemory.append((state, action, reward, newState, terminalState))


    def sampleExperiences(self):
        if len(self.replayMemory) < BATCH_SIZE:
            return list(self.replayMemory)
        return random.sample(self.replayMemory, BATCH_SIZE)
