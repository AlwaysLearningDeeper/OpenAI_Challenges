import numpy as np
import tensorflow as tf
import random
import gym
from Project.src.dqn.Replay_Memory import Replay_Memory
import time
import cv2


MEMORY_LENGTH = 4
ACTIONS = 4
LEARNING_RATE = 0.00025
FINAL_EXPLORATION_FRAME = 100000
RMSPROP_MOMENTUM = 0.95
RMSPROP_EPSILON = 0.01
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 5000
RANDOM_STEPS_REPLAY_MEMORY_INIT = 5000
TRAINING_STEPS = 200000
ENVIRONMENT = 'Breakout-v0'
NO_OP_CODE = 1
TF_RANDOM_SEED = 17

env=gym.make(ENVIRONMENT)
memory = Replay_Memory(REPLAY_MEMORY_SIZE)


def stack(frames):
    return np.stack(frames,2)


def getEpsilon(step):
    if step > FINAL_EXPLORATION_FRAME:
        return 0.1
    else:
        return 1 - step*(0.9/FINAL_EXPLORATION_FRAME)

def randomSteps(steps=RANDOM_STEPS_REPLAY_MEMORY_INIT,initial_no_ops=4):
    t0 = time.time()
    env.reset()
    i = 0
    frame_stack = []

    for _ in range(0,steps):
        if i < initial_no_ops:
            action = NO_OP_CODE
            observation, reward, done, info = env.step(action)
            greyObservation = rgb2gray(observation)
            downObservation = downSample(greyObservation)
            frame_stack.append(downObservation)
            i+=1

        else:
            s_t = stack(frame_stack)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            greyObservation = rgb2gray(observation)
            downObservation = downSample(greyObservation)

            frame_stack.pop(0)
            frame_stack.append(downObservation)
            s_t_plus1 = stack(frame_stack)

            if done:
                memory.store_transition(
                    (
                    s_t.astype(type),
                    action,
                    reward,
                    None,
                )
                )

            else:
                memory.store_transition(
                    (
                        s_t.astype(type),
                        action,
                        reward,
                        s_t_plus1.astype(type),
                    )
                )

        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            env.reset()
            i=0
            frame_stack=[]



    t1 = time.time()
    print("This operation took:",t1-t0,)

    #Sample random transitions and show
    #for _ in range(5):
        #t = np.split(memory.sample_transition()[0],4,2)
        #plt.imshow(t[0].reshape(84, 84), cmap=matplotlib.cm.Greys_r)
        #plt.show()


input = tf.placeholder("float", [None, 84, 84, 4],name='input')
actions = tf.placeholder(tf.int32, [None], name="actions")
r = tf.placeholder(tf.float32, [None], name="r")
def createNetwort():

    # input layer


    # hidden layers
    conv1 = tf.contrib.layer.conv2d(inputs=input,
                                    num_outputs=32,
                                    kernel_size=[8,8],
                                    stride=[4,4],
                                    padding='same')
    # pool1 = tf.contrib.layers.max_pooling2d(inputs=conv1,
    #                                 pool_size=2,
    #                                 strides=4)

    conv2 = tf.contrib.layer.conv2d(inputs=conv1,
                                    num_outputs=64,
                                    kernel_size=[4,4],
                                    stride=[2,2],
                                    padding='same')
    # pool2 = tf.contrib.layers.max_pooling2d(inputs=conv2,
    #                                 pool_size=2,
    #                                 strides=2)
    conv2 = tf.contrib.layer.conv2d(inputs=conv1,
                                    num_outputs=64,
                                    kernel_size=[3,3],
                                    stride=[1,1],
                                    padding='same')

    #conv2_flat = tf.reshape(pool2, [-1, 256])
    relu_1 = tf.contrib.layers.relu(conv2, num_outputs=512)
    output = tf.contrib.layers.fully_connected(tf.reshape(relu_1,[-1,11*11*64]),num_outputs=ACTIONS)


    #Jairsan magic
    actions_one_hot = tf.one_hot(actions, ACTIONS, name="actions_one_hot")
    eliminate_other_Qs = tf.multiply(output, actions_one_hot)
    Q_of_selected_action = tf.reduce_sum(eliminate_other_Qs)

    loss = tf.square(tf.subtract(Q_of_selected_action, r))

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.RMSPropOptimizer(momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON).minimize(cost)

    return output, optimizer



def trainDQN(nn,sess):
        tf.set_random_seed(TF_RANDOM_SEED)
        sess.run(tf.global_variables_initializer())

        output, optimizer = createNetwort()

        i = 0
        frame_stack = []
        initial_no_op = np.random.randint(4, 50)

        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print
            "Successfully loaded:", checkpoint.model_checkpoint_path

        for step in range(TRAINING_STEPS):

            if i < initial_no_op:
                # WE PERFORM A RANDOM NUMBER OF NO_OP ACTIONS
                action = NO_OP_CODE
                observation, reward, done, info = env.step(action)
                greyObservation = rgb2gray(observation)
                downObservation = downSample(greyObservation)
                frame_stack.append(downObservation)
                i += 1

            else:
                # CHOOSING ACTION
                s_t = stack(frame_stack)
                if np.random.rand() < getEpsilon(step):
                    # We make random action with probability epsilon
                    action = env.action_space.sample()
                else:
                    # Pick action in a greedy way
                    action = np.argmax(sess.run([output], {input_tensor: s_t}))

                # STORE TRANSITION

                observation, reward, done, info = env.step(action)

                # Process received frame
                greyObservation = rgb2gray(observation)
                downObservation = downSample(greyObservation)

                # Remove oldest frame
                frame_stack.pop(0)
                frame_stack.append(downObservation)

                # Obtain state at t+1
                s_t_plus1 = stack(frame_stack)

                if done:
                    memory.store_transition(
                        (
                            s_t.astype(type),
                            action,
                            reward,
                            None,
                        )
                    )

                else:
                    memory.store_transition(
                        (
                            s_t.astype(type),
                            action,
                            reward,
                            s_t_plus1.astype(type),
                        )
                    )

                # OBTAIN MINIBATCH
                t = memory.sample_transition()
                frames_t = t[0]
                actions = t[1]
                rewards = t[2]
                frames_t_plus1 = t[3]
                for i in range(1, MINIBATCH_SIZE):
                    s = memory.sample_transition()
                    frames = np.concatenate((frames, s[0]))
                    action = np.concatenate((frames, s[1]))


def play():
    sess = tf.InteractiveSession()
    nn=createNetwort()
    trainDQN(nn, sess)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)