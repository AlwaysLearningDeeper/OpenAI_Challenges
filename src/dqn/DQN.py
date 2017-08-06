import gym,time,copy
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import matplotlib
from Replay_Memory import Replay_Memory





MEMORY_LENGTH = 4
ACTIONS = 4
LEARNING_RATE=0.00025
FINAL_EXPLORATION_FRAME = 100000
RMSPROP_MOMENTUM = 0.95
RMSPROP_EPSILON = 0.01
REPLAY_MEMORY_SIZE = 5000
RANDOM_STEPS_REPLAY_MEMORY_INIT = 5000
TRAINING_STEPS = 200000
ENVIRONMENT = 'Breakout-v0'
type = np.dtype(np.float32)

env = gym.make(ENVIRONMENT)
env.reset()
memory = Replay_Memory(REPLAY_MEMORY_SIZE)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)

def stack(frames):
    return np.stack(frames,2)


def randomSteps(steps=RANDOM_STEPS_REPLAY_MEMORY_INIT,initial_no_ops=4):
    t0 = time.time()
    env.reset()
    i = 0
    frame_stack = []

    for _ in range(0,steps):
        if i < initial_no_ops:
            action = 1
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


def getEpsilon(step):
    if step > FINAL_EXPLORATION_FRAME:
        return 0.1
    else:
        return 1 - step*(0.9/FINAL_EXPLORATION_FRAME)

def model():
    input = tf.placeholder(tf.float32,(None,84,84,4),name="input")
    actions = tf.placeholder(tf.int32, [None], name="actions")
    r = tf.placeholder(tf.float32, [None], name="r")

    conv_1 = tf.contrib.layers.conv2d(input,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='SAME')
    conv_2 = tf.contrib.layers.conv2d(conv_1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='SAME')
    conv_3 = tf.contrib.layers.conv2d(conv_2, num_outputs=64, kernel_size=[3,3],stride=[1,1],padding='SAME')
    relu_1 = tf.contrib.layers.relu(conv_3, num_outputs=512)
    output = tf.contrib.layers.fully_connected(tf.reshape(relu_1,[-1,11*11*64]),num_outputs=ACTIONS)

    #MB ERR HERE
    actions_one_hot = tf.one_hot(actions, ACTIONS, name="actions_one_hot")
    eliminate_other_Qs = tf.multiply(output,actions_one_hot)
    Q_of_selected_action = tf.reduce_sum(eliminate_other_Qs)


    loss = tf.square(tf.subtract(Q_of_selected_action,r))

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.RMSPropOptimizer(momentum=RMSPROP_MOMENTUM,epsilon=RMSPROP_EPSILON).minimize(cost)

    return output,loss

def train():


#randomSteps()
#output = model()
output,x1 = model()
print(x1.shape)


a = tf.constant(value=[
    [1., 2., 3. ,4.],
    [1., 2., 3. ,4.],
    [1., 2., 3. ,4.],
])

b = tf.constant(value=[
    [ 1., 0., 0., 0.],
    [ 0., 1., 0., 0.],
    [ 0., 0., 1., 0.],
])

r = tf.multiply(a,b)

t = tf.reduce_sum(r,1)

ans = tf.reshape(t,[-1,1])

with tf.Session() as sess:
    print(sess.run(ans))

