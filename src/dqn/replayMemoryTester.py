from DQN_J2 import *
from utils.Stack import *
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2,re,random,time

STEPS= 100000000000
ENVIRONMENT = 'Breakout-v0'
SAVE_NETWORK = True
LOAD_NETWORK = True
BACKUP_RATE = 500
UPDATE_TIME = 100
NUM_CHANNELS = 4  # image channels
IMAGE_SIZE = 84  # 84x84 pixel images
SEED = 17  # random initialization seed
ACTIONS = [0,1,2,3]  # number of actions for this game
#BATCH_SIZE = 32
INITIAL_EPSILON = 1.0
GAMMA = 0.99
RMS_LEARNING_RATE = 0.00025
RMS_DECAY = 0.95
RMS_MOMENTUM = 0.95

REPLAY_MEMORY_SIZE = 15000
#RMS_EPSILON = 1e-6
RMS_EPSILON = 0.01
REPLAY_MEMORY = 15000
FINAL_EXPLORATION_FRAME = 1000000
NO_OP_MAX = 30
NO_OP_CODE = 1

env = gym.make(ENVIRONMENT)
dqn = DQN(ACTIONS)

def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def randomSteps(env,steps,dqn):
    t0 = time.time()
    env.reset()
    i = 0
    frame_stack = Stack(4)
    initial_no_op = np.random.randint(4, NO_OP_MAX)

    for _ in range(0,steps):
        if i < initial_no_op:
            # WE PERFORM A RANDOM NUMBER OF NO_OP ACTIONS
            action = NO_OP_CODE
            state, reward, done, info = env.step(action)
            greyObservation = rgb2gray(state)
            state = downSample(greyObservation)
            frame_stack.push(state)
            i += 1
        else:

            state = np.stack(frame_stack.items, axis=2).reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

            action = np.random.randint(0, len(ACTIONS))
            actionH =np.zeros(len(ACTIONS))
            actionH[action] = 1
            next_state, reward, game_over, info = env.step(action)


            greyObservation = rgb2gray(next_state)
            next_state = downSample(greyObservation)

            frame_stack.push(next_state)

            next_state = np.stack(frame_stack.items, axis=2).reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

            dqn.storeExperience(state, actionH, reward, next_state, game_over)
            if done:
                #print("Episode finished after {} timesteps".format(_ + 1))
                env.reset()
                i=0
                frame_stack=[]



    t1 = time.time()
    print("Fullfilling replay memory operation took:",t1-t0,)


    randomSteps()

