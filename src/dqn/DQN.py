import gym,time,copy
import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib
from Replay_Memory import Replay_Memory

env = gym.make('Breakout-v0')
env.reset()


MEMORY_LENGTH = 4
EPISODES = 10000

# Observation is of shape 210,160,3

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)

def stack(frames):
    return np.stack(frames,2)


def random_game():
    env.reset()
    for _ in range(1000):
        env.render()
        # time.sleep(0.1)
        env.step(env.action_space.sample())


def randomSteps(steps=1500,length=4,initial_no_ops=4):
    t0 = time.time()
    env.reset()
    memory = Replay_Memory()
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
            s_t1 = stack(frame_stack)


            memory.store_transition(
                (
                copy.copy(s_t),
                action,
                reward,
                copy.copy(s_t1)
            )
            )





        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            env.reset()
            i=0



    t1 = time.time()
    print("This operation took:",t1-t0,)
randomSteps()

