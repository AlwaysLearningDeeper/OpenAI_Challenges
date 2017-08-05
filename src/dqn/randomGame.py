import gym,time
import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib

env = gym.make('Breakout-v0')
env.reset()
goal_steps=500

# Observation is of shape 210,160,3

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)


def random_game():
    env.reset()
    for _ in range(1000):
        env.render()
        # time.sleep(0.1)
        env.step(env.action_space.sample())


def stateExploration():
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        greyobs = rgb2gray(observation)
        downobs = downSample(greyobs)

        plt.imshow(downobs, cmap = matplotlib.cm.Greys_r)
        plt.show()
        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            break


stateExploration()

