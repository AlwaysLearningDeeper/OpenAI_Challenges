import gym,time
import numpy as np
from matplotlib import pyplot as plt


env = gym.make('Breakout-v0')
env.reset()
goal_steps=500

def stateExploration():
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if _ == 50:
            nobs = np.reshape(observation,(160,210,3))
            print(observation.shape)
            print(nobs.shape)
            plt.imshow(nobs, interpolation='nearest')
            plt.show()
        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            break

stateExploration()

