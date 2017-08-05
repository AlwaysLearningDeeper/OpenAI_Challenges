import gym,time
import numpy as np
from matplotlib import pyplot as plt

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

