import gym
import random

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import simple_rnn, gru
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('MountainCar-v0')
env.reset()
goal_steps=200
score_requirement= -180

initial_games = 20000
#noise_scaling=0.8
discreteActions=3

def random_values():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()

def intToOneHot(number, possibilities):
    return np.eye(possibilities)[number]
"""
def initial_population():
    training_data = []
    scores=[]
    acceptedScores=[]
    observation=[]
    for _ in range(initial_games):
        score = 0
        game_memory=[]
        prev_observation=[]
        #action = np.eye(discreteActions)[np.random.choice(discreteActions, goal_steps)] #Select random one-hot action
        for frame in range(goal_steps):
            action=random.randrange(0, 3)
            observation, reward, done, info = env.step(action)
            if len(prev_observation)>0:
                game_memory.append([prev_observation,action])
            prev_observation = observation
            score+=reward
            if done: break

        if score >= score_requirement:
            acceptedScores.append(score)
            for data in game_memory:
                output=intToOneHot(data[1],discreteActions)
                training_data.append([data[0],output])
        env.reset()
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(acceptedScores))
    print('Median score for accepted scores:', median(acceptedScores))
    print(Counter(acceptedScores))

    return training_data
"""
def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            env.render()

            # choose random action (0 or 1)
            action = random.randrange(0, 3)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                output=intToOneHot(data[1])
                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    print(accepted_scores)
initial_population()
