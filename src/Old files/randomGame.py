import gym,time
import numpy as np
import random
from statistics import median, mean
from collections import Counter
import sys
import pickle
from matplotlib import pyplot as plt

#1 NO Op, 2-3 Either Right or Left

env = gym.make('Breakout-v0')
env.reset()
goal_steps=50000000000000000000000000000000
score_requirement = 50
initial_games = 100000000000000000000000000000000000000000000000000000000000000


def save_object(object, file_name):
    with open(file_name, 'wb') as fh:
        pickle.dump(object, fh)



class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

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

def initial_population():
    """
    Extracts good runs from random games. Code from sentdex
    :return training_data:
    """
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    cc=0
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for game in range(initial_games):
        env.reset()
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        #Episodes of 10 frames
        stack=Stack()
        # for each frame in 200
        for step in range(goal_steps):
            # choose random action (0 or 1)
            #env.render()
            action = random.randrange(1, 4)
            # do it!
            observation, reward, done, info = env.step(action)
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward

            stack.push([prev_observation,action])
            if(stack.size()>10):
                stack.pop()

            #print('Game: ' + str(game)+' Frame: ' + str(step) + '  Reward: ' + str(reward))
            if reward==1:
                training_data.extend(stack.items)
            if done: break

            #Check if the list has >4gb size
            cuatrogb=1e+9
            if sys.getsizeof(training_data)>cuatrogb:
                save_object(training_data,'File'+str(cc))
                print('Size of training data: ' + str(sys.getsizeof(training_data)))
                cc +=1
                training_data= []
                print('Save file')
                #sys.exit(0)
            print('Size of training data: ' + str(sys.getsizeof(training_data)))
            if(cc>50):
                sys.exit(0)



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
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    print(training_data)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

pop = initial_population()