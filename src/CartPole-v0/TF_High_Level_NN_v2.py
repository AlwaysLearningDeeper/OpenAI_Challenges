import gym
import time
import random
import numpy as np
import tensorflow as tf
from statistics import median, mean
from collections import Counter
import os


#tf.logging.set_verbosity(tf.logging.FATAL)

LR = 1e-3
DROPOUT_RATE = 0.3
env = gym.make("CartPole-v1")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000


def initial_population():
    """
    Extracts good runs from random games. Code from sentdex
    :return training_data:
    """
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
            # choose random action (0 or 1)
            action = random.randrange(0, 2)
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


    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data



def modelv2(x):
    network = tf.contrib.layers.relu(x, 128)
    network = tf.layers.dropout(network,rate=DROPOUT_RATE,training=dropout)
    network = tf.contrib.layers.relu(network, 256)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 512)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 256)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 128)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    output = tf.contrib.layers.fully_connected(network, 2,activation_fn = tf.nn.softmax)

    return output

x = tf.placeholder(tf.float32, shape=(None,4), name='x')
y = tf.placeholder(tf.float32, shape=(None,2), name='y')
dropout = tf.placeholder(tf.bool,shape=None,name ='dropout')


#training_data = initial_population()
#np.save("jairsanTrainingData", training_data)
training_data = np.load("jairsanTrainingData.npy")

Xtrain = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
ytrain = [i[1] for i in training_data]

nn = modelv2(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)


with tf.Session() as sess:
    tf.set_random_seed(7)
    sess.run(tf.global_variables_initializer())

    # Train
    for epoch in range(15):
        epoch_loss = 0
        ca, c = sess.run([optimizer, cost], feed_dict={x:Xtrain,y:ytrain,dropout:True})
        epoch_loss += c
        print('Epoch', epoch, 'loss', epoch_loss)



    scores = []
    choices = []
    for each_game in range(100):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):

            #env.render()


            if len(prev_obs) == 0:
                action = 0
            else:
                action = np.argmax(sess.run([nn], feed_dict={x:prev_obs.reshape(-1, len(prev_obs)),dropout:False}))

            choices.append(action)

            new_observation, reward, done, info = env.step(action)

            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward

            if done:
                break


        scores.append(score)

    print('Average Score:', sum(scores) / len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
    print(score_requirement)
