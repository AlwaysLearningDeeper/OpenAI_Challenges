import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import tensorflow as tf
import sys

tf.logging.set_verbosity(tf.logging.FATAL)

LR = 1e-3
env = gym.make("CartPole-v1")
env.reset()
DROPOUT_RATE = 0.3
goal_steps = 500
score_requirement = 50
initial_games = 100000

n_nodes_hl1 = 128
n_nodes_hl2 = 256
n_nodes_hl3 = 512
n_nodes_hl4 = 256
n_nodes_hl5 = 128
n_classes = 2
epochN=15


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


def neural_network_modelv2():

    x = tf.placeholder(tf.float32, shape=(None, 4), name='x')

    # # #(input_data * weights) +biases
    #
    # hidden_1_layer = {'weights': tf.Variable(tf.random_normal([4, n_nodes_hl1])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #
    # hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #
    # hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    #
    # hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    #
    # hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}
    #
    # output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])),
    #                 'biases': tf.Variable(tf.random_normal([n_classes])), }
    #
    # l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # l1 = tf.nn.relu(l1)
    # #l1 = tf.nn.dropout(l1,0.2)
    #
    # l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    # l2 = tf.nn.relu(l2)
    # #l2 = tf.nn.dropout(l2,0.2)
    #
    # l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)
    # #l3 = tf.nn.dropout(l3,0.2)
    #
    # l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    # l4 = tf.nn.relu(l4)
    # #l4 = tf.nn.dropout(l4,0.2)
    #
    # l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    # l5 = tf.nn.relu(l5)
    # #l5 = tf.nn.dropout(l5,0.2)
    #
    # output_layer = tf.matmul(l5, output_layer['weights']) + output_layer['biases']
    # output_layer = tf.nn.softmax(output_layer)
    #
    # return output_layer

    network = tf.contrib.layers.relu(x, 128)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 256)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 512)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 1024)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 2046)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 4096)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 2046)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 1024)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 512)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 256)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    network = tf.contrib.layers.relu(network, 128)
    network = tf.layers.dropout(network, rate=DROPOUT_RATE, training=dropout)
    output = tf.contrib.layers.fully_connected(network, 2, activation_fn=tf.nn.softmax)

    return output

dropout = tf.placeholder(tf.bool,shape=None,name ='dropout')

def playthegame(training_data):
    y=tf.placeholder(tf.float32,shape=(None,2), name='y')
    trainingX = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    trainingY = [i[1] for i in training_data]

    nn = neural_network_modelv2()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tf.set_random_seed(7)
        # Train
        for epoch in range(epochN):
            epoch_loss = 0
            ca, c = sess.run([optimizer, cost], feed_dict={'x:0': trainingX, 'y:0': trainingY,dropout:True})
            epoch_loss += c
            print('Epoch', epoch, 'loss', epoch_loss)
        print('Training done')


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
                    action = random.randrange(0, 2)
                else:
                    action = np.argmax(sess.run([nn], feed_dict={'x:0':prev_obs.reshape(-1, len(prev_obs)),dropout:True}))
                choices.append(action)
                new_observation, reward, done, info = env.step(action)

                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score += reward

                if done: break


            scores.append(score)

        print('Average Score:', sum(scores) / len(scores))
        print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
        print(score_requirement)

training_data = initial_population()
playthegame(training_data)