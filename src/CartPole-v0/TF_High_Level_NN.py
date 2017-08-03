import gym
import time
import random
import numpy as np
import tensorflow as tf
from statistics import median, mean
from collections import Counter
import os


tf.logging.set_verbosity(tf.logging.FATAL)

LR = 1e-3
env = gym.make("CartPole-v0")
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


def model_fn(features, targets, mode, params):
    """Model function for Estimator."""

    network = tf.contrib.layers.relu(features, 128)
    network = tf.contrib.layers.relu(network, 256)
    network = tf.contrib.layers.relu(network, 512)
    network = tf.contrib.layers.relu(network, 256)
    network = tf.contrib.layers.relu(network, 128)
    predictions = tf.contrib.layers.fully_connected(network, 2,activation_fn = tf.nn.softmax)



    # Reshape output layer to 1-dim Tensor to return predictions
    predictions_dict = {"actions": predictions}

    # Calculate loss using softmax
    loss =tf.losses.softmax_cross_entropy(targets, predictions)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="SGD")

    return predictions_dict, loss, train_op


def train_model(training_data):
    model_params = {"learning_rate": LR}
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = [i[1] for i in training_data]

    nn = tf.contrib.learn.SKCompat(tf.contrib.learn.Estimator(
    model_fn=model_fn, params=model_params))

    nn.fit(x=X, y=y, batch_size=None, max_steps=5)



    return nn


model = train_model(training_data=initial_population())

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        #env.render()


        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            #t0 = time.time()
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))['actions'][0])
            #t1 = time.time()
            #print("Took: ", t1 - t0)



        choices.append(action)


        new_observation, reward, done, info = env.step(action)

        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward

        if done: break


    scores.append(score)