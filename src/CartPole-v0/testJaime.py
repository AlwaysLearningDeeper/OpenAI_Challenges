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

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 70
initial_games = 30000

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 2
batch_size = 200


def create_randoms():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break



def create_population():
    training_data = []
    scores = []
    accepted_scores= []
    for iteration in range(initial_games):
        if (iteration%100 == 0):
            print('Initial game number ',iteration)

        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])

            prev_observation = observation
            score += reward
            if done:
                break
        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output =[0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0],output])
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)

    print('Average accepted score:',mean(accepted_scores))
    print('Median accepted score:',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data



def neural_network_modelv1(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def neural_network_modelv2(data):
    # (input_data * weights) +biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([4, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }
    data = tf.cast(data, tf.float32)
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output_layer = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output_layer


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_modelv1(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

x=tf.placeholder('float')
y=tf.placeholder('float')

def train_modelv2(trainingdata, model=False):
    #Divide data for training and test
    maxL = len(trainingdata)
    corner = int(maxL*0.8)
    print(corner)
    test_data,training_data =np.split(trainingdata,[corner])

    print(test_data)
    trainingX = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    trainingY = [i[1] for i in training_data]

    testX = np.array([i[0] for i in test_data]).reshape(-1, len(test_data[0][0]))
    testY = [i[1] for i in test_data]

    prediction = neural_network_modelv2(trainingX)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 1000

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Train
        for epoch in range(hm_epochs):
            epoch_loss = 0
            ca, c = sess.run([optimizer, cost], feed_dict={x: trainingX, y: trainingY})
            epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

        # Test
        print(ca)
        print(tf.argmax(prediction, 1))
        print(tf.argmax(y, 1))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x: testX, y: testY}))









training_data = create_population()
model = train_modelv2(training_data)

# scores = []
# choices = []
# for each_game in range(10):
#     score = 0
#     game_memory = []
#     prev_obs = []
#     env.reset()
#     for _ in range(goal_steps):
#         #env.render()
#
#         if len(prev_obs) == 0:
#             action = random.randrange(0, 2)
#         else:
#             action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
#
#         choices.append(action)
#
#         new_observation, reward, done, info = env.step(action)
#         prev_obs = new_observation
#         game_memory.append([new_observation, action])
#         score += reward
#         if done: break
#
#     scores.append(score)
#
# print('Average Score:', sum(scores) / len(scores))
# print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
# print(score_requirement)


