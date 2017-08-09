from DQN_J2 import *
from utils.Stack import *
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2,re,random

STEPS= 10000000
ENVIRONMENT = 'Breakout-v0'
SAVE_NETWORK = True
BACKUP_RATE = 500
NUM_CHANNELS = 4  # image channels
IMAGE_SIZE = 84  # 84x84 pixel images
SEED = 17  # random initialization seed
ACTIONS = [1,2,3]  # number of actions for this game
BATCH_SIZE = 20000
INITIAL_EPSILON = 1.0
GAMMA = 0.99
RMS_LEARNING_RATE = 0.00025
RMS_DECAY = 0.99
RMS_MOMENTUM = 0.0
RMS_EPSILON = 1e-6
NO_OP_CODE = 1

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)

if __name__ == '__main__':
    env = gym.make(ENVIRONMENT)
    actions = ACTIONS

    #Instanciate the DQN
    dqn = DQN(actions)
    action = NO_OP_CODE
    env.reset()

    sess = tf.InteractiveSession()

    #Saving and loading networks
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        game = int(re.match('.*?([0-9]+)$', checkpoint.model_checkpoint_path).group(1))
    else:
        print("Could not find old network weights")

    sess.run(tf.global_variables_initializer())
    game = 0
    game_scores =[]
    initial_no_op = np.random.randint(4, 30)
    i=0
    frame_stack=Stack(4)
    score=0

    print('Started training')
    for step in range(STEPS):
        if i < initial_no_op:
            # WE PERFORM A RANDOM NUMBER OF NO_OP ACTIONS
            action = NO_OP_CODE
            state, reward, done, info = env.step(action)
            greyObservation = rgb2gray(state)
            state = downSample(greyObservation)
            frame_stack.push(state)
            i+=1
        else:

            state = np.stack(frame_stack.items, axis=2).reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            action = dqn.selectAction(state)
            actionN = np.argmax(dqn.selectAction(state))
            #print(actionN)
            next_state, reward, game_over, info = env.step(actionN)

            #if reward > -1:
            greyObservation = rgb2gray(next_state)
            next_state = downSample(greyObservation)
            next_state = np.stack((next_state, next_state, next_state, next_state), axis=2).reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            #next_state = np.append(next_state, state, axis=2)
            dqn.storeExperience(state, action, reward, next_state, game_over)

            score += reward

            minibatch = dqn.sampleExperiences()

            state_batch = [experience[0] for experience in minibatch]

            action_batch = [experience[1] for experience in minibatch]
            reward_batch = [experience[2] for experience in minibatch]
            nextState_batch = [experience[3] for experience in minibatch]
            # print(len(nextState_batch))
            # sys.exit(0)
            terminal_batch = [experience[4] for experience in minibatch]



            y_batch = []
            Q_batch = sess.run(dqn.targetQNet.QValue, feed_dict={dqn.targetQNet.stateInput: nextState_batch})
            for i in range(len(minibatch)):
                terminal = terminal_batch[i]
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * np.max(Q_batch[i]))

            currentQ_batch = sess.run(dqn.currentQNet.QValue,
                                      feed_dict={dqn.currentQNet.stateInput: state_batch})

            sess.run(dqn.trainStep, feed_dict={dqn.yInput: y_batch, dqn.actionInput: action_batch,dqn.currentQNet.stateInput: state_batch})

            state = next_state

            if game_over:
                frame_stack.empty()
                game +=1
                game_scores.append(score)
                if game % BACKUP_RATE == 0 and SAVE_NETWORK:
                    saver.save(sess, 'saved_networks/' + ENVIRONMENT + '-dqn', global_step=game)
                    print('Network backup done')
                if (game % 20) == 0:
                    print("The average score of the last 20 games is:", np.mean(game_scores[-20:]),
                          " currently at game ", game, " , step ", step)
                    print("The average score of all games is:", np.mean(game_scores))
                # else:
                #     print('Game %s finished with score %s' % (game, score))
                env.reset()
                initial_no_op = np.random.randint(4, 50)
                i=0
                score = 0