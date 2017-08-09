from DQN_J2 import *
from utils.Stack import *
import gym
import tensorflow as tf
import numpy as np
import cv2,re,random

STEPS= 10000
ENVIRONMENT = 'Breakout-v0'
UPDATE_TIME = 100
NUM_CHANNELS = 4  # image channels
IMAGE_SIZE = 84  # 84x84 pixel images
SEED = 17  # random initialization seed
ACTIONS = [1,2,3,4]  # number of actions for this game
BATCH_SIZE = 100
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
    dqn = DQN(actions)
    action = NO_OP_CODE
    env.reset()
    #state, reward, done, info = env.step(action)
    #greyObservation = rgb2gray(state)
    #state = downSample(greyObservation)
    #state = np.stack((state, state, state, state), axis=2).reshape((84, 84, 4))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        game = int(re.match('.*?([0-9]+)$', checkpoint.model_checkpoint_path).group(1))
    else:
        print("Could not find old network weights")
    sess.run(tf.initialize_all_variables())
    game = 0
    game_scores =[]
    initial_no_op = np.random.randint(4, 30)
    i=0
    frame_stack=Stack(4)
    score=0
    for step in range(STEPS):
        if i < initial_no_op:
            # WE PERFORM A RANDOM NUMBER OF NO_OP ACTIONS
            action = NO_OP_CODE
            state, reward, done, info = env.step(action)
            greyObservation = rgb2gray(state)
            state = downSample(greyObservation)
            frame_stack.push(state)
            i+=1
        # elif i==initial_no_op:
        #     print(len(frame_stack))
        #     num = random.sample(range(1, len(frame_stack) - 1), 4)
        #     frame_stack = (frame_stack[num[0]], frame_stack[num[1]], frame_stack[num[2]], frame_stack[num[3]])
        else:
            state = np.stack(frame_stack.items, axis=2).reshape((84, 84, 4))
            action = dqn.selectAction(state)
            actionN = np.argmax(dqn.selectAction(state))
            #print(actionN)
            next_state, reward, game_over, info = env.step(actionN)

            #if reward > -1:
            greyObservation = rgb2gray(next_state)
            next_state = downSample(greyObservation)
            next_state = np.stack((next_state, next_state, next_state, next_state), axis=2).reshape((84, 84, 4))
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
                if game % 1000 == 0:
                    saver.save(sess, 'saved_networks/' + ENVIRONMENT + '-dqn', global_step=game)
                    print('Network backup done')
                if (game % 20) == 0:
                    print("The average score of the last 20 games is:", np.mean(game_scores[-20:]),
                          " currently at game ", game, " , step ", step)
                    print("The average score of all games is:", np.mean(game_scores))
                else:
                    print('Game %s finished with score %s' % (game, score))
                env.reset()
                initial_no_op = np.random.randint(4, 50)
                i=0
                score = 0