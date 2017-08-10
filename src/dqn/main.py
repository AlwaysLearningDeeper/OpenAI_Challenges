from DQN_J2 import *
from utils.Stack import *
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2,re,random,time

STEPS= 100000000000
ENVIRONMENT = 'Breakout-v0'
SAVE_NETWORK = True
LOAD_NETWORK = True
BACKUP_RATE = 500
UPDATE_TIME = 100
NUM_CHANNELS = 4  # image channels
IMAGE_SIZE = 84  # 84x84 pixel images
SEED = 17  # random initialization seed
ACTIONS = [0,1,2,3]  # number of actions for this game
BATCH_SIZE = 32
INITIAL_EPSILON = 1.0
GAMMA = 0.99
RMS_LEARNING_RATE = 0.00025
RMS_DECAY = 0.95
RMS_MOMENTUM = 0.95

REPLAY_MEMORY_SIZE = 15000
#RMS_EPSILON = 1e-6
RMS_EPSILON = 0.01
REPLAY_MEMORY = 15000
FINAL_EXPLORATION_FRAME = 1000000
NO_OP_MAX = 30
NO_OP_CODE = 1

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def randomSteps(env,steps=REPLAY_MEMORY_SIZE):
    t0 = time.time()
    env.reset()
    i = 0
    frame_stack = Stack(4)
    initial_no_op = np.random.randint(4, NO_OP_MAX)

    for _ in range(0,steps):
        if i < initial_no_op:
            # WE PERFORM A RANDOM NUMBER OF NO_OP ACTIONS
            action = NO_OP_CODE
            state, reward, done, info = env.step(action)
            greyObservation = rgb2gray(state)
            state = downSample(greyObservation)
            frame_stack.push(state)
            i += 1
        else:

            state = np.stack(frame_stack.items, axis=2).reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

            action = dqn.selectAction(state,step)
            actionN = np.argmax(dqn.selectAction(state,step))

            next_state, reward, game_over, info = env.step(actionN)


            greyObservation = rgb2gray(next_state)
            next_state = downSample(greyObservation)

            frame_stack.push(next_state)

            next_state = np.stack(frame_stack.items, axis=2).reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

            dqn.storeExperience(state, action, reward, next_state, game_over)
            if done:
                #print("Episode finished after {} timesteps".format(_ + 1))
                env.reset()
                i=0
                frame_stack=[]



    t1 = time.time()
    print("This operation took:",t1-t0,)

def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)

if __name__ == '__main__':

    avg_Score_l20_plhldr = tf.placeholder(tf.float32,None,name="avg_scores")
    avg_Score_l20 = tf.summary.scalar("avg_Score_l20",avg_Score_l20_plhldr)
    env = gym.make(ENVIRONMENT)
    randomSteps(env,REPLAY_MEMORY_SIZE)
    actions = ACTIONS

    #Instanciate the DQN
    dqn = DQN(actions)
    action = NO_OP_CODE
    env.reset()

    sess = tf.InteractiveSession()

    #Saving and loading networks
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path and LOAD_NETWORK:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        game = int(re.match('.*?([0-9]+)$', checkpoint.model_checkpoint_path).group(1))
    else:
        print("Could not find old network weights")

    sess.run(tf.global_variables_initializer())
    game = 0
    game_scores =[]
    initial_no_op = np.random.randint(4, NO_OP_MAX)
    i=0
    frame_stack=Stack(4)
    score=0
    summary_writer = tf.summary.FileWriter('logs',sess.graph)
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

            action = dqn.selectAction(state,step)
            actionN = np.argmax(dqn.selectAction(state,step))

            next_state, reward, game_over, info = env.step(actionN)
            greyObservation = rgb2gray(next_state)
            next_state = downSample(greyObservation)
            frame_stack.push(next_state)
            next_state = np.stack(frame_stack.items, axis=2).reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))


            dqn.storeExperience(state, action, reward, next_state, game_over)

            score += reward

            minibatch = dqn.sampleExperiences()

            state_batch = [experience[0] for experience in minibatch]

            action_batch = [experience[1] for experience in minibatch]
            reward_batch = [experience[2] for experience in minibatch]
            nextState_batch = [experience[3] for experience in minibatch]
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
            # if step % 100 == 0:
            #     m, opt = sess.run([dqn.merged, dqn.trainStep],
            #                       {dqn.yInput: y_batch, dqn.actionInput: action_batch,
            #                        dqn.currentQNet.stateInput: state_batch})
            #     summary_writer.add_summary(m, step)
            # if step % UPDATE_TIME == 0:
            #     sess.run(dqn.copyCurrentToTargetOperation())

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
                    summary_scores = sess.run(avg_Score_l20, {avg_Score_l20_plhldr: np.mean(game_scores[-20:])})
                    summary_writer.add_summary(summary_scores, step)
                    print("The average score of all games is:", np.mean(game_scores))
                # else:
                #     print('Game %s finished with score %s' % (game, score))
                env.reset()
                initial_no_op = np.random.randint(4, NO_OP_MAX)
                i=0
                score = 0

