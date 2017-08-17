import gym,time,copy,re
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import matplotlib
from Replay_Memory import Replay_Memory


tf.logging.set_verbosity(tf.logging.INFO)


MEMORY_LENGTH = 4
ACTIONS = 4
LEARNING_RATE_SGD = 0.0002
LEARNING_RATE_RMSPROP = .0002
FINAL_EXPLORATION_FRAME = 2000000
TRAINING_STEPS = 20000000
DISCOUNT_RATE = 0.95
RMSPROP_MOMENTUM = 0.0
RMSPROP_DECAY = 0.99
RMSPROP_EPSILON = 1e-6
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 175000
RANDOM_STEPS_REPLAY_MEMORY_INIT = 175000
SUMMARY_STEPS = 100
initial_step = 0
NO_OP_MAX = 30
SAVE_PATH = "saved_networks"
LOG_DIRECTORY = "tmp/logs/"
RUN_STRING = "lr_0.0002,decay_0.99,momentum_0,discountRate_0.95,replayMemorySize_175000uint8,decaySteps_2000000,bias_0.1,weights_N(0,0.01),fast,fixedReduceSum"
ENVIRONMENT = 'Breakout-v0'
NO_OP_CODE = 0
TF_RANDOM_SEED = 7
type = np.dtype(np.uint8)

env = gym.make(ENVIRONMENT)
env.reset()
memory = Replay_Memory(REPLAY_MEMORY_SIZE)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def downSample(image):
    return cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)

def stack(frames):
    return np.stack(frames,2)


def randomSteps(steps=RANDOM_STEPS_REPLAY_MEMORY_INIT,initial_no_ops=4):
    t0 = time.time()
    env.reset()
    i = 0
    frame_stack = []


    for _ in range(0,steps):
        if i < initial_no_ops:
            action = NO_OP_CODE
            observation, reward, done, info = env.step(action)
            greyObservation = rgb2gray(observation)
            downObservation = downSample(greyObservation)
            frame_stack.append(downObservation)
            i+=1

        else:
            s_t = stack(frame_stack)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            greyObservation = rgb2gray(observation)
            downObservation = downSample(greyObservation)

            frame_stack.pop(0)
            frame_stack.append(downObservation)
            s_t_plus1 = stack(frame_stack)

            if done:
                memory.store_transition(
                    (
                    s_t.astype(type),
                    action,
                    reward,
                    None,
                    True
                )
                )

            else:
                memory.store_transition(
                    (
                        s_t.astype(type),
                        action,
                        reward,
                        s_t_plus1.astype(type),
                        False
                    )
                )

        if done:
            env.reset()
            i=0
            frame_stack=[]



    t1 = time.time()
    print("Filling the replay memory with random steps took: ",t1-t0," s")


def getEpsilon(step):
    if step > FINAL_EXPLORATION_FRAME:
        return 0.1
    else:
        return 1 - step*(0.9/FINAL_EXPLORATION_FRAME)


input_tensor = tf.placeholder(tf.float32,(None,84,84,4),name="input")
actions_tensor = tf.placeholder(tf.int32, [None], name="actions")
y_tensor = tf.placeholder(tf.float32, [None], name="r")

def model():
    #He initializer
    #weights_initializer = tf.contrib.layers.variance_scaling_initializer()

    #Xavier Glorot initializer
    #weights_initializer = tf.contrib.layers.xavier_initializer()


    #NIPS 2013 SPRAUGR parameters
    weights_initializer = tf.random_normal_initializer(stddev=0.01)
    biases_initializer = tf.constant_initializer(0.1)

    #Placeholders could be here
    conv_1 = tf.contrib.layers.conv2d(input_tensor,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID',weights_initializer=weights_initializer,biases_initializer=biases_initializer,activation_fn=tf.nn.relu)
    conv_2 = tf.contrib.layers.conv2d(conv_1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID',weights_initializer=weights_initializer,biases_initializer=biases_initializer,activation_fn=tf.nn.relu)
    conv_3 = tf.contrib.layers.conv2d(conv_2, num_outputs=64, kernel_size=[3,3],stride=[1,1],padding='VALID',weights_initializer=weights_initializer,biases_initializer=biases_initializer,activation_fn=tf.nn.relu)
    conv_3_flat = tf.contrib.layers.flatten(conv_3)
    relu_1 = tf.contrib.layers.relu(conv_3_flat, num_outputs=512,weights_initializer=weights_initializer,biases_initializer=biases_initializer)
    output = tf.contrib.layers.fully_connected(relu_1,activation_fn=None,num_outputs=ACTIONS,weights_initializer=weights_initializer,biases_initializer=biases_initializer)

    actions_one_hot = tf.one_hot(actions_tensor, ACTIONS, name="actions_one_hot")
    apply_action_mask = tf.multiply(output,actions_one_hot)
    Q_of_selected_action = tf.reduce_sum(apply_action_mask,axis=1)


    delta = tf.subtract(Q_of_selected_action, y_tensor)

    # Huber loss with delta=1
    # loss = tf.where(tf.abs(delta) < 1.0, 0.5 * tf.square(delta), tf.abs(delta) - 0.5)

    # MSE
    loss = tf.square(delta)

    cost = tf.reduce_mean(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE_SGD).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE_RMSPROP,momentum=RMSPROP_MOMENTUM,epsilon=RMSPROP_EPSILON,decay=RMSPROP_DECAY).minimize(cost)

    #Summary tensors
    cost_s = tf.summary.scalar("cost",cost)
    avg_Q = tf.summary.scalar("avg_Q",tf.reduce_mean(output))
    merged = tf.summary.merge_all()

    avg_Score_l20_plhldr = tf.placeholder(tf.float32,None,name="avg_scores")
    avg_Score_l20 = tf.summary.scalar("avg_Score_l20",avg_Score_l20_plhldr)


    return output,optimizer,merged,avg_Score_l20_plhldr,avg_Score_l20

def train():
    with tf.Session() as sess:
        tf.set_random_seed(TF_RANDOM_SEED)
        output, optimizer, merged,avg_Score_l20_plhldr,avg_Score_l20 = model()

        summary_writer = tf.summary.FileWriter(LOG_DIRECTORY + RUN_STRING,
                              sess.graph)

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(SAVE_PATH + "/" + RUN_STRING)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            initial_step = int(re.match('.*?([0-9]+)$', checkpoint.model_checkpoint_path).group(1))
        else:
            print("Could not find old network weights")
            initial_step = 0


        sess.run(tf.global_variables_initializer())
        score = 0
        game_scores = []
        i = 0
        frame_stack = []
        initial_no_op = np.random.randint(4,NO_OP_MAX)
        game = 1
        for step in range(initial_step,TRAINING_STEPS):
            if i < initial_no_op:
                # WE PERFORM A RANDOM NUMBER OF NO_OP ACTIONS
                action = NO_OP_CODE
                observation, reward, done, info = env.step(action)
                greyObservation = rgb2gray(observation)
                downObservation = downSample(greyObservation)
                if i > 3:
                    frame_stack.pop(0)
                frame_stack.append(downObservation)
                i += 1

            else:
                # CHOOSING ACTION
                s_t = stack(frame_stack)
                if np.random.rand() < getEpsilon(step):
                    # We make random action with probability epsilon
                    action = env.action_space.sample()
                else:
                    # Pick action in a greedy way
                    Q = sess.run([output],{input_tensor:np.array(s_t, ndmin=4)})
                    action = np.argmax(Q)


                # STORE TRANSITION

                observation, reward, done, info = env.step(action)
                score += reward

                #Process received frame
                greyObservation = rgb2gray(observation)
                downObservation = downSample(greyObservation)

                #Remove oldest frame
                frame_stack.pop(0)
                frame_stack.append(downObservation)

                #Obtain state at t+1
                s_t_plus1 = stack(frame_stack)

                if done:
                    memory.store_transition(
                        (
                            s_t.astype(type),
                            action,
                            reward,
                            None,
                            True
                        )
                    )

                else:
                    memory.store_transition(
                        (
                            s_t.astype(type),
                            action,
                            reward,
                            s_t_plus1.astype(type),
                            False
                        )
                    )

                # OBTAIN MINIBATCH
                actionsTerminal = []
                actionsNonTerminal = []
                yTerminal = []
                yNonTerminal = []
                framesTerminal = []
                framesNonTerminal = []


                for batch_i in range(0, MINIBATCH_SIZE):
                    t = memory.sample_transition()
                    if t[-1]:
                        framesTerminal.append(t[0])
                        actionsTerminal.append(t[1])
                        yTerminal.append(t[2])
                    else:
                        framesNonTerminal.append(t[0])
                        actionsNonTerminal.append(t[1])
                        yNonTerminal.append(t[2])

                if len(framesNonTerminal) > 0:
                    V = []
                    out = sess.run([output], {input_tensor: np.array(framesNonTerminal, ndmin=4)})[0]
                    for out_index in range(0,len(out)):
                        V.append(DISCOUNT_RATE*np.max(out[out_index]))
                    yNonTerminal = np.sum((np.array(yNonTerminal),V), axis=0)


                if len (yNonTerminal) == MINIBATCH_SIZE:
                    frames = np.array(framesNonTerminal, ndmin=4)
                    actions = np.array(actionsNonTerminal)
                    y = np.array(yNonTerminal)

                elif len(yTerminal) == MINIBATCH_SIZE:
                    frames = np.array(framesTerminal, ndmin=4)
                    actions = np.array(actionsTerminal)
                    y = np.array(yTerminal)

                else:
                    framesTerminal = np.array(framesTerminal, ndmin=4)
                    framesNonTerminal = np.array(framesNonTerminal, ndmin=4)

                    frames = np.concatenate((framesTerminal,framesNonTerminal))
                    actions = np.concatenate((actionsTerminal,actionsNonTerminal))
                    y = np.concatenate((yTerminal,yNonTerminal))

                if step % SUMMARY_STEPS == 0:
                    m, opt = sess.run([merged,optimizer],
                             {input_tensor: frames, actions_tensor:actions, y_tensor: y})
                    summary_writer.add_summary(m, step)

                else:
                    sess.run([optimizer],{input_tensor:frames,actions_tensor:actions,y_tensor:y})


                if done:
                    frame_stack = []
                    game += 1
                    game_scores.append(score)
                    if game % 1000 == 0:
                        saver.save(sess, SAVE_PATH +"/" +RUN_STRING +"/" + ENVIRONMENT + '-dqn', global_step=step)
                        print('Network backup done')
                    if (game % 20) == 0:
                        print("The average score of the last 20 games is:", np.mean(game_scores[-20:]),
                              " currently at game ", game, " , step ", step)
                        summary_scores = sess.run(avg_Score_l20, {avg_Score_l20_plhldr: np.mean(game_scores[-20:])})
                        summary_writer.add_summary(summary_scores, step)
                        print("The average score of all games is:", np.mean(game_scores))
                    score = 0
                    env.reset()
                    initial_no_op = np.random.randint(4, NO_OP_MAX)
                    i=0


def sampleFrames():
    for _ in range(10):
        t = np.split(memory.sample_transition()[0],4,2)
        plt.imshow(t[0].reshape(84, 84), cmap=matplotlib.cm.Greys_r)
        plt.show()

randomSteps()
#sampleFrames()
train()
