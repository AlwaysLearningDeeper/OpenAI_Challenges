import gym
import numpy as np
import neat
import os
import random
import time
from gym import wrappers

env = gym.make('MountainCar-v0')
print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))

print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 200
print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))

min_reward = -200
max_reward = 200
goal_steps = 200
discounted_reward = 0.9
score_range=[]
trials=100
generations=200
#Fitness
def compute_fitness(net, rewards, episodes):
    reward_error=[]

    for reward, episode in zip(rewards,episodes):
        for(episode, reward) in zip(episode,[reward]):
            action=episode[2]
            output=net.activate(episode[1])
            reward_error.append((float(output[action]))**2)
    return reward_error


def eval_genomes(genomes, config):
    nets=[]
    for genome_id, genome in genomes:
        nets.append((genome,neat.nn.FeedForwardNetwork.create(genome, config)))
        genome.fitness = []
    episodes=[]
    t0=time.time()
    for genome, net in nets:
        observation = env.reset()
        episode_data=[]
        j=0
        score=0
        for _ in range(goal_steps):
            if net is not None:
                output = net.activate(observation)
                action = np.argmax(output)
            else:
                action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            score+=reward
            episode_data.append((j,observation,action,score)) #reward instead of score?
            if done: break
            j+=1

        episodes.append((score, episode_data))
        genome.fitness = score

    print('Simulation run time {0}'.format(time.time() - t0))
    t0=time.time()
    scores = [s for s, e in episodes]
    score_range.append((min(scores),np.mean(scores),max(scores)))
    print(score_range)
    #Normalising
    normRewards=[]

    for i in range(len(episodes)):
        normRewards.append(2*(episodes[i][0]-min_reward)/(max_reward-episodes[i][0])-1.0)

    comparison_episodes = [random.choice(episodes)[1] for _ in range(100)]
    reward_errors=[]

    for genome,net in nets:
        reward_errors.append(compute_fitness(net,normRewards,comparison_episodes))

    for reward_error,(genome_id,genome) in zip(reward_errors,genomes):
        genome.fitness -= np.mean(reward_error)

    print("Final fitness compute time {0}\n".format(time.time()-t0))
    #########################################################################

def run(config_file):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # Create the population, which is the top-level object for a NEAT run
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    #Run for 300 generation
    winner = population.run(eval_genomes,generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    scores=[]

    # Create the environment for the test and wrap it with a Monitor
    env = gym.make('MountainCar-v0')
    env = wrappers.Monitor(env,'tmp/MountainCar-v0')

    for i in range(trials):
        score=0
        observation=env.reset()
        for _ in range(goal_steps):
            action = np.argmax(winner_net.activate(observation))
            # do it!
            observation, reward, done, info = env.step(action)
            score += reward
            if done: break
        scores.append(score)
    print("The winning neural network obtained an average score of: "+str(np.average(scores)))
    if(np.average(scores)>-110):
        gym.upload('tmp/MountainCar-v0',api_key='sk_tiwKaUHVQDChjmO9JmK2Gg')
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
