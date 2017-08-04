import random
import gym
import numpy as np


mutation_chance = 0.01
env = gym.make("CartPole-v1")

goal_steps = 200
def create_individual():
    # Create individual

    # [OBS, MOVES]
    training_data = []
    #Score
    score = 0
    # moves specifically from this environment:
    game_memory = []
    # previous observation that we saw
    prev_observation = env.reset()
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

    for data in game_memory:
        # convert to one-hot (this is the output layer for our neural network)
        if data[1] == 1:
            output = [0, 1]
        elif data[1] == 0:
            output = [1, 0]

        # saving our training data
        training_data.append([data[0], output])

    individual=[training_data,score]

    return individual

def create_population(count):
    return [ create_individual() for _ in range(count) ]

def evaluateIndividual(individual):
    #Evaluate individual by his score
    return individual[1]

def evaluatePopulation(population):
    summ=0
    for individual in population:
        summ += evaluateIndividual(individual)

    return summ/len(population)

def mutatePopulation(population):
    for individual in population:
        if mutation_chance > random.random():
            #mutate individual
            individual[0] = random.shuffle(individual[0])
            print('Individual mutated')
    return population


def evolve(population, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = sorted(population,reverse=True,key= lambda x:x[1])
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # mutate some individuals
    population = mutatePopulation(population)

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(population) - parents_length
    children = []
    while len(children) < desired_length:
        male = random.randint(0, parents_length - 1)
        female = random.randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents

