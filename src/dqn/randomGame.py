import gym,time

env = gym.make('Breakout-v0')
env.reset()
goal_steps=500

def random_game():
    env.reset()
    for _ in range(1000):
        env.render()
        #time.sleep(0.1)
        env.step(env.action_space.sample())

random_game()