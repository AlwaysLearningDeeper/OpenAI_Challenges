import numpy as np
from matplotlib import pyplot as plt
import matplotlib

class Replay_Memory:
    def __init__(self,memory_size):
        self.memory = []
        self.memory_size = memory_size
        self.stored_transitions = 0
        self.oldest_transition = 0


    def store_transition(self,transition):
        if self.stored_transitions < self.memory_size:
            self.memory.append(transition)
            self.stored_transitions+=1
        else:
            self.memory[self.oldest_transition] = transition
            self.oldest_transition = (self.oldest_transition + 1) % self.memory_size

    def sample_transition(self):
        return self.memory[np.random.randint(0,self.stored_transitions)]



    #AUX_Methods
    def dump_memory(self):
        for t in self.memory:
            print(t[0].shape,t[1],t[2],t[3].shape)
            #print("Type:",t[0].dtype)

    def split_transition(self,t):
        sa = np.split(t,4,2)
        print(sa[0].shape)
        plt.imshow(sa[0].reshape(84,84), cmap=matplotlib.cm.Greys_r)
        plt.show()


    def show_memory(self):
        for t in self.memory:
            self.split_transition(t[0])