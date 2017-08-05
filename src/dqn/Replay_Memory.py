class Replay_Memory:
    def __init__(self):
        self.memory = []

    def store_transition(self,transition):
        self.memory.append(transition)

    def dump_memory(self):
        for t in self.memory:
            print(t[0].shape,t[1],t[2],t[3].shape)