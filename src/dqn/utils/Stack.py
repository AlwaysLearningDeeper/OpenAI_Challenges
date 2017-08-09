class Stack:
    def __init__(self,length):
        self.items = []
        self.length = length

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)
        if(self.size() > self.length):
            self.pop()

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def empty(self):
        self.items=[]