import numpy as np 

## Memory 
class ReplayBuffer(object):  
    def __init__(self, max_size, input_shape):
        #input shape : shape of a state array
        self.memory_size = max_size #max_size of memory 
        self.memory_count = 0 # to keep track of storage
        self.state_memory = np.zeros((self.memory_size,input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, input_shape),
                                         dtype=np.float32)
        # to keep track of the new states after taking an action
        self.action_memory = np.zeros(self.memory_size,dtype=np.int8)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.int8)
        
    # store the transitions
    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_count % self.memory_size 
        self.memory_count += 1
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] =int(done)#done=True if episode is over
            
    # sample a subset from memory
    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_count, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, new_states, terminal
