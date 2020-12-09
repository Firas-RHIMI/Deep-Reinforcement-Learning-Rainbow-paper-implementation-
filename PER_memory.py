## Credits : https://pylessons.com/CartPole-PER/
## adapted version
## We use the SumTree in case of PER (Prioritized Experience Replay) 

import numpy as np
# in a SumTree with n=capacity leaf nodes, we have : 
# 2*capacity-1 nodes 
# capacity -1 parent nodes
# in the root : we store the total priority (sum of priorities)
# in each leaf node we store the priority of an experience
# we store in the different data arrays the data (state,action,reward,new_state,done)
 

""" tree:
           0
          / \
         0   0
        / \ / \
tree_idx  0 0  0  We fill the leaves from left to right
 """

class SumTree:   
    def __init__(self, capacity,input_shape):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.state_memory = np.zeros((self.capacity,input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.capacity, input_shape),
                                         dtype=np.float32)
        # to keep track of the new states after taking an action
        self.action_memory = np.zeros(self.capacity,dtype=np.int8)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.terminal_memory = np.zeros(self.capacity, dtype=np.int8)
        self.pointer = 0 # index in data arrays
                      
    def add(self, p, experience):# add priority in SumTree leaf and experience in data arrays
        tree_idx = self.pointer + self.capacity - 1 #index in the tree
        state, action, reward, new_state, done = experience
        self.state_memory[self.pointer] = state
        self.new_state_memory[self.pointer] = new_state
        self.action_memory[self.pointer] = action
        self.terminal_memory[self.pointer] = int(done)
        self.reward_memory[self.pointer] = reward
        self.update(tree_idx, p) # change priority value and propagate the change through the tree
        self.pointer += 1
        if self.pointer >= self.capacity: #overwrite(back to first index)
            self.pointer = 0        
    
    def update(self, idx, p):# update priority values and propagate changes through the tree
        change = p - self.tree[idx]
        self.tree[idx] = p
        self.propagate(idx, change) #propagate the change through the tree
        
    def propagate(self, idx, change): #propagate the change to parent nodes
        while idx !=0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
           
    def get_priority(self, v): #get the leaf_index, priority value and experience associated
        parent_index = 0
        while True:
            left_child_index = 2*parent_index + 1
            right_child_index =left_child_index + 1
            # If we reach bottom
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
        # search for a higher priority node
            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index =leaf_index-self.capacity + 1
        state =self.state_memory[data_index] 
        action = self.action_memory[data_index]
        reward= self.reward_memory[data_index]
        new_state = self.new_state_memory[data_index]
        done= self.terminal_memory[data_index] 
        data = (state,action,reward,new_state,done)

        return leaf_index,self.tree[leaf_index],data

    

class PER_ReplayBuffer(object):  # stored as ( state, action, reward, next_state ) in SumTree
    #PER_eps = 0.01  # constant assures that no experience has 0 probability to be taken
    #PER_a = 0.6  #a tradeoff between taking exp with high priority and sampling randomly
    #PER_b  # importance-sampling, from initial value increasing to 1
        
    def __init__(self, max_size, input_shape, PER_eps = 0.01,PER_a = 0.6,PER_b = 0.4,
                 PER_b_increment=1.005 ,PER_b_end=1):
        # Making the tree 
        self.tree = SumTree(max_size,input_shape)
        self.PER_eps = PER_eps
        self.PER_a = PER_a
        self.PER_b = PER_b
        self.PER_b_increment = PER_b_increment
        self.PER_b_end = PER_b_end
        self.empty = True # all priorities are 0
      
    def store_transition(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.empty:
            self.tree.add(1, experience) # 1 is a random choice , could be any other value 
        else:
            priority = np.max(self.tree.tree[-self.tree.capacity:])
            self.tree.add(priority, experience)
    
    def sample_buffer(self,batch_size):
        total=self.tree.tree[0]
        states=[]
        actions=[]
        rewards=[]
        new_states=[]
        terminals=[]
        segment= total/batch_size
        indexes=[]
        importances=[] # for importance sampling
        for i in range(batch_size):
            low,high= segment * i, segment * (i + 1)
            v= np.random.uniform(low,high)
            index,priority,transition=self.tree.get_priority(v)
            state,action,reward,new_state,terminal=transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)
            terminals.append(terminal)
            indexes.append(index) 
            p_i=priority/total
            self.PER_b = min(self.PER_b*self.PER_b_increment,self.PER_b_end)
            w_i=(p_i*batch_size)**(- self.PER_b)
            importances.append(w_i)
            
        return np.array(states),np.array(actions),np.array(rewards),np.array(new_states),np.array(terminals),np.array(indexes),np.array(importances)
            
    def update_batch(self,indexes,errors):
        errors += self.PER_eps # to avoid division by 0
        priorities = errors ** self.PER_a
        for index, priority in zip(indexes, priorities):
            self.tree.update(index,priority)
        return
            
            

