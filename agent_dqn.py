from keras.layers import Dense,Activation
from keras.models import Sequential,load_model
from keras.optimizers import Adam
import numpy as np 
from memory import ReplayBuffer

## build the neural network 
def Neural_Network(lr,n_actions,input_dim):
    model=Sequential()
    model.add(Dense(256,input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))
    model.compile(optimizer=Adam(lr=lr),loss='mse')
    return model


## Class DQN Agent 
class DQN_Agent(object):
    
    def __init__(self,env,input_dim,n_actions,alpha,gamma,epsilon,batch_size,lr=5e-4,
                 epsilon_dec=0.995,epsilon_end=0.05,memory_size=10000000,replace_target=5,
                 filename='dqn.h5'):
        self.env = env
        self.action_space= np.arange(n_actions)
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.alpha = alpha #learning rate
        self.gamma=gamma #discount factor 
        self.epsilon = epsilon #eps-greedy
        self.batch_size=batch_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.filename = filename
        self.memory = ReplayBuffer(memory_size,input_dim)
        self.scores = [] # to keep track of scores
        self.avg_scores=[]
        self.replace_target = replace_target
        self.online_network=Neural_Network(lr,n_actions,input_dim) #network for evaluation
        self.target_network=Neural_Network(lr,n_actions,input_dim) #network for computing target
        # online and target network are the same except that parameters of target network
        # are copied each "replace target" steps from online network's parameters and kept
        # fixed on all other steps
        
        
    # to interface with memory
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    # choose epsilon greedy action (to keep exploration)
    def choose_action(self, state):
        state = state.reshape(1,-1)
        rand=np.random.random()
        if rand<self.epsilon:
            action=np.random.choice(self.action_space)
        else:
            actions=self.online_network.predict(state)
            action= np.argmax(actions)          
        return action 
    
    def update_online(self):#update parameters of the online network
        #we start learning after at least batch_size sample in memory
        if self.memory.memory_count< self.batch_size: 
            return   
        states, actions, rewards, new_states, done =self.memory.sample_buffer(self.batch_size)
        q_estimate = self.online_network.predict(states)
        q_next = self.target_network.predict(new_states) # used to compute target
        q_target = q_estimate.copy()
        batch_index= np.arange(self.batch_size,dtype=np.int32)
        q_target[batch_index,actions] = rewards + self.gamma * np.max(q_next,axis=1)*(1-done)
        # if episode over, 1-done = 0 , Q(terminal,)=0
        self.online_network.fit(states,q_target,verbose=0)
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon>self.epsilon_end else self.epsilon_end
        if self.memory.memory_count % self.replace_target ==0:
            self.update_target()
        
    def update_target(self): #update the parameters of target network from online network
        self.target_network.model.set_weights(self.online_network.model.get_weights())
        
        
    def train(self,n_games,path):
        # path : path where to save the model
        for i  in range(n_games):   
            score=0
            done = False
            state = self.env.reset()
            while not done:
                action = self.choose_action(state)
                new_state,reward,done,info= self.env.step(action)
                score+= reward
                self.remember(state, action, reward, new_state, done)
                state = new_state
                self.update_online()     
            self.scores.append(score)
            avg_score = np.mean(self.scores[max(0,i-50):i+1]) # rolling score : mean 
            self.avg_scores.append(avg_score)
            print('episode ',i,'score = %.2f'%score,' Rolling-score = %.2f'%avg_score)  
            # save the model after 100 games
            if i%100 ==0 and i>0:
                self.save_model(path)
            
    def save_model(self,path):
        self.online_network.save(path+'/'+ self.filename)
    
    
    def load_model(self,path):
        self.online_network= load_model(path)
