from keras.layers import Input,Dense
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.base_layer import InputSpec
from keras import initializers
import numpy as np 
from memory import ReplayBuffer


# Build a class for Noisy Nets Layers (inspired from source code of Dense Layer in keras):
# https://github.com/keras-team/keras/blob/master/keras/layers/core.py/#L765
class Noisy_Layer(Dense):
    def __init__(self,units, **kwargs):
        self.units = units #output dimension
        super(Noisy_Layer, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
            
        #trainable parameters : mu_w,mu_b,sigma_w,sigma_b

        self.mu_w = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='mu_w',
                                      regularizer=None,
                                      constraint=None)
        
        self.mu_b = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='mu_b',
                                        regularizer=None,
                                        constraint=None)
        
        self.sigma_w = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(0.01), #constant initialization
                                      name='sigma_w',
                                      regularizer=None,
                                      constraint=None)
        
        self.sigma_b = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(0.01), #constant initialization
                                        name='sigma_b',
                                        regularizer=None,
                                        constraint=None)
        
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True
               
    def call(self,inputs):
        # non trainable parameters : epsilon_w, epsilon_b
        self.epsilon_w = K.random_normal(shape=(self.input_dim, self.units))
        self.epsilon_b = K.random_normal(shape=(self.units,))
        w = self.mu_w + self.sigma_w * self.epsilon_w # multiply : elementwise
        b = self.mu_b + self.sigma_b * self.epsilon_b # multiply : elementwise
        output = K.dot(inputs, w) + b
        if self.activation is not None:
            output = self.activation(output)
        return output
        
        
        
## Build the Neural Network
        
def Neural_Network(lr,n_actions,input_dim):
    Model_input = Input(shape=(input_dim,))
    Noisy1 = Noisy_Layer(256,input_shape=(input_dim,),activation='relu')(Model_input)
    Noisy2 = Noisy_Layer(256,activation='relu')(Noisy1)
    Model_output = Noisy_Layer(n_actions)(Noisy2)
    model = Model(inputs = Model_input, outputs = Model_output)
    model.compile(optimizer=Adam(lr=lr),loss='mse')
    return model
    
    
## Class Noisy DQN Agent 
class Noisy_DQN_Agent(object):
    
    def __init__(self,env,input_dim,n_actions,alpha,gamma,batch_size,lr=5e-4,
                 memory_size=10000000,replace_target=5,
                 filename='noisy_dqn.h5'):
        self.env = env
        self.action_space= np.arange(n_actions)
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.alpha = alpha #learning rate
        self.gamma=gamma #discount factor 
        self.batch_size=batch_size
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
        
    # choose greedy action (exploration is kept with noisy nets)
    def choose_action(self, state):
        state = state.reshape(1,-1)
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
        if self.memory.memory_count % self.replace_target ==0:
            self.update_target()
        
    def update_target(self): #update the parameters of target network from online network
        self.target_network.set_weights(self.online_network.get_weights())
        
        
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


    
    


 
