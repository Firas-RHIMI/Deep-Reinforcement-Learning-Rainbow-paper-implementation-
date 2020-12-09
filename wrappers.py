# credits for visualizing videos in Colab environment
#https://colab.research.google.com/drive/1flu31ulJlgiRL1dnN2ir8wGh9p7Zij2t#scrollTo=8nj5sjsk15IT
# adapted version
"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay


def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env


def test_agent(agent,env):
  state=env.reset()
  done = False
  while not done:
      env.render()
      action = np.argmax(agent.online_network.predict(state.reshape(1,-1)))
      new_state,reward,done,info= env.step(action)
      state = new_state
  env.close()

def reward_plots(scores,avg_scores,n_episodes):
  plt.figure(figsize=(15,8))
  index_episodes = np.arange(1,n_episodes+1)
  plt.subplot(1,2,1)
  plt.plot(index_episodes,scores)
  plt.xlabel('Episodes')
  plt.ylabel('Reward')
  plt.subplot(1,2,2)
  plt.plot(index_episodes,avg_scores)
  plt.xlabel('Episodes')
  plt.ylabel('Rolling Reward')
  plt.show()

def store_scores(scores,avg_scores,path,name):
    data=pd.DataFrame({'scores':scores, 'avg_scores':avg_scores})
    data.to_csv(path+'/'+name,index=False)
    



