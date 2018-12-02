from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length = 1):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    state_history = np.zeros((1, state.shape[0], state.shape[1], history_length))
    #state_history = np.zeros((1, state.shape[0], state.shape[1]))
    
    #print("state:", state)
    print("\n\n\nstate shape", state.shape, "\n\n\n\n")
    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        #    state = ...

        state = rgb2gray(state)
        state = np.expand_dims(state, axis=3)
        state_history[0,:,:,0:history_length-1] = state_history[0,:,:,1:]        
        #state_history[0,:,:] = state_history[0,:,:]
        #temp = state_history.reshape([1,96,96])
        print("state_history.shape", state_history.shape, "\n\n\n\n\n")
        print("\n\n\nstate shape again", state.shape, "\n\n\n\n")
        state_history[0,:,:] = state
        #state_history[0,:] = state

        
        
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        # a = ...
        a = agent.sess.run(agent.output, feed_dict={agent.x_image:state_history})[0]
        
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    history_length = 1
    
    # TODO: load agent
    agent = Model(history_length = history_length)
    agent.load("models/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
