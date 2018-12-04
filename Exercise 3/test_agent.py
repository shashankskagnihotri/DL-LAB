from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

import argparse

from scipy.misc import toimage


from model import Model
from utils import *


def run_episode(env, agent, history_length, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0
    
    state = env.reset()



    '''
    _history = np.zeros((1, state.shape[0], state.shape[1], history_length))
    #_history = []
    
    #_history = np.zeros((1, state.shape[0], state.shape[1]))
    
    #print("state:", state)
    print("\n\n\nstate shape", state.shape, "\n\n\n\n")
    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        #    state = ...

        #state = rgb2gray(state)
        state = np.reshape(rgb2gray(state), (1, 96, 96, 1))
        #state = np.expand_dims(state, axis=3)


        
        _history[0,:,:] = state

        _history[0,:,:,0:history_length-1] = _history[0,:,:,1:]        
        #_history[0,:,:] = _history[0,:,:]
        #temp = _history.reshape([1,96,96])
        
        #_history.append(state)
        
        #print("_history.shape", _history.shape, "\n\n\n\n\n")
        #print("\n\n\nstate shape again", state.shape, "\n\n\n\n")
        #_history[0,:,:] = state
        #_history[0,:] = state
        #temp_history = _history[0:]
        #temp_history = np.zeros((state.shape[0], state.shape[1], history_length))
        
        #_history= np.transpose(np.array(_history))
        
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                for k in range(history_length):
                    temp_history[i][j][history_length] = _history[i][j][history_length]
        

                    
        
        
        #temp_history = np.array(temp_history[0:1])
        #print("\n\n\ntemp_history.shape\n\n\n", temp_history.shape) 

        
        
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        # a = ...


        '''
        #a = agent.sess.run(agent.output, feed_dict={agent.x_image:_history})[0]

    count = 1

    

    with agent.sess:
        while True:

            state = rgb2gray(state)

            #toimage(state).show()
            
            state= np.expand_dims(state, axis = 0)

            #state = state.reshape(state.shape[0], 96, 1)

            print("state.shape", state.shape)

            '''
            _history = np.zeros((1, state.shape[1], state.shape[1], history_length))
            _history[0:,:,:,:] = state
            '''

            state = reshaped_history(state, history_length)

            #print("state.shape", state.shape)

            #print("\n\nstate:", state)

            #prediction = agent.predict.eval(feed_dict={agent.x_image: state})[0]
                                            
        
            if count < 3 :
                prediction = 3
                count += 1
        
            else:
                prediction = agent.predict.eval(feed_dict={agent.x_image: state})[0]
                count += 1
            

            
                
            
            print("\n\nprediction:", prediction)
            #prediction[0] = 3
            a = id_to_action(prediction)
            '''
            if all(a == [0., 0. , 0.]):
                a = [0.0, 1.0, 0.0]

            '''

            print("a:", a)

            #a = id_to_action(agent.predict.eval(feed_dict ={agent.x_image: state}))
            
        
            next_state, r, done, info = env.step(a)   
            episode_reward += r       
            state = next_state
            #print("state:", state)
            #print("episode_reward:", episode_reward)
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

    cmdline_parser = argparse.ArgumentParser('exercise3_R_NR')


    cmdline_parser.add_argument(
        '-l', '--history_length', default=1,
        help='History Length', type=int)
    args, unknowns = cmdline_parser.parse_known_args()

    history_length = args.history_length
    
    # TODO: load agent
    agent = Model(history_length = history_length)
    agent.load("models/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, history_length, rendering=rendering)
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
