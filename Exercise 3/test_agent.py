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
    
    #a = agent.sess.run(agent.output, feed_dict={agent.x_image:_history})[0]
    

    
    episode_reward = 0
    step = 0
    count = 1

    state = env.reset()
    while True:

        state = rgb2gray(state)

        #toimage(state).show()
        
        state= np.expand_dims(state, axis = 0)

        #state = state.reshape(state.shape[0], 96, 1)

        #print("state.shape", state.shape)

        '''
        _history = np.zeros((1, state.shape[1], state.shape[1], history_length))
        _history[0:,:,:,:] = state
        '''

        state = reshaped_history(state, history_length)

        #print("state.shape", state.shape)

        #print("\n\nstate:", state)

        prediction = agent.predict.eval(feed_dict={agent.x_image: state}, session = agent.sess)[0]
                                        
    
        '''
        if count < 3 :
            prediction = 3
            count += 1
    
        else:
            prediction = agent.predict.eval(feed_dict={agent.x_image: state}, session = agent.sess)[0]
            count += 1
        '''
        

        
            
        
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
    
    n_test_episodes = 10                 # number of episodes to test

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
