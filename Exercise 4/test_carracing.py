from __future__ import print_function

import gym
from dqn.agent import DQNAgent
from train_carracingimport run_episode
from dqn.networks import *
import numpy as np
import argparse

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    #history_length =  3

    #TODO: Define networks and load agent
    # ....
    state_dim = (96, 96)
    cmdline_parser = argparse.ArgumentParser('Exercise 4')


    cmdline_parser.add_argument(
        '-l', '--history_length', default=3,
        help='Number of states to be considered in the history for prediction', type=int)

    cmdline_parser.add_argument(
        '-n', '--num_actions', default=5,
        help='Number of Actions', type=int)

    
    args, unknowns = cmdline_parser.parse_known_args()

    

    history_length = args.history_length
    num_actions = args.num_actions

    Q = CNN(state_dim, num_actions, history_length, hidden=256, lr=1e-3)
    Q_target = CNNTargetNetwork(state_dim, num_actions, history_length, hidden=256, lr=1e-3)
    agent = DQNAgent(Q, Q_target, num_actions, discount_factor=0.99, batch_size=64, epsilon=0.05)
    agent.load("./models_carracing/dqn_agent.ckpt"

    

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

