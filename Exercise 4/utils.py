import numpy as np



class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))

    def id_to_action(labels_id):
    	# convert id format to action format
        labels_action = np.zeros(3)
        labels_action[labels_id==0] = [0.0, 0.0, 0.0]	#STRAIGHT
        labels_action[labels_id==1] = [-1.0, 0.0, 0.0]  #LEFT
        labels_action[labels_id==2] = [1.0, 0.0, 0.0]	#RIGHT	
        labels_action[labels_id==3] = [0.0, 1.0, 0.0]	#ACCELERATE
        labels_action[labels_id==4] = [0.0, 0.0, 1.0]	#BRAKE
        return labels_action

    def rgb2gray(rgb):
        gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
        return gray.astype('float32') 

