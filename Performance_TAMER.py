# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:27:45 2022

@author: Hang Yu
"""

from sac import SAC
from ddpg import DDPG
from buffer import ReplayBuffer

import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse
import rescalors
import csv

def save_score(scores,filename = "scores"):
    with open(filename + ".csv", 'w') as csvfile:
        kf_writer = csv.writer(csvfile, delimiter=' ',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for s in scores:
            kf_writer.writerow([s])
def read_score( filename="score", obs_size = 4, action_size = 1):
    scores = []
    with open(filename+ ".csv", 'rb') as file:
        reader = csv.reader(file,
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        for s in reader:
            scores.append(int(s[0]))
    return scores


Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)


env = gym.make('LunarLanderContinuous-v2')
config = {
    'dim_obs': env.observation_space.shape[0],  # Q network input
    'dim_action': env.action_space.shape[0],  # Q network output
    'dims_hidden_neurons': (256, 256),  # Q network hidden
    'lr': 3e-4,  # learning rate
    'tau': 0.005,  # target smoothing coefficient
    'discount': 0.99,  # discount factor
    'batch_size': 256,  # batch size for SAC only
    'min_batch': 64,  # minimal BS for DDPG only
    'max_batch': 512,  # maximal BS for DDPG only
    'replay_buffer_size': 1000000,
    'reward_scale': 5,
    'seed': 1,
}
def perform(filename, test_t = 50):
    agent = DDPG(config)
    # oracle = DDPG(config)
    # oracle.load_weights("oracle")
    buffer = ReplayBuffer(config)
    num_of_episodes = 900
    training_time = 1000
    agent.load_weights(filename)
    # for i in range(num_of_episodes):
    #     #print(i)
    #     agent.read_only(buffer, "z" + str(i) + "traj")
    # for i in range(training_time):
    #    # print(i)
    #     agent.update(buffer)
    # #agent.load_weights("pkls")
    #agent.save_model("pkls")
    
    steps = 0  # total number of steps
    scores = []
    buffer = ReplayBuffer(config)
    for i_episode in range(test_t):
        #if args.type == 'DDPG':
        #agent.Actor.process_reset()
        obs = env.reset()
        done = False
        t = 0  # time steps within each episode
        ret = 0.  # episodic return
        while done is False:
            #env.render()  # render to screen
    
            action, _ = agent.take_action(obs[None, :])  # take action
    
            next_obs, reward, done, info = env.step(np.array(action.view(-1).detach()))  # environment advance to next step
            
            #feedback = rescalors.rescale(oracle.get_mean(obs[None, :]), agent.get_mean(obs[None, :]))
    
            obs = next_obs
            # buffer.append_memory(obs=torch.from_numpy(obs).to(torch.double),  # put the transition to memory
            #                       action=action,
            #                       reward=torch.from_numpy(np.array([reward])).to(torch.double),
            #                       next_obs=torch.from_numpy(next_obs).to(torch.double),
            #                       done=done)
            # agent.update(buffer)  # agent learn
            #agent.save_one(obs, action, reward, next_obs, done, filename)
            
            
            t += 1
            steps += 1
            ret += reward  # update episodic return
            if done and i_episode%10 == 0:
                scores.append(ret)
                print("Episode {} finished after {} timesteps and reward is {}".format(i_episode, t+1, ret))
            #     train_writer.add_scalar('Performance/reward', ret, i_episode)  # plot
            # train_writer.add_scalar('Performance/episodic_return', ret, i_episode)  # plot
        
        
        #agent.read(buffer, filename+"_f")
    print(np.mean(scores))
    save_score(scores, "PS_n")
    env.close()
perform("TAMER")
perform("TAMER//B")
perform("TAMER//F")
perform("TAMER//C")