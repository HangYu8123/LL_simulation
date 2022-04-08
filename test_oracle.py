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
import Classifiers

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
oracle = DDPG(config)
agent = DDPG(config)
oracle.load_weights("oracle")
agent.load_weights("half_train")
buffer = ReplayBuffer(config)
# train_writer = SummaryWriter(log_dir='tensorboard/{}_{date:%Y-%m-%d_%H:%M:%S}'.format(
#                              args.type, date=datetime.datetime.now()))
classifiers = Classifiers.Classifiers()
steps = 0  # total number of steps
cnt = 0
max_rwd = -10
min_rwd = 10
for i_episode in range(1):
    #if args.type == 'DDPG':
    agent.Actor.process_reset()
    obs = env.reset()
    done = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return
    feedbacks =0
    
    while done is False:
        #env.render()  # render to screen

        action, _ = oracle.take_action(obs[None, :])  # take action

        next_obs, reward, done, info = env.step(np.array(action.view(-1).detach()))  # environment advance to next step
        feedback = rescalors.rescale(oracle.get_mean(obs[None, :]), agent.get_mean(obs[None, :]))
        # if max_rwd < reward and reward < 5:
        #     max_rwd = reward
        # if min_rwd > reward and reward > -5:
        #     min_rwd = reward
        # feedback = rescalors.rescale(reward)
        #print(reward, action)
        f = rescalors.dynamic_feedback(feedback, 1, 0, 0)
        print(feedback, f)
        if f > 0 and feedback - 5 < 0:
            cnt += 1
            #print(feedback, feedback-5 , f)
        if f < 0 and feedback - 5 > 0:
            cnt += 1
            #print(feedback, feedback-5 , f)
        obs = next_obs
        

        t += 1
        steps += 1
        feedbacks += feedback
        ret += reward  # update episodic return
        if done:
            print("Episode {} finished after {} timesteps and reward is {}, feedback avg is {}".format(i_episode, t+1, ret, feedbacks/(t+1)))
            
        #     train_writer.add_scalar('Performance/reward', ret, i_episode)  # plot
        # train_writer.add_scalar('Performance/episodic_return', ret, i_episode)  # plot
    #agent.save_model("pkls")
    #agent.load_weights("pkls")
print(cnt, steps, cnt/(steps))
print(classifiers.steady(0))
print(max_rwd, min_rwd)
env.close()
#train_writer.close()
