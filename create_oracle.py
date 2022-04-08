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
print(env.observation_space.shape[0])
agent = DDPG(config)
# oracle = DDPG(config)
# oracle.load_weightsz`("oracle")
#agent.load_weights("half_train")
buffer = ReplayBuffer(config)
# train_writer = SummaryWriter(log_dir='tensorboard/{}_{date:%Y-%m-%d_%H:%M:%S}'.format(
#                              args.type, date=datetime.datetime.now()))

steps = 0  # total number of steps
for i_episode in range(1000):
    #if args.type == 'DDPG':
    agent.Actor.process_reset()
    obs = env.reset()
    done = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return
    #filename  = "z" + str(i_episode ) + "traj"
   # agent.create_one(filename)
    #agent.create_one(filename+"_f")
    while done is False:
        #env.render()  # render to screen

        action, _ = agent.take_action(obs[None, :])  # take action

        next_obs, reward, done, info = env.step(np.array(action.view(-1).detach()))  # environment advance to next step

        buffer.append_memory(obs=torch.from_numpy(obs).to(torch.double),  # put the transition to memory
                             action=action,
                             reward=torch.from_numpy(np.array([reward])).to(torch.double),
                             next_obs=torch.from_numpy(next_obs).to(torch.double),
                             done=done)
        obs = next_obs

        agent.update(buffer)  # agent learn
        # agent.save_one(obs, action, reward, next_obs, done, filename)
        # feedback = rescalors.rescale(oracle.get_mean(obs[None, :]), agent.get_mean(obs[None, :]))
        #agent.save_one(obs, action, reward, next_obs, done, filename)
        t += 1
        steps += 1
        ret += reward  # update episodic return
        if done:
            print("Episode {} finished after {} timesteps and reward is {}".format(i_episode, t+1, ret))
        #     train_writer.add_scalar('Performance/reward', ret, i_episode)  # plot
        # train_writer.add_scalar('Performance/episodic_return', ret, i_episode)  # plot
    agent.save_model("oracle")
    #agent.load_weights("pkls")
    #agent.read(buffer, filename+"_f")
env.close()
#train_writer.close()
