import torch
import torch.nn as nn
import random
from models import QNetwork, Actor
import csv
import numpy as np
import rescalors
Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)
device = torch.device("cuda" if True else "cpu")
#model = MyRNN().to(device)
class DDPG():

    def __init__(self, config):

        torch.manual_seed(config['seed'])
        random.seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.tau = config['tau']  # target smoothing coefficient
        self.discount = config['discount']  # discount factor
        self.min_batch = config['min_batch']  # min of random batch size
        self.max_batch = config['max_batch']  # max of random batch size
        self.reward_scale = config['reward_scale']  # reward scale

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.Q = QNetwork(dim_obs=self.dim_obs,
                          dim_action=self.dim_action,
                          dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q_tar = QNetwork(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.Actor = Actor(dim_obs=self.dim_obs,
                                   dim_action=self.dim_action,
                                   dims_hidden_neurons=self.dims_hidden_neurons)
        self.Actor_tar = Actor(dim_obs=self.dim_obs,
                                       dim_action=self.dim_action,
                                       dims_hidden_neurons=self.dims_hidden_neurons)

        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.optimizer_Actor = torch.optim.Adam(self.Actor.parameters(), lr=self.lr)
        self.training_step = 0
        self.loss_func = nn.MSELoss()

        self.Q_tar.load_state_dict(self.Q.state_dict())
        self.Actor_tar.load_state_dict(self.Actor.state_dict())

    def update(self, buffer):

        t = buffer.sample(random.randint(self.min_batch, self.max_batch))

        done = t.done
        s = t.obs
        a = t.action
        sp = t.next_obs
        r = t.reward

        self.training_step += 1

        self.update_Q(s, a, sp, r, done)
        self.update_Actor(s)

    def update_Q(self, s, a, sp, r, done):
        ap = self.Actor_tar(sp)
        y = self.reward_scale * r + ~done * self.discount * self.Q_tar(sp, ap)
        q = self.Q(s, a)
        loss = self.loss_func(y, q)
        self.optimizer_Q.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer_Q.step()

        state_dict = self.Q.state_dict().copy()
        state_dict_ = self.Q_tar.state_dict().copy()
        for n, p in state_dict.items():
            state_dict_[n] = self.tau * p + (1-self.tau) * state_dict_[n]
        self.Q_tar.load_state_dict(state_dict_)

    def update_Actor(self, s):
        action = self.Actor(s).clone()
        loss = torch.mean(-self.Q(s, action))
        self.optimizer_Actor.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer_Actor.step()

        state_dict = self.Actor.state_dict().copy()
        state_dict_ = self.Actor_tar.state_dict().copy()
        for n, p in state_dict.items():
            state_dict_[n] = self.tau * p + (1-self.tau) * state_dict_[n]
        self.Actor_tar.load_state_dict(state_dict_)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.double)
        return self.Actor.sample_normal(state), None
    def get_mean(self, state):
        state = torch.tensor(state, dtype=torch.double)
        return self.Actor.get_mean(state)
    
    def load_weights(self,  output):
        if output is None: return

        self.Actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.Q.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )
        self.Q_tar.load_state_dict(self.Q.state_dict())
        self.Actor_tar.load_state_dict(self.Actor.state_dict())


    def save_model(self,output):
        torch.save(
            self.Actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.Q.state_dict(),
            '{}/critic.pkl'.format(output)
        )
    def create_one(self,  filename="buffer"):
        with open(filename + ".csv", 'wb') as csvfile:
            pass
    def save_one(self, state, action, reward, next_state, done, filename="buffer"):
        with open(filename + ".csv", 'a') as csvfile:
            kf_writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if done:
                done = 1
            else:
                done =0
           # print (state.tolist() , action.numpy()[0].tolist() , [reward] , next_state.tolist() ,  [done])
            temp = state.tolist() + action.numpy()[0].tolist() + [reward] + next_state.tolist() +  [done]

                #kf_writer.writerow( [ self.obs[i].numpy(), self.action[i].numpy(), self.reward[i].numpy() ,self.next_obs[i].numpy(),  self.done[i] ] )
            kf_writer.writerow( temp)
                #print("done for once")
    def read(self, buffer, filename="buffer", obs_size = 8, action_size = 2):
        with open(filename+ ".csv") as file:
            reader = csv.reader(file, 
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        
        # storing all the rows in an output list
            output = []
            
            cnt = 0
            for row_ in reader:
                cnt += 1 
                self.Actor.process_reset()
                if cnt % 2 == 0:
                    pass
                else:
                    #row = map(float, row)
                    #print(row, type(row))
                    row =  [float(i) for i in row_]
                    #print(row, len(row)) #row[len(row)])
                    if row[len(row) - 1] == 1:
                        done_ = True
                    else:
                        done_ = False
                    obs_ = np.array(row[:obs_size])
                    #print(obs_, obs_[0])
                    action_ = np.array([row[obs_size: obs_size+action_size]])
                    reward_ = np.array(row[obs_size+action_size])
                    next_obs_ = np.array(row[obs_size+action_size+1: 1 + obs_size+action_size + obs_size ])
                    #done_ = row[-1]
                    #self.add(obs_, int(action_[0]), reward_, next_obs_, done_)
                    #print( action_)
                    buffer.append_memory(obs=torch.from_numpy(obs_).to(torch.double),  # put the transition to memory
                                 action=torch.from_numpy(action_).to(torch.double),
                                 reward=torch.from_numpy(np.array([reward_])).to(torch.double),
                                 next_obs=torch.from_numpy(next_obs_).to(torch.double),
                                 done=done_)
                    self.update(buffer)
                #print(type(obs_[0]))
                #output.append(row[:])
        #print(output)
        return output
    def read_only(self, buffer, filename="buffer", obs_size = 8, action_size = 2):
        with open(filename+ ".csv") as file:
            reader = csv.reader(file, 
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        
        # storing all the rows in an output list
            output = []
            
            cnt = 0
            for row_ in reader:
                cnt += 1 
                #self.Actor.process_reset()
                if cnt % 2 == 0:
                    pass
                else:
                    #row = map(float, row)
                    #print(row, type(row))
                    row =  [float(i) for i in row_]
                    #print(row, len(row)) #row[len(row)])
                    if row[len(row) - 1] == 1:
                        done_ = True
                    else:
                        done_ = False
                    obs_ = np.array(row[:obs_size])
                    #print(obs_, obs_[0])
                    action_ = np.array([row[obs_size: obs_size+action_size]])
                    reward_ = np.array(row[obs_size+action_size])
                    next_obs_ = np.array(row[obs_size+action_size+1: 1 + obs_size+action_size + obs_size ])
                    #done_ = row[-1]
                    #self.add(obs_, int(action_[0]), reward_, next_obs_, done_)
                    #print( action_)
                    buffer.append_memory(obs=torch.from_numpy(obs_).to(torch.double),  # put the transition to memory
                                 action=torch.from_numpy(action_).to(torch.double),
                                 reward=torch.from_numpy(np.array([reward_])).to(torch.double),
                                 next_obs=torch.from_numpy(next_obs_).to(torch.double),
                                 done=done_)
                    #self.update(buffer)
                #print(type(obs_[0]))
                #output.append(row[:])
        #print(output)
        return output

