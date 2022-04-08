import torch
import torch.nn as nn
import random
from models import QNetwork, Actor

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
                          dims_hidden_neurons=self.dims_hidden_neurons).to(device)
        self.Q_tar = QNetwork(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons).to(device)
        self.Actor = Actor(dim_obs=self.dim_obs,
                                   dim_action=self.dim_action,
                                   dims_hidden_neurons=self.dims_hidden_neurons).to(device)
        self.Actor_tar = Actor(dim_obs=self.dim_obs,
                                       dim_action=self.dim_action,
                                       dims_hidden_neurons=self.dims_hidden_neurons).to(device)

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
        return self.Actor.get_mean(state)
    
    def load_weights(self, output):
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


