import pdb
import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs, PPO_type):
        if PPO_type == 'CPG':
            logits = torch.clamp(torch.relu(self.architecture.architecture(obs)), min=0.1, max=1.)
            actions, log_prob = self.distribution.sample(logits)

            ## For real action ##
            actions = torch.clamp(torch.relu(actions), min=0.1, max=1.)  # clipping amplitude (Architecture 5)
            # actions[:, 1:] = torch.tanh(actions[:, 1:])
            # actions[:, 2:] = torch.clamp(actions[:, 2:], min=-1, max=1)
            return actions.cpu().detach(), log_prob.cpu().detach()

        elif PPO_type == 'local' or PPO_type == None:
            logits = self.architecture.architecture(obs)
            logits[:, 0] = torch.relu(logits[:, 0])  # clipping amplitude (Architecture 5)
            # logits[:, 2:] = torch.clamp(logits[:, 2:], min=-1.3, max=1.3)  # clipping shaft & calf (Architecture 5)
            # logits[:, 1:] = torch.clamp(logits[:, 1:], min=-1.3, max=1.3)  # clipping shaft & calf (Architecture 5)
            actions, log_prob = self.distribution.sample(logits)
            # actions[:, :4] = F.relu(actions[:, :4])  # clipping amplitude (Architecture 4)

            ## For real action ##
            actions[:, 0] = torch.relu(actions[:, 0])  # clipping amplitude (Architecture 5)
            # actions[:, 2:] = torch.clamp(actions[:, 2:], min=-1.3, max=1.3)
            # actions[:, 1:] = torch.clamp(actions[:, 1:], min=-1.3, max=1.3)
            # actions[:, 1:] = torch.tanh(actions[:, 1:])
            # actions[:, 2:] = torch.clamp(actions[:, 2:], min=-1, max=1)
            return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions, PPO_type):
        if PPO_type == 'CPG':
            action_mean = torch.relu(self.architecture.architecture(obs))
            return self.distribution.evaluate(obs, action_mean, actions)
        elif PPO_type == 'local' or PPO_type == None:
            action_mean = self.architecture.architecture(obs)
            # action_mean_clipped = torch.cat((F.relu(action_mean[:, :4]), action_mean[:, 4:]), dim=1)  # clipping amplitude (Architecture 4)
            # action_mean_clipped = torch.cat((torch.relu(action_mean[:, 0]).unsqueeze(-1), torch.tanh(action_mean[:, 1:])), dim=1)  # clipping amplitude & shaft & calf (Architecture 5)
            action_mean_clipped = torch.cat((torch.relu(action_mean[:, 0]).unsqueeze(-1), action_mean[:, 1:]), dim=1)  # clipping amplitude & shaft & calf (Architecture 5)
            return self.distribution.evaluate(obs, action_mean_clipped, actions)  # clipping amplitude (Architecture 4, 5)
            # return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std, type=None):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        if type == 'CPG':
            self.std = init_std * torch.ones(dim) # fixed
        else:
            self.std = nn.Parameter(init_std * torch.ones(dim))  # trainable
        
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        try:
            self.past_distribution = Normal(logits, self.std.reshape(self.dim))
            distribution = self.past_distribution
        except:
            distribution = self.past_distribution
            print("logit error occurred! (Nan)")

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
