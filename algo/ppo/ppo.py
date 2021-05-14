from datetime import datetime
import os
from matplotlib.pyplot import step
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage import RolloutStorage
import numpy as np
import wandb

torch.autograd.set_detect_anomaly(True)


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True,
                 logger='tb'):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        if logger == 'tb':
            self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            self.tot_timesteps = 0
            self.tot_time = 0
            self.logger = logger

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def observe(self, actor_obs):
        self.actor_obs = actor_obs
        self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.cpu().numpy()

    def step(self, value_obs, rews, dones):
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, rews, dones, values,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update, auxilory_value=None):
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Add auxilary reward for each step
        # self.storage.add_auxiliary_reward(auxilory_value)

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step()
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables, width=80, pad=28):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()

        if self.logger == 'tb':
            self.writer.add_scalar('Loss/value_function', variables['mean_value_loss'], variables['it'])
            self.writer.add_scalar('Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), variables['it'])

        elif self.logger == 'wandb':
            wandb.log({'Loss/value_function': variables['mean_value_loss'], \
                    'Loss/surrogate': variables['mean_surrogate_loss'], \
                    'Policy/mean_noise_std': mean_std.item()})

    def extra_log(self, log_value, step, type=None):
        if self.logger == 'tb':
            if type == 'reward':
                self.writer.add_scalar('Reward/torque/mean', np.mean(log_value[:, 0]), step)
                # self.writer.add_scalar('Reward/torque/std', np.std(log_value[:, 0]), step)
                self.writer.add_scalar('Reward/velocity/mean', np.mean(log_value[:, 1]), step)
                # self.writer.add_scalar('Reward/velocity/std', np.std(log_value[:, 1]), step)
                self.writer.add_scalar('Reward/GRF_entropy/mean', np.mean(log_value[:, 2]), step)
                # self.writer.add_scalar('Reward/GRF_entropy/std', np.std(log_value[:, 2]), step)
                self.writer.add_scalar('Reward/impulse/mean', np.mean(log_value[:, 3]), step)
            elif type == 'action':
                # amplitude
                self.writer.add_scalar('CPG_tranform/FR_amplitude/mean', np.mean(log_value[:, 0]), step)
                self.writer.add_scalar('CPG_tranform/FR_amplitude/std', np.std(log_value[:, 0]), step)
                self.writer.add_scalar('CPG_tranform/FL_amplitude/mean', np.mean(log_value[:, 1]), step)
                self.writer.add_scalar('CPG_tranform/FL_amplitude/std', np.std(log_value[:, 1]), step)
                self.writer.add_scalar('CPG_tranform/RR_amplitude/mean', np.mean(log_value[:, 2]), step)
                self.writer.add_scalar('CPG_tranform/RR_amplitude/std', np.std(log_value[:, 2]), step)
                self.writer.add_scalar('CPG_tranform/RL_amplitude/mean', np.mean(log_value[:, 3]), step)
                self.writer.add_scalar('CPG_tranform/RL_amplitude/std', np.std(log_value[:, 3]), step)

                # shaft position
                self.writer.add_scalar('CPG_tranform/FR_shaft/mean', np.mean(log_value[:, 4]), step)
                self.writer.add_scalar('CPG_tranform/FR_shaft/std', np.std(log_value[:, 4]), step)
                self.writer.add_scalar('CPG_tranform/FL_shaft/mean', np.mean(log_value[:, 5]), step)
                self.writer.add_scalar('CPG_tranform/FL_shaft/std', np.std(log_value[:, 5]), step)
                self.writer.add_scalar('CPG_tranform/RR_shaft/mean', np.mean(log_value[:, 6]), step)
                self.writer.add_scalar('CPG_tranform/RR_shaft/std', np.std(log_value[:, 6]), step)
                self.writer.add_scalar('CPG_tranform/RL_shaft/mean', np.mean(log_value[:, 7]), step)
                self.writer.add_scalar('CPG_tranform/RL_shaft/std', np.std(log_value[:, 7]), step)


        elif self.logger == 'wandb':
            if type == 'reward':
                wandb.log({'Reward/torque/mean': np.mean(log_value[:, 0]), 'Reward/torque/std': np.std(log_value[:, 0]), \
                        'Reward/velocity/mean': np.mean(log_value[:, 1]), 'Reward/velocity/std': np.std(log_value[:, 1]), \
                        'Reward/GRF_entropy/mean': np.mean(log_value[:, 2]), 'Reward/GRF_entropy/std': np.std(log_value[:, 2])}, \
                            step=step)
            elif type == 'action':
                # amplitude
                wandb.log({'CPG_tranform/FR_amplitude/mean': np.mean(log_value[:, 0]), \
                        'CPG_tranform/FR_amplitude/std': np.std(log_value[:, 0]), \
                        'CPG_tranform/FL_amplitude/mean': np.mean(log_value[:, 1]), \
                        'CPG_tranform/FL_amplitude/std': np.std(log_value[:, 1]), 
                        'CPG_tranform/RR_amplitude/mean': np.mean(log_value[:, 2]), \
                        'CPG_tranform/RR_amplitude/std': np.std(log_value[:, 2]), \
                        'CPG_tranform/RL_amplitude/mean': np.mean(log_value[:, 3]), \
                        'CPG_tranform/RL_amplitude/std': np.std(log_value[:, 3])}, step=step)

                # shaft position
                wandb.log({'CPG_tranform/FR_shaft/mean': np.mean(log_value[:, 4]), \
                        'CPG_tranform/FR_shaft/std': np.std(log_value[:, 4]), \
                        'CPG_tranform/FL_shaft/mean': np.mean(log_value[:, 5]), \
                        'CPG_tranform/FL_shaft/std': np.std(log_value[:, 5]), 
                        'CPG_tranform/RR_shaft/mean': np.mean(log_value[:, 6]), \
                        'CPG_tranform/RR_shaft/std': np.std(log_value[:, 6]), \
                        'CPG_tranform/RL_shaft/mean': np.mean(log_value[:, 7]), \
                        'CPG_tranform/RL_shaft/std': np.std(log_value[:, 7])}, step=step)

    def _train_step(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()
