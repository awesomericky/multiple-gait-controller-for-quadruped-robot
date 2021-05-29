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
                 PPO_type=None,
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
        
        assert PPO_type in ['CPG', 'local', None], 'Unavailable PPO type'
        self.PPO_type = PPO_type

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
            if PPO_type == 'CPG':
                self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_CPG')
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            elif PPO_type == 'local':
                self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_Local')
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
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
        self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device), self.PPO_type)
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.cpu().numpy()

    def step(self, value_obs, rews, dones):
        if self.PPO_type == 'local' or self.PPO_type == None:
            values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        elif self.PPO_type == 'CPG':
            # values = torch.zeros(rews.shape).to(self.device)
            values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, rews, dones, values,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update, auxilory_value=None):
        if self.PPO_type == 'local' or self.PPO_type == None:
            last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        elif self.PPO_type == 'CPG':
            # last_values = torch.zeros((value_obs.shape[0], 1))
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
            if self.PPO_type == 'CPG':
                self.writer.add_scalar('CPG/Loss/value_function', variables['mean_value_loss'], variables['it'])
                self.writer.add_scalar('CPG/Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
                self.writer.add_scalar('CPG/Policy/mean_noise_std', mean_std.item(), variables['it'])
            elif self.PPO_type == 'local':
                self.writer.add_scalar('Local/Loss/value_function', variables['mean_value_loss'], variables['it'])
                self.writer.add_scalar('Local/Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
                self.writer.add_scalar('Local/Policy/mean_noise_std', mean_std.item(), variables['it'])
            else:
                self.writer.add_scalar('Loss/value_function', variables['mean_value_loss'], variables['it'])
                self.writer.add_scalar('Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
                self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), variables['it'])

        elif self.logger == 'wandb':
            wandb.log({'Loss/value_function': variables['mean_value_loss'], \
                    'Loss/surrogate': variables['mean_surrogate_loss'], \
                    'Policy/mean_noise_std': mean_std.item()})

    def extra_log(self, log_value, step, type=None):
        if self.logger == 'tb':
            if self.PPO_type == 'CPG':
                if type == 'reward':
                    self.writer.add_scalar('CPG/Reward/mean', np.mean(log_value), step)
                elif type == 'action':
                    self.writer.add_scalar('CPG/period/mean', np.mean(log_value), step)
                    # self.writer.add_scalar('CPG/FR_phase/mean', np.mean(log_value[:, 1]), step)
                    # self.writer.add_scalar('CPG/FL_phase/mean', np.mean(log_value[:, 2]), step)
                    # self.writer.add_scalar('CPG/RR_phase/mean', np.mean(log_value[:, 3]), step)
                    # self.writer.add_scalar('CPG/RL_phase/mean', np.mean(log_value[:, 4]), step)
                elif type == 'target_veloicty':
                    self.writer.add_scalar('CPG/target_velocity/mean', np.mean(log_value), step)
            elif self.PPO_type == 'local':
                if type == 'reward':
                    self.writer.add_scalar('Local/Reward/joint_torque/mean', np.mean(log_value[:, 0]), step)
                    self.writer.add_scalar('Local/Reward/linear_vel_error/mean', np.mean(log_value[:, 1]), step)
                    self.writer.add_scalar('Local/Reward/angular_vel_error/mean', np.mean(log_value[:, 2]), step)
                    self.writer.add_scalar('Local/Reward/foot_clearance/mean', np.mean(log_value[:, 3]), step)
                    self.writer.add_scalar('Local/Reward/foot_slip/mean', np.mean(log_value[:, 4]), step)
                    self.writer.add_scalar('Local/Reward/foot_z_vel/mean', np.mean(log_value[:, 5]), step)
                    self.writer.add_scalar('Local/Reward/joint_vel/mean', np.mean(log_value[:, 6]), step)
                    self.writer.add_scalar('Local/Reward/previous_action_smooth/mean', np.mean(log_value[:, 7]), step)
                    self.writer.add_scalar('Local/Reward/orientation/mean', np.mean(log_value[:, 8]), step)
                    self.writer.add_scalar('Local/Reward/height/mean', np.mean(log_value[:, 9]), step)
                    self.writer.add_scalar('Local/cost_scale/mean', np.mean(log_value[:, 10]), step)

                    # self.writer.add_scalar('Local/Reward/torque/mean', np.mean(log_value[:, 0]), step)
                    # self.writer.add_scalar('Reward/torque/std', np.std(log_value[:, 0]), step)
                    # self.writer.add_scalar('Local/Reward/velocity/mean', np.mean(log_value[:, 1]), step)
                    # self.writer.add_scalar('Reward/velocity/std', np.std(log_value[:, 1]), step)
                    # self.writer.add_scalar('Local/Reward/height/mean', np.mean(log_value[:, 2]), step)
                    # self.writer.add_scalar('Local/Reward/orientation/mean', np.mean(log_value[:, 3]), step)
                    # self.writer.add_scalar('CPG/Reward/GRF_entropy/mean', np.mean(log_value[:, 4]), step)
                    # self.writer.add_scalar('Reward/LegWorkEntropy/mean', np.mean(log_value[:, 4]), step)
                    # self.writer.add_scalar('Reward/uncontactPenalty/mean', np.mean(log_value[:, 5]), step)
                    # self.writer.add_scalar('Reward/GRF_entropy/mean', np.mean(log_value[:, 4]), step)
                    # self.writer.add_scalar('Reward/GRF_entropy/std', np.std(log_value[:, 2]), step)
                    # self.writer.add_scalar('Reward/impulse/mean', np.mean(log_value[:, 5]), step)
                elif type == 'action':
                    # Architecture 5
                    self.writer.add_scalar('Local/CPG_tranform/Thigh_amplitude/mean', np.mean(log_value[:, 0]), step)
                    self.writer.add_scalar('Local/CPG_tranform/Thigh_shaft/mean', np.mean(log_value[:, 1]), step)
                    self.writer.add_scalar('Local/CPG_tranform/FR_calf/mean', np.mean(log_value[:, 2]), step)
                    self.writer.add_scalar('Local/CPG_tranform/FL_calf/mean', np.mean(log_value[:, 3]), step)
                    self.writer.add_scalar('Local/CPG_tranform/RR_calf/mean', np.mean(log_value[:, 4]), step)
                    # self.writer.add_scalar('Local/CPG_tranform/RL_calf/mean', np.mean(log_value[:, 5]), step)
            else:
                if type == 'reward':
                    self.writer.add_scalar('Reward/torque/mean', np.mean(log_value[:, 0]), step)
                    # self.writer.add_scalar('Reward/torque/std', np.std(log_value[:, 0]), step)
                    self.writer.add_scalar('Reward/velocity/mean', np.mean(log_value[:, 1]), step)
                    # self.writer.add_scalar('Reward/velocity/std', np.std(log_value[:, 1]), step)
                    self.writer.add_scalar('Reward/height/mean', np.mean(log_value[:, 2]), step)
                    self.writer.add_scalar('Reward/orientation/mean', np.mean(log_value[:, 3]), step)
                    # self.writer.add_scalar('Reward/LegWorkEntropy/mean', np.mean(log_value[:, 4]), step)
                    # self.writer.add_scalar('Reward/uncontactPenalty/mean', np.mean(log_value[:, 5]), step)
                    # self.writer.add_scalar('Reward/GRF_entropy/mean', np.mean(log_value[:, 4]), step)
                    # self.writer.add_scalar('Reward/GRF_entropy/std', np.std(log_value[:, 2]), step)
                    # self.writer.add_scalar('Reward/impulse/mean', np.mean(log_value[:, 5]), step)
                elif type == 'action':
                    # Architecture 5
                    self.writer.add_scalar('CPG_tranform/Thigh_amplitude/mean', np.mean(log_value[:, 0]), step)
                    self.writer.add_scalar('CPG_tranform/Thigh_shaft/mean', np.mean(log_value[:, 1]), step)
                    self.writer.add_scalar('CPG_tranform/FR_calf/mean', np.mean(log_value[:, 2]), step)
                    self.writer.add_scalar('CPG_tranform/FL_calf/mean', np.mean(log_value[:, 3]), step)
                    self.writer.add_scalar('CPG_tranform/RR_calf/mean', np.mean(log_value[:, 4]), step)
                    self.writer.add_scalar('CPG_tranform/RL_calf/mean', np.mean(log_value[:, 5]), step)


                """
                # Architecture 4
                # amplitude
                self.writer.add_scalar('CPG_tranform/FR_amplitude/mean', np.mean(log_value[:, 0]), step)
                # self.writer.add_scalar('CPG_tranform/FR_amplitude/std', np.std(log_value[:, 0]), step)
                self.writer.add_scalar('CPG_tranform/FL_amplitude/mean', np.mean(log_value[:, 1]), step)
                # self.writer.add_scalar('CPG_tranform/FL_amplitude/std', np.std(log_value[:, 1]), step)
                self.writer.add_scalar('CPG_tranform/RR_amplitude/mean', np.mean(log_value[:, 2]), step)
                # self.writer.add_scalar('CPG_tranform/RR_amplitude/std', np.std(log_value[:, 2]), step)
                self.writer.add_scalar('CPG_tranform/RL_amplitude/mean', np.mean(log_value[:, 3]), step)
                # self.writer.add_scalar('CPG_tranform/RL_amplitude/std', np.std(log_value[:, 3]), step)

                # shaft position
                self.writer.add_scalar('CPG_tranform/FR_shaft/mean', np.mean(log_value[:, 4]), step)
                # self.writer.add_scalar('CPG_tranform/FR_shaft/std', np.std(log_value[:, 4]), step)
                self.writer.add_scalar('CPG_tranform/FL_shaft/mean', np.mean(log_value[:, 5]), step)
                # self.writer.add_scalar('CPG_tranform/FL_shaft/std', np.std(log_value[:, 5]), step)
                self.writer.add_scalar('CPG_tranform/RR_shaft/mean', np.mean(log_value[:, 6]), step)
                # self.writer.add_scalar('CPG_tranform/RR_shaft/std', np.std(log_value[:, 6]), step)
                self.writer.add_scalar('CPG_tranform/RL_shaft/mean', np.mean(log_value[:, 7]), step)
                # self.writer.add_scalar('CPG_tranform/RL_shaft/std', np.std(log_value[:, 7]), step)
                """


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

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch, self.PPO_type)
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
