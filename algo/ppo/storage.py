import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape).to(self.device)
        self.actor_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape).to(self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape).to(self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1).byte().to(self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

        self.reward_normalize_type = 1
        assert self.reward_normalize_type in [0, 1, 2, 3], f"Unavailable reward normalize method {self.reward_normalize_type}"

    def add_transitions(self, actor_obs, critic_obs, actions, rewards, dones, values, actions_log_prob):
        """
        Function saving (s, a, r(s, a), v(s)) pairs

        Dimension: 
            - actor_obs : (n_env, 34)
            - critic_obs : (n_env, 26)
            - actions : (n_env, 8)
            - rewards : (n_env, )
            - dones : (n_env, )
            - values : (n_env, 1)
            - actions_log_prob : (n_env)
        
        cf) 34 = obs_dim + CPG_signal_dim
            26 = obs_dim
            8 = action_dim (= n_joint)
        """
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step].copy_(torch.from_numpy(critic_obs).to(self.device))
        self.actor_obs[self.step].copy_(torch.from_numpy(actor_obs).to(self.device))
        self.actions[self.step].copy_(actions.to(self.device))
        self.rewards[self.step].copy_(torch.from_numpy(rewards).view(-1, 1).to(self.device))
        self.dones[self.step].copy_(torch.from_numpy(dones).view(-1, 1).to(self.device))
        self.values[self.step].copy_(values.to(self.device))
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1).to(self.device))
        self.step += 1

    def clear(self):
        self.step = 0
    
    def add_auxiliary_reward(self, auxiliary_reward):
        # self.dones = torch.zeros(num_transitions_per_env, num_envs, 1).byte().to(self.device)
        not_dones = 1.0 - self.dones.float()
        n_states_before_done = torch.maximum(torch.sum(not_dones, axis=0), torch.tensor(1).to(self.device))
        each_step_auxiliary_loss = torch.from_numpy(auxiliary_reward).to(self.device) / n_states_before_done  # (n_envs, 1)
        self.rewards += not_dones * each_step_auxiliary_loss.unsqueeze(0)

    def reward_normalize(self):
        if self.reward_normalize_type == 0:
            pass
        elif self.reward_normalize_type == 1:
            # normalize 1 (normalize total)
            self.rewards -= torch.mean(self.rewards)
            self.rewards /= torch.std(self.rewards + 1e-6)
        elif self.reward_normalize_type == 2:
            # normalize 2 (normalize for each env data) ==> layer normalization
            self.rewards -= torch.mean(self.rewards, axis=0).unsqueeze(0)
            self.rewards /= torch.std(self.rewards, axis=0).unsqueeze(0)
        elif self.reward_normalize_type == 3:
            # normalize 3 (normalize for each step) ==> batch normalization
            self.rewards -= torch.mean(self.rewards, axis=1).unsqueeze(1)
            self.rewards /= torch.std(self.rewards, axis=1).unsqueeze(1)

    def compute_returns(self, last_values, gamma, lam):
        self.reward_normalize()  # normalize reward

        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs.view(-1, *self.actor_obs.size()[2:])[indices]
            critic_obs_batch = self.critic_obs.view(-1, *self.critic_obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            values_batch = self.values.view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob.view(-1, 1)[indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]
            yield actor_obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.actor_obs.view(-1, *self.actor_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.critic_obs.view(-1, *self.critic_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.advantages.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_log_prob.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
