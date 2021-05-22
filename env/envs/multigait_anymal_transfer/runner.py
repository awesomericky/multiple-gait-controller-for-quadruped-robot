from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import multigait_anymal_transfer
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.utils import joint_angle_plotting, contact_plotting
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import wandb
import matplotlib.pyplot as plt
import pdb

"""
# TODO

1. Check result for pace, bound (experiment with excel row 52, 58)
2. Think way for hierarchical RL w/ reward
(3. Test w/ small std?)

"""


def sin(x, a, b, c, d):
    return a * np.sin(b*x + c) + d

# task specification
task_name = "multigait_anymal_transfer"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument(
    '-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(multigait_anymal_transfer.RaisimGymEnv(home_path + "/rsc",
             dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
act_dim = env.num_acts - 2  # 8 - 2 (w/ HAA joints fixed)
CPG_signal_dim = 5

# Training
n_CPG_steps = math.floor(cfg['environment']['max_time'] /
                         cfg['environment']['CPG_control_dt'])
n_steps = math.floor(cfg['environment']['CPG_control_dt'] /
                     cfg['environment']['control_dt'])
total_CPG_steps = n_CPG_steps * env.num_envs
total_steps = n_steps * env.num_envs

min_vel = cfg['environment']['velocity']['min']
max_vel = cfg['environment']['velocity']['max']

avg_rewards = []

CPG_actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['CPG_policy_net'], nn.LeakyReLU, CPG_signal_dim + 1, CPG_signal_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(
                             CPG_signal_dim, 0.1),  # 1.0
                         device)
CPG_critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['CPG_value_net'], nn.LeakyReLU, CPG_signal_dim + 1, 1),
                           device)

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + 2, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(
                             act_dim, 1.0),  # 1.0
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + 2, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

# logging
if cfg['logger'] == 'tb':
    tensorboard_launcher(saver.data_dir+"/..")   # press refresh (F5) after the first ppo update
elif cfg['logger'] == 'wandb':
    wandb.init(project='multigait', name='experiment 1', config=dict(cfg))

CPG_ppo = PPO.PPO(actor=CPG_actor,
                  critic=CPG_critic,
                  num_envs=cfg['environment']['num_envs'],
                  num_transitions_per_env=n_CPG_steps,
                  num_learning_epochs=4,
                  gamma=0.996,
                  lam=0.95,
                  num_mini_batches=4,
                  PPO_type='CPG',
                  device=device,
                  log_dir=saver.data_dir,
                  shuffle_batch=False,
                  logger=cfg['logger']
                  )

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              PPO_type='local',
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              logger=cfg['logger']
              )

if mode == 'retrain':
    load_param(weight_path, env, CPG_actor, CPG_critic, CPG_ppo.optimizer, saver.data_dir)
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

"""
[Joint order]

1) Laikago
    - FR_thigh_joint
    - FR_calf_joint
    - FL_thigh_joint
    - FL_calf_joint
    - RR_thigh_joint
    - RR_calf_joint
    - RL_thigh_joint
    - RL_calf_joint

2) Anymal
    - LF_HFE
    - LF_KFE
    - RF_HFE                
    - RF_KFE
    - LH_HFE
    - LH_KFE
    - RH_HFE
    - RH_KFE


[Target signal order]

1) Laikago
    - FR_thigh_joint
    - FL_thigh_joint
    - RR_thigh_joint
    - RL_thigh_joint

2) Anymal
    - LF_HFE
    - RF_HFE
    - LH_HFE
    - RH_HFE

"""

t_range = (np.arange(n_steps) * cfg['environment']['control_dt'])[np.newaxis, :]
CPG_signal = np.zeros((env.num_envs, 4, n_steps))
env_action = np.zeros((cfg['environment']['num_envs'], 8), dtype=np.float32)

for update in range(1000000):
    start = time.time()
    env.reset()
    average_ll_performance_total = []
    average_dones_total = []

    """
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'CPG_actor_architecture_state_dict': CPG_actor.architecture.state_dict(),
            'CPG_actor_distribution_state_dict': CPG_actor.distribution.state_dict(),
            'CPG_critic_architecture_state_dict': CPG_critic.architecture.state_dict(),
            'CPG_optimizer_state_dict': CPG_ppo.optimizer.state_dict(),
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        CPG_loaded_graph = ppo_module.MLP(cfg['architecture']['CPG_policy_net'], nn.LeakyReLU, CPG_signal_dim + 1, CPG_signal_dim)
        CPG_loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['CPG_actor_architecture_state_dict'])
        local_loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + 2, act_dim)
        local_loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        contact_log = np.zeros((4, n_steps*4), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
        previous_CPG = np.zeros((env.num_envs, CPG_signal_dim), dtype=np.float32)

        for CPG_step in range(4):

            normalized_velocity = np.broadcast_to(np.random.uniform(low = 0, high=1, size=1)[:, np.newaxis], (env.num_envs, 1)).astype(np.float32)
            velocity = normalized_velocity * (max_vel - min_vel) + min_vel
            CPG_obs = np.concatenate([normalized_velocity, previous_CPG], axis=1)
            env.set_target_velocity(velocity)
            current_CPG = CPG_loaded_graph.architecture(torch.from_numpy(CPG_obs)).cpu().detach().numpy()
            previous_CPG = current_CPG

            # generate target signal
            period = current_CPG[:, 0][:, np.newaxis]  # [s]
            period_param = 2 * np.pi / period
            for i in range(4):
                # 0: FR, 1: FL, 2: RR, 3: RL
                CPG_signal[:, i, :] = sin(t_range, 1, period_param, np.pi * current_CPG[:, i+1][:, np.newaxis], 0.0)

            for step in range(n_steps):
                frame_start = time.time()
                obs = env.observe(False)
                phase = ((step * cfg['environment']['control_dt']) / period) - ((step * cfg['environment']['control_dt']) / period).astype(int)
                obs = np.concatenate([obs, normalized_velocity, phase], axis=1, dtype=np.float32)
                action_ll = local_loaded_graph.architecture(torch.from_numpy(obs))
                action_ll[:, 0] = torch.relu(action_ll[:, 0])
                action_ll = action_ll.cpu().detach().numpy()
                env_action[:, [0, 2, 4, 6]] = action_ll[:, 0][:, np.newaxis] * CPG_signal[:, :, step] + action_ll[:, 1][:, np.newaxis]
                env_action[:, [1, 3, 5, 7]] = action_ll[:, 2:]
                reward_ll, dones = env.step(env_action)
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)
                
                # contact logging
                env.contact_logging()
                contact_log[:, CPG_step * n_steps + step] = env.contact_log[0, :]
        
        # save & plot contact log
        np.savez_compressed(f'contact_plot/contact_{update}.npz', contact=contact_log)
        contact_plotting(update, contact_log)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

        """
    previous_CPG = np.zeros((env.num_envs, CPG_signal_dim), dtype=np.float32)

    for CPG_step in range(n_CPG_steps):
        normalized_velocity = np.random.uniform(low = 0, high=1, size=env.num_envs)[:, np.newaxis].astype(np.float32)
        velocity = normalized_velocity * (max_vel - min_vel) + min_vel
        CPG_obs = np.concatenate([normalized_velocity, previous_CPG], axis=1)
        env.set_target_velocity(velocity)
        current_CPG = CPG_ppo.observe(CPG_obs)
        previous_CPG = current_CPG
        
        # generate target signal
        period = current_CPG[:, 0][:, np.newaxis]  # [s]
        period_param = 2 * np.pi / period
        for i in range(4):
            # 0: FR, 1: FL, 2: RR, 3: RL
            CPG_signal[:, i, :] = sin(t_range, 1, period_param, np.pi * current_CPG[:, i+1][:, np.newaxis], 0.0)

        CPG_rewards = np.zeros((env.num_envs, 1))
    
        FR_thigh_joint_history = np.zeros(n_steps)
        FR_calf_joint_history = np.zeros(n_steps)
        FL_thigh_joint_history = np.zeros(n_steps)
        FL_calf_joint_history = np.zeros(n_steps)
        RR_thigh_joint_history = np.zeros(n_steps)
        RR_calf_joint_history = np.zeros(n_steps)
        RL_thigh_joint_history = np.zeros(n_steps)
        RL_calf_joint_history = np.zeros(n_steps)

        reward_ll_sum = 0
        done_sum = 0
        average_dones = 0.

        # actual training
        for step in range(n_steps):
            obs, non_obs = env.observe_logging()
            phase = ((step * cfg['environment']['control_dt']) / period) - ((step * cfg['environment']['control_dt']) / period).astype(int)
            obs = np.concatenate([obs, normalized_velocity, phase], axis=1, dtype=np.float32)
            action = ppo.observe(obs)

            if update % 50 == 0:
                FR_thigh_joint_history[step] = non_obs[0, 4]
                FR_calf_joint_history[step] = non_obs[0, 5]
                FL_thigh_joint_history[step] = non_obs[0, 6]
                FL_calf_joint_history[step] = non_obs[0, 7]
                RR_thigh_joint_history[step] = non_obs[0, 8]
                RR_calf_joint_history[step] = non_obs[0, 9]
                RL_thigh_joint_history[step] = non_obs[0, 10]
                RL_calf_joint_history[step] = non_obs[0,11]

            # Architecture 5
            env_action[:, [0, 2, 4, 6]] = action[:, 0][:, np.newaxis] * CPG_signal[:, :, step] + action[:, 1][:, np.newaxis]
            env_action[:, [1, 3, 5, 7]] = action[:, 2:]

            reward, dones = env.step(env_action)
            env.get_CPG_reward()
            temp_CPG_rewards = env._CPG_reward
            CPG_rewards += (temp_CPG_rewards * (1 - dones))[:, np.newaxis]
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + sum(dones)
            reward_ll_sum = reward_ll_sum + sum(reward)

            if (update % 5 == 0) and (step + 1) % 100 == 0 and 1 < cfg['environment']['num_envs']:
                env.reward_logging()
                ppo.extra_log(env.reward_log, (update // 5) * n_steps + step, type='reward')
            
            if update % 10 == 0:
                ppo.extra_log(action, (update // 10) * n_steps + step, type='action')
            
        if update % 100 == 0:
            joint_angle_plotting(update, np.squeeze(t_range), CPG_signal[0],\
                                 FR_thigh_joint_history, FL_thigh_joint_history, RR_thigh_joint_history, RL_thigh_joint_history,\
                                 FR_calf_joint_history, FL_calf_joint_history, RR_calf_joint_history, RL_calf_joint_history)
        
        # take st step to get value obs
        obs, _ = env.observe_logging()
        phase = ((n_steps * cfg['environment']['control_dt']) / period) - ((n_steps * cfg['environment']['control_dt']) / period).astype(int)
        obs = np.concatenate([obs, normalized_velocity, phase], axis=1, dtype=np.float32)
        # ppo.update(actor_obs=obs_and_target, value_obs=obs, log_this_iteration=update % 10 == 0, update=update, auxilory_value=sin_fitting_loss[:, np.newaxis])
        ppo.update(actor_obs=obs, value_obs=obs,
                log_this_iteration=update % 10 == 0, update=update)
        actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))
        
        CPG_ppo.step(value_obs=CPG_obs, rews=CPG_rewards, dones=np.zeros(env.num_envs).astype(bool))

        average_ll_performance = reward_ll_sum / total_steps
        average_ll_performance_total.append(average_ll_performance)
        average_dones = done_sum / total_steps
        average_dones_total.append(average_dones)
    
    normalized_velocity = np.zeros(env.num_envs)[:, np.newaxis].astype(np.float32)
    CPG_obs = np.concatenate([normalized_velocity, previous_CPG], axis=1)
    CPG_ppo.update(actor_obs=CPG_obs, value_obs=CPG_obs,
                   log_this_iteration=update % 10 == 0, update=update)
    CPG_actor.distribution.enforce_minimum_std((torch.ones(CPG_signal_dim)*0.1).to(device))

    average_ll_performance_total = np.mean(average_ll_performance_total)
    average_dones_total = np.mean(average_dones_total)
    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ",
        '{:0.10f}'.format(average_ll_performance_total)))
    # print('{:<40} {:>6}'.format("sin fitting reward: ", '{:0.10f}'.format(np.mean(sin_fitting_loss))))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones_total)))
    print('{:<40} {:>6}'.format(
        "time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format(
        "fps: ", '{:6.0f}'.format((total_steps * n_CPG_steps) / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format((total_steps * n_CPG_steps) / (end - start)
                                                                    * cfg['environment']['control_dt'])))
    print('CPG std: ')
    print(np.exp(CPG_actor.distribution.std.cpu().detach().numpy()))
    print('local std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
