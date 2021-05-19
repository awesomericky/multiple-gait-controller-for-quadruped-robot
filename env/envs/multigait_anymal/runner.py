from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import multigait_anymal
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
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

Don't expect the gait you want!!

"""


def sin(x, a, b, c, d):
    return a * np.sin(b*x + c) + d


def shift_sin_param(param1, param2, mean_param):
    new_param = (param2 - param1)/mean_param
    new_param -= (new_param // (2*np.pi)) * (2*np.pi)
    assert (0 <= new_param).all() and (new_param < 2*np.pi).all()
    return new_param

# task specification
task_name = "multigait_anymal"

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

# logging
if cfg['logger'] == 'tb':
    saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                               save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
    # press refresh (F5) after the first ppo update
    tensorboard_launcher(saver.data_dir+"/..")
elif cfg['logger'] == 'wandb':
    wandb.init(project='multigait', name='experiment 1', config=dict(cfg))


# create environment from the configuration file
env = VecEnv(multigait_anymal.RaisimGymEnv(home_path + "/rsc",
             dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
# act_dim = env.num_acts  # 12 (w/ HAA joints fixed)
# act_dim = env.num_acts - 4
# act_dim = env.num_acts + 4
act_dim = env.num_acts - 2
target_signal_dim = 0

# Training
n_steps = math.floor(cfg['environment']['max_time'] /
                     cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + target_signal_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(
                             act_dim, 1.0),  # 1.0
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
# tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              logger=cfg['logger']
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

DESIRED_VELOCITY = cfg['environment']['velocity']  # m/s

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

t_range = np.arange(n_steps*2) * cfg['environment']['control_dt']
target_signal = np.zeros((4, n_steps*2))
period = 0.7  # [s]
period_param = 2 * np.pi / period  # period:
FR_target = np.pi
FL_target = 0
RR_target = 0
RL_target = np.pi
target_signal[0] = sin(t_range, 1, period_param, FR_target, 0.0)
target_signal[1] = sin(t_range, 1, period_param, FL_target, 0.0)
target_signal[2] = sin(t_range, 1, period_param, RR_target, 0.0)
target_signal[3] = sin(t_range, 1, period_param, RL_target, 0.0)

env_action = np.zeros((cfg['environment']['num_envs'], 8), dtype=np.float32)

for update in range(1000000):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + target_signal_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        contact_log = np.zeros((4, n_steps*2), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL

        for step in range(n_steps*2):
            frame_start = time.time()
            obs = env.observe(False)
            # obs_and_target = np.concatenate((obs, target_signal), axis=1, dtype=np.float32)
            action_ll = loaded_graph.architecture(torch.from_numpy(obs))
            action_ll[:, 0] = torch.relu(action_ll[:, 0])
            action_ll = action_ll.cpu().detach().numpy()
            env_action[:, [0, 2, 4, 6]] = action_ll[:, 0][:, np.newaxis] * target_signal[:, step] + action_ll[:, 1][:, np.newaxis]
            env_action[:, [1, 3, 5, 7]] = action_ll[:, 2:]
            reward_ll, dones = env.step(env_action)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)
            
            # contact logging
            env.contact_logging()
            contact_log[:, step] = env.contact_log[0, :]
        
        # save & plot contact log
        np.savez_compressed(f'contact_plot/contact_{update}.npz', contact=contact_log)

        start = 400
        total_step = 200
        single_step = 50
        fig, ax = plt.subplots(1,1, figsize=(20,10))
        img = ax.imshow(contact_log[:, start:start + total_step], aspect='auto')
        x_label_list = [i*0.01 for i in range(start + single_step, start + total_step + 1, single_step)]
        y_label_list = ['FR', 'FL', 'RR', 'RL']
        ax.set_xticks([i for i in range(single_step, total_step + 1, single_step)])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(y_label_list)
        fig.colorbar(img)
        ax.set_title("contact", fontsize=20)
        ax.set_xlabel('time [s]')
        plt.savefig(f'contact_plot/contact_{update}.png')
        plt.close()

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))
    
    # amplitude_history = np.zeros(n_steps)
    # shaft_history = np.zeros(n_steps)
    FR_thigh_joint_history = np.zeros(n_steps)
    FR_calf_joint_history = np.zeros(n_steps)
    FL_thigh_joint_history = np.zeros(n_steps)
    FL_calf_joint_history = np.zeros(n_steps)
    RR_thigh_joint_history = np.zeros(n_steps)
    RR_calf_joint_history = np.zeros(n_steps)
    RL_thigh_joint_history = np.zeros(n_steps)
    RL_calf_joint_history = np.zeros(n_steps)

    # actual training
    for step in range(n_steps):
        obs, non_obs = env.observe_logging()
        # obs_and_target = np.concatenate((obs, target_signal), axis=1, dtype=np.float32)
        action = ppo.observe(obs)
        # amplitude_history[step] = action[0, 0]
        # shaft_history[step] = action[0, 4]
        if update % 50 == 0:
            FR_thigh_joint_history[step] = non_obs[0, 4]
            FR_calf_joint_history[step] = non_obs[0, 5]
            FL_thigh_joint_history[step] = non_obs[0, 6]
            FL_calf_joint_history[step] = non_obs[0, 7]
            RR_thigh_joint_history[step] = non_obs[0, 8]
            RR_calf_joint_history[step] = non_obs[0, 9]
            RL_thigh_joint_history[step] = non_obs[0, 10]
            RL_calf_joint_history[step] = non_obs[0,11]

        # action[:, [8, 10, 12, 14]] = action[:, :4] * target_signal[:, step] + action[:, 4:8] + action[:, [8, 10, 12, 14]]
        # action[:, [0, 2, 4, 6]] = np.tile(A[:, np.newaxis], (1, 4)) * target_signal[:, step] + action[:, [0, 2, 4, 6]]

        # # Architecture 1
        # env_action[:, [0, 2, 4, 6]] = target_signal[:, step] + action[:, :4]
        # env_action[:, [1, 3, 5, 7]] =  action[:, 4:]

        # # Architecture 2
        # env_action[:, [0, 2, 4, 6]] = target_signal[:, step]
        # env_action[:, [1, 3, 5, 7]] =  action`

        # # Architecture 4
        # env_action[:, [0, 2, 4, 6]] = action[:, :4] * target_signal[:, step] + action[:, 4:8]
        # env_action[:, [1, 3, 5, 7]] = action[:, 8:]

        # Architecture 5
        env_action[:, [0, 2, 4, 6]] = action[:, 0][:, np.newaxis] * target_signal[:, step] + action[:, 1][:, np.newaxis]
        env_action[:, [1, 3, 5, 7]] = action[:, 2:]

        reward, dones = env.step(env_action)
        # reward, dones = env.step(action)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)

        if (update % 5 == 0) and (step + 1) % 100 == 0 and 1 < cfg['environment']['num_envs']:
            env.reward_logging()
            ppo.extra_log(env.reward_log, (update // 5) * n_steps + step, type='reward')
        
        if update % 10 == 0:
            ppo.extra_log(action, (update // 10) * n_steps + step, type='action')
        

        # LF_HFE_history.append(obs[:, 4])
        # RF_HFE_history.append(obs[:, 7])
        # LH_HFE_history.append(obs[:, 10])
        # RH_HFE_history.append(obs[:, 13])

    if update % 50 == 0:
        fig, ax = plt.subplots(2,2,figsize=(28, 15))

        # FR_thigh
        ax[0, 0].plot(t_range[:n_steps], FR_thigh_joint_history, 'o', label='joint angle [rad]')
        ax[0, 0].plot(t_range[:n_steps], target_signal[0, :n_steps], label='signal')
        ax[0, 0].set_xlabel('time [s]', fontsize=20)
        ax[0, 0].set_title('FR', fontsize=25)

        # FL_thigh
        ax[0, 1].plot(t_range[:n_steps], FL_thigh_joint_history, 'o', label='joint angle [rad]')
        ax[0, 1].plot(t_range[:n_steps], target_signal[1, :n_steps], label='signal')
        ax[0, 1].set_xlabel('time [s]', fontsize=20)
        ax[0, 1].set_title('FL', fontsize=25)

        # RR_thigh
        ax[1, 0].plot(t_range[:n_steps], RR_thigh_joint_history, 'o', label='joint angle [rad]')
        ax[1, 0].plot(t_range[:n_steps], target_signal[2, :n_steps], label='signal')
        ax[1, 0].set_xlabel('time [s]', fontsize=20)
        ax[1, 0].set_title('RR', fontsize=25)

        # RL_thigh
        ax[1, 1].plot(t_range[:n_steps], RL_thigh_joint_history, 'o', label='joint angle [rad]')
        ax[1, 1].plot(t_range[:n_steps], target_signal[3, :n_steps], label='signal')
        ax[1, 1].set_xlabel('time [s]', fontsize=20)
        ax[1, 1].set_title('RL', fontsize=25)

        plt.legend()
        plt.savefig(f'joint_plot/Thigh_joint_angle_{update}.png')
        plt.close()


        fig, ax = plt.subplots(2,2,figsize=(28,15))

        # FR_calf
        ax[0, 0].plot(t_range[:n_steps], FR_calf_joint_history, 'o', label='joint angle [rad]')
        ax[0, 0].set_title('FR', fontsize=25)
        ax[0, 0].set_xlabel('time [s]', fontsize=20)

        # FL_calf
        ax[0, 1].plot(t_range[:n_steps], FL_calf_joint_history, 'o', label='joint angle [rad]')
        ax[0, 1].set_title('FL', fontsize=25)
        ax[0, 1].set_xlabel('time [s]', fontsize=20)

        # RR_calf
        ax[1, 0].plot(t_range[:n_steps], RR_calf_joint_history, 'o', label='joint angle [rad]')
        ax[1, 0].set_title('RR', fontsize=25)
        ax[1, 0].set_xlabel('time [s]', fontsize=20)

        # RL_calf
        ax[1, 1].plot(t_range[:n_steps], RL_calf_joint_history, 'o', label='joint angle [rad]')
        ax[1, 1].set_title('RL', fontsize=25)
        ax[1, 1].set_xlabel('time [s]', fontsize=20)

        plt.legend()
        plt.savefig(f'joint_plot/Calf_joint_angle_{update}.png')
        plt.close()

    # take st step to get value obs
    obs, _ = env.observe_logging()
    # obs_and_target = np.concatenate((obs, target_signal), axis=1, dtype=np.float32)
    # ppo.update(actor_obs=obs_and_target, value_obs=obs, log_this_iteration=update % 10 == 0, update=update, auxilory_value=sin_fitting_loss[:, np.newaxis])
    ppo.update(actor_obs=obs, value_obs=obs,
               log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std(
        (torch.ones(act_dim)*0.2).to(device))

    end = time.time()
    # pdb.set_trace()

    # wandb.log({'amplitude std': np.std(amplitude_history), 'shaft std': np.std(shaft_history)})

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ",
          '{:0.10f}'.format(average_ll_performance)))
    # print('{:<40} {:>6}'.format("sin fitting reward: ", '{:0.10f}'.format(np.mean(sin_fitting_loss))))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format(
        "time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format(
        "fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
