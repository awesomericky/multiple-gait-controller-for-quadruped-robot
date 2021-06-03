from numpy.lib.type_check import real_if_close
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import multigait_anymal_transfer
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.utils import joint_angle_plotting, contact_plotting, CPG_and_velocity_plotting
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
This file is for discrete CPG runner. 
The CPG signal is narrowed down to three types (pace, trot, bound)

"""


def sin(x, a, b):
    return np.sin(a*x + b)

def sin_derivatice(x, a, b):
    return a * np.cos(a*x + b)

def compute_phase(previous_CPG, current_CPG):
    """
    previous_CPG: (env.num_envs, 4)
    current_CPG: (env.num_envs, 4)

    if current_CPG - previous_CPG < 0 ==> swing phase (= 0)
    elif current_CPG - previous_CPG >= 0 ==> stance phase (= 1)
    """
    return np.where(current_CPG - previous_CPG >= 0, 1, 0).astype(np.float32)

# task specification
task_name = "single_CPG_w_velocity_change"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument(
    '-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# check if gpu is available
if cfg['force_CPU']:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task_specific_folder_name = f"{cfg['environment']['gait']}_{int(cfg['environment']['velocity']['min'])}_{int(cfg['environment']['velocity']['max'])}"

# create environment from the configuration file
env = VecEnv(multigait_anymal_transfer.RaisimGymEnv(home_path + "/rsc",
             dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

target_gait_dict = {'pace': [np.pi, 0, np.pi, 0], 'trot': [np.pi, 0, 0, np.pi], 'bound': [np.pi, np.pi, 0, 0]}

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
act_dim = env.num_acts - 3  # 8 - 3 (w/ HAA joints fixed)
CPG_signal_dim = 1
CPG_signal_state_dim = 4

# Training
n_CPG_steps = math.floor(cfg['environment']['max_time'] /
                         cfg['environment']['CPG_control_dt'])
n_steps = math.floor(cfg['environment']['max_time'] /
                     cfg['environment']['control_dt'])
total_CPG_steps = n_CPG_steps * env.num_envs
total_steps = n_steps * env.num_envs
CPG_period = int(cfg['environment']['CPG_control_dt'] / cfg['environment']['control_dt'])  # 5
velocity_period = int(cfg['environment']['velocity_sampling_dt'] / cfg['environment']['control_dt'])  # 400

assert velocity_period % CPG_period == 0, "velocity_sampling_dt should be integer multiple of CPG_control_dt"

min_vel = cfg['environment']['velocity']['min']
max_vel = cfg['environment']['velocity']['max']

avg_rewards = []

CPG_actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['CPG_policy_net'], nn.LeakyReLU, 1, CPG_signal_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(
                             CPG_signal_dim, 0.1, type='CPG', device=device),
                         device)
CPG_critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['CPG_value_net'], nn.LeakyReLU, 1, 1),
                           device)

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + CPG_signal_dim + CPG_signal_state_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(
                             act_dim, 1.0, device=device),  # 1.0
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + CPG_signal_dim + CPG_signal_state_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"], task_specific_name=task_specific_folder_name)

# logging
if cfg['logger'] == 'tb':
    tensorboard_launcher(saver.data_dir+"/..")   # press refresh (F5) after the first ppo update
    pass
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
    load_param(weight_path, env, CPG_actor, CPG_critic, CPG_ppo.optimizer, saver.data_dir, type='CPG')
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir, type='local')

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

target_gait_phase = np.array(target_gait_dict[cfg['environment']['gait']])
evaluate_n_steps = n_steps * 2

# Initialize
FR_thigh_joint_history = np.zeros(n_steps)
FR_calf_joint_history = np.zeros(n_steps)
FL_thigh_joint_history = np.zeros(n_steps)
FL_calf_joint_history = np.zeros(n_steps)
RR_thigh_joint_history = np.zeros(n_steps)
RR_calf_joint_history = np.zeros(n_steps)
RL_thigh_joint_history = np.zeros(n_steps)
RL_calf_joint_history = np.zeros(n_steps)
env_action = np.zeros((env.num_envs, 8), dtype=np.float32)
CPG_signal = np.zeros((env.num_envs, 4, evaluate_n_steps), dtype=np.float32)

# initialize logging value (when evaluating)
contact_log = np.zeros((4, evaluate_n_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
CPG_signal_period_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
target_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
real_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)

for update in range(4000):
    
    ## Evaluating ##
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

        make_new_graph = (update % 2000 == 0) and (update != 0)
        
        if make_new_graph:
            # we create another graph just to demonstrate the save/load method
            CPG_loaded_graph = ppo_module.MLP(cfg['architecture']['CPG_policy_net'], nn.LeakyReLU, 1, CPG_signal_dim)
            CPG_loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['CPG_actor_architecture_state_dict'])
            local_loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + CPG_signal_dim + CPG_signal_state_dim, act_dim)
            local_loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.reset()
        # env.turn_on_visualization()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        # initialize value
        CPG_a_new = np.zeros((env.num_envs, 1))
        CPG_b_new = np.broadcast_to(target_gait_phase[:, np.newaxis], (env.num_envs, 4, 1))
        CPG_a_old = np.zeros((env.num_envs, 1))
        CPG_b_old = np.broadcast_to(target_gait_phase[:, np.newaxis], (env.num_envs, 4, 1))
        # CPG_signal = np.zeros((env.num_envs, 4, evaluate_n_steps), dtype=np.float32)
        # env_action = np.zeros((env.num_envs, 8), dtype=np.float32)

        # # initialize logging value
        # contact_log = np.zeros((4, evaluate_n_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
        # CPG_signal_period_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
        # target_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
        # real_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
        
        for step in range(evaluate_n_steps):
            frame_start = time.time()

            if step % CPG_period == 0:
                if step % velocity_period == 0:
                    # sample new velocity
                    if cfg['environment']['single_velocity']:
                        normalized_velocity = np.broadcast_to(np.random.uniform(low = 0, high=1, size=1)[:, np.newaxis], (env.num_envs, 1)).astype(np.float32)
                    else:
                        normalized_velocity = np.random.uniform(low = 0, high=1, size=env.num_envs)[:, np.newaxis].astype(np.float32)
                    velocity = normalized_velocity * (max_vel - min_vel) + min_vel

                    env.set_target_velocity(velocity)
                
                # generate new CPG signal parameter
                CPG_a_old = CPG_a_new.copy()
                CPG_b_old = CPG_b_new.copy()
                # CPG_signal_period = CPG_loaded_graph.architecture(torch.from_numpy(normalized_velocity))
                with torch.no_grad():
                    if make_new_graph:
                        CPG_signal_period = CPG_loaded_graph.architecture(torch.from_numpy(velocity))
                        CPG_signal_period = torch.clamp(CPG_signal_period, min=0.1, max=1.).cpu().detach().numpy()
                    else:
                        CPG_signal_period = CPG_ppo.inference(velocity)

                CPG_a_new = 2 * np.pi / (CPG_signal_period + 1e-6)
                CPG_b_new = ((CPG_a_old - CPG_a_new) * (step * cfg['environment']['control_dt']))[:, np.newaxis, :] + CPG_b_old

                assert id(CPG_a_old) != id(CPG_a_new), "Check memory allocation"
                assert id(CPG_b_old) != id(CPG_b_new), "Check memory allocation"

                # generate CPG signal
                t_period = CPG_period
                t_range = (np.arange(step, step + t_period) * cfg['environment']['control_dt'])[np.newaxis, np.newaxis, :]
                CPG_signal[:, :, step:step + t_period] = sin(t_range, CPG_a_new[:, :, np.newaxis], CPG_b_new)
                # CPG_signal & CPG_signal_derivative dimension change is as following
                # (1, 1, CPG_period) (n_env, 1, 1) (n_env, 4, 1)  ==> (n_env, 4, CPG_period)
            
            obs, non_obs = env.observe_logging(False)
            CPG_phase = ((step * cfg['environment']['control_dt']) + (CPG_b_new / CPG_a_new[:, np.newaxis, :])) / (2 * np.pi / CPG_a_new[:, np.newaxis, :]) \
                    - (((step * cfg['environment']['control_dt']) + (CPG_b_new / CPG_a_new[:, np.newaxis, :])) / (2 * np.pi / CPG_a_new[:, np.newaxis, :])).astype(int)
            CPG_phase = np.squeeze(CPG_phase)
            assert (0 <= CPG_phase).all() and (CPG_phase <= 1).all(), "CPG_phase not in correct range"
            
            obs = np.concatenate([obs, CPG_signal_period, CPG_phase], axis=1, dtype=np.float32)

            with torch.no_grad():
                if make_new_graph:
                    action_ll = local_loaded_graph.architecture(torch.from_numpy(obs))
                    action_ll[:, 0] = torch.relu(action_ll[:, 0])
                    action_ll = action_ll.cpu().detach().numpy()
                else:
                    action_ll = ppo.inference(obs)

            # env_action[:, [0, 2, 4, 6]] = action_ll[:, 0][:, np.newaxis] * CPG_signal[:, :, step] + action_ll[:, 1][:, np.newaxis]
            env_action[:, [0, 2, 4, 6]] = action_ll[:, 0][:, np.newaxis] * CPG_signal[:, :, step]
            env_action[:, [1, 3, 5, 7]] = action_ll[:, 1:]
            reward_ll, dones = env.step(env_action)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)
            
            # contact logging
            env.contact_logging()
            contact_log[:, step] = env.contact_log[0, :]

            # CPG_signal_param, target velocity, actual velocity logging
            CPG_signal_period_traj[step] = CPG_signal_period[0]
            target_velocity_traj[step] = velocity[0]
            real_velocity_traj[step] = non_obs[0, 12]
            # CPG_ppo.extra_log(CPG_signal_period, update * n_steps + step, type='action')
            # CPG_ppo.extra_log(velocity, update * n_steps + step, type='target_veloicty')

        
        # save & plot contact log
        contact_plotting(update, task_specific_folder_name, contact_log)
        CPG_and_velocity_plotting(update, task_specific_folder_name, evaluate_n_steps, CPG_signal_period_traj, target_velocity_traj, real_velocity_traj)

        # env.stop_video_recording()
        # env.turn_off_visualization()

        env.save_scaling(saver.data_dir, str(update))
    
    ### TRAINING ###

    start = time.time()
    env.reset()

    reward_CPG_sum = 0
    reward_local_sum = 0
    done_sum = 0

    CPG_a_new = np.zeros((env.num_envs, 1))
    CPG_b_new = np.broadcast_to(target_gait_phase[:, np.newaxis], (env.num_envs, 4, 1))
    CPG_a_old = np.zeros((env.num_envs, 1))
    CPG_b_old = np.broadcast_to(target_gait_phase[:, np.newaxis], (env.num_envs, 4, 1))
    CPG_rewards = np.zeros((env.num_envs, 1), dtype=np.float32)
    CPG_not_dones = np.ones((env.num_envs,), dtype=np.float32)

    for step in range(n_steps):
        if step % CPG_period == 0:
            if step % velocity_period == 0:
                # sample new velocity
                if cfg['environment']['single_velocity']:
                    normalized_velocity = np.broadcast_to(np.random.uniform(low = 0, high=1, size=1)[:, np.newaxis], (env.num_envs, 1)).astype(np.float32)
                else:
                    normalized_velocity = np.random.uniform(low = 0, high=1, size=env.num_envs)[:, np.newaxis].astype(np.float32)
                velocity = normalized_velocity * (max_vel - min_vel) + min_vel

                env.set_target_velocity(velocity)
            
            # generate new CPG signal parameter (w/o random initialize)
            CPG_a_old = CPG_a_new.copy()
            CPG_b_old = CPG_b_new.copy()
            # CPG_signal_period = CPG_ppo.observe(normalized_velocity)  # CPG_ppo policy outputs period
            CPG_signal_period = CPG_ppo.observe(velocity)

            CPG_a_new = 2 * np.pi / CPG_signal_period
            CPG_b_new = ((CPG_a_old - CPG_a_new) * (step * cfg['environment']['control_dt']))[:, np.newaxis, :] + CPG_b_old

            assert id(CPG_a_old) != id(CPG_a_new), "Check memory allocation"
            assert id(CPG_b_old) != id(CPG_b_new), "Check memory allocation"

            # generate CPG signal
            t_period = CPG_period + 1 if (step == n_steps - CPG_period) else CPG_period
            t_range = (np.arange(step, step + t_period) * cfg['environment']['control_dt'])[np.newaxis, np.newaxis, :]
            CPG_signal[:, :, step:step + t_period] = sin(t_range, CPG_a_new[:, :, np.newaxis], CPG_b_new)
            # CPG_signal & CPG_signal_derivative dimension change is as following
            # (1, 1, CPG_period) (n_env, 1, 1) (n_env, 4, 1)  ==> (n_env, 4, CPG_period)

        # set leg phase to compute reward corresponding to gait contact pattern
        if step > 0:
            leg_phase = compute_phase(previous_CPG=CPG_signal[:, :, step-1], current_CPG=CPG_signal[:, :, step])
            env.set_leg_phase(leg_phase)
        
        obs, non_obs = env.observe_logging()
        
        CPG_phase = ((step * cfg['environment']['control_dt']) + (CPG_b_new / CPG_a_new[:, np.newaxis, :])) / (2 * np.pi / CPG_a_new[:, np.newaxis, :]) \
                    - (((step * cfg['environment']['control_dt']) + (CPG_b_new / CPG_a_new[:, np.newaxis, :])) / (2 * np.pi / CPG_a_new[:, np.newaxis, :])).astype(int)
        CPG_phase = np.squeeze(CPG_phase)
        assert (0 <= CPG_phase).all() and (CPG_phase <= 1).all(), "CPG_phase not in correct range"
        
        obs = np.concatenate([obs, CPG_signal_period, CPG_phase], axis=1, dtype=np.float32)
        action = ppo.observe(obs)

        # save joint value for plotting
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
        env_action[:, [0, 2, 4, 6]] = action[:, 0][:, np.newaxis] * CPG_signal[:, :, step]
        env_action[:, [1, 3, 5, 7]] = action[:, 1:]

        reward, dones = env.step(env_action)

        env.get_CPG_reward()
        temp_CPG_rewards = env._CPG_reward
        CPG_rewards += (temp_CPG_rewards * (1 - dones))[:, np.newaxis]
        CPG_not_dones *= (1 - dones)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum += sum(dones)
        reward_local_sum += sum(reward)

        # update CPG rewards in storage
        if (step + 1) % CPG_period == 0:
            # CPG_ppo.step(value_obs=normalized_velocity, rews=CPG_rewards, dones=(1 - CPG_not_dones).astype(bool))
            CPG_ppo.step(value_obs=velocity, rews=CPG_rewards, dones=(1 - CPG_not_dones).astype(bool))
            
            if (update % 5 == 0) and (1 < cfg['environment']['num_envs']):
                # CPG_ppo.extra_log(CPG_rewards, update * n_steps + step, type='reward')
                CPG_ppo.extra_log(CPG_signal_period, update * n_steps + step, type='action')
                CPG_ppo.extra_log(velocity, update * n_steps + step, type='target_veloicty')
            
            reward_CPG_sum += np.sum(CPG_rewards)
            CPG_rewards = np.zeros((env.num_envs, 1), dtype=np.float32)
            CPG_not_dones = np.ones((env.num_envs,), dtype=np.float32)
        
        # log reward for CPG policy
        if (update % 5 == 0) and (step % 25 == 0) and (1 < cfg['environment']['num_envs']):
            env.reward_logging()
            ppo.extra_log(env.reward_log, update * n_steps + step, type='reward')
            ppo.extra_log(action, update * n_steps + step, type='action')
    
    # update CPG policy
    velocity = np.zeros(env.num_envs)[:, np.newaxis].astype(np.float32)
    CPG_ppo.update(actor_obs=velocity, value_obs=velocity,
                   log_this_iteration=update % 10 == 0, update=update)
    CPG_actor.distribution.enforce_minimum_std((torch.ones(CPG_signal_dim)*0.03).to(device))

    # update local policy
    obs, _ = env.observe_logging()

    CPG_phase = ((step * cfg['environment']['control_dt']) + (CPG_b_new / CPG_a_new[:, np.newaxis, :])) / (2 * np.pi / CPG_a_new[:, np.newaxis, :]) \
                - (((step * cfg['environment']['control_dt']) + (CPG_b_new / CPG_a_new[:, np.newaxis, :])) / (2 * np.pi / CPG_a_new[:, np.newaxis, :])).astype(int)
    CPG_phase = np.squeeze(CPG_phase)
    assert (0 <= CPG_phase).all() and (CPG_phase <= 1).all(), "CPG_phase not in correct range"

    obs = np.concatenate([obs, CPG_signal_period, CPG_phase], axis=1, dtype=np.float32)
    ppo.update(actor_obs=obs, value_obs=obs,
            log_this_iteration=update % 10 == 0, update=update)
    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    # compute average performance
    average_CPG_performance = reward_CPG_sum / total_CPG_steps
    average_local_performance = reward_local_sum / total_steps
    average_dones = done_sum / total_steps

    # plot joint value
    if update % 50 == 0:
        joint_angle_plotting(update, task_specific_folder_name, np.arange(n_steps) * cfg['environment']['control_dt'], CPG_signal[0, :, :n_steps],\
                                FR_thigh_joint_history, FL_thigh_joint_history, RR_thigh_joint_history, RL_thigh_joint_history,\
                                FR_calf_joint_history, FL_calf_joint_history, RR_calf_joint_history, RL_calf_joint_history)

    end = time.time()

    # # increase cost (curriculum learning)
    # env.increase_cost_scale()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average CPG reward: ",
        '{:0.10f}'.format(average_CPG_performance)))
    print('{:<40} {:>6}'.format("average local reward: ",
        '{:0.10f}'.format(average_local_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
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
