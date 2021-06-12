from numpy.lib.type_check import real_if_close
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import multigait_anymal_transfer
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher, hierarchical_load_param
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

task_specific_folder_name = f"{cfg['environment']['gait']}_{cfg['environment']['velocity']['min']}_{cfg['environment']['velocity']['max']}"

# create environment from the configuration file
env = VecEnv(multigait_anymal_transfer.RaisimGymEnv(home_path + "/rsc",
             dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
act_dim = env.num_acts  # 8 (w/ HAA joints fixed)

# Training
n_steps = math.floor(cfg['environment']['max_time'] /
                     cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
velocity_period = int(cfg['environment']['velocity_sampling_dt'] / cfg['environment']['control_dt'])  # 400

min_vel = cfg['environment']['velocity']['min']
max_vel = cfg['environment']['velocity']['max']

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + 1, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(
                             act_dim, 1.0, device=device),  # 1.0
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + 1, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"], task_specific_name=task_specific_folder_name)

# logging
if cfg['logger'] == 'tb':
    tensorboard_launcher(saver.data_dir+"/..")   # press refresh (F5) after the first ppo update
    #pass
elif cfg['logger'] == 'wandb':
    wandb.init(project='multigait', name='experiment 1', config=dict(cfg))

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              PPO_type=None,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              logger=cfg['logger']
              )

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

# initialize logging value (when evaluating)
contact_log = np.zeros((4, evaluate_n_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
target_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
real_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)

if weight_path != '':
    iteration_number = int(weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0])
else:
    iteration_number = 0

for update in range(iteration_number, 10000):
    
    ## Evaluating ##
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        make_new_graph = (update % 2000 == 0) and (update != 0)
        
        if make_new_graph:
            # we create another graph just to demonstrate the save/load method
            local_loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + 1, act_dim)
            local_loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.reset()
        # env.turn_on_visualization()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        # initialize logging value
        contact_log = np.zeros((4, evaluate_n_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
        target_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
        real_velocity_traj = np.zeros((evaluate_n_steps, ), dtype=np.float32)
        
        for step in range(evaluate_n_steps):
            frame_start = time.time()

            if step % velocity_period == 0:
                # sample new velocity
                if cfg['environment']['single_velocity']:
                    normalized_velocity = np.broadcast_to(np.random.uniform(low = 0, high=1, size=1)[:, np.newaxis], (env.num_envs, 1)).astype(np.float32)
                else:
                    normalized_velocity = np.random.uniform(low = 0, high=1, size=env.num_envs)[:, np.newaxis].astype(np.float32)
                velocity = normalized_velocity * (max_vel - min_vel) + min_vel

                env.set_target_velocity(velocity)
                
            obs, non_obs = env.observe_logging(False)
            
            obs = np.concatenate([obs, velocity], axis=1, dtype=np.float32)

            with torch.no_grad():
                if make_new_graph:
                    action_ll = local_loaded_graph.architecture(torch.from_numpy(obs))
                    action_ll = action_ll.cpu().detach().numpy()
                else:
                    action_ll = ppo.inference(obs)

            reward_ll, dones = env.step(action_ll)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)
            
            # contact logging
            env.contact_logging()
            contact_log[:, step] = env.contact_log[0, :]

            # CPG_signal_param, target velocity, actual velocity logging
            target_velocity_traj[step] = velocity[0]
            real_velocity_traj[step] = non_obs[0, 12]
        
        # save & plot contact log
        contact_plotting(update, saver.plot_dir, contact_log)
        CPG_and_velocity_plotting(update, saver.plot_dir, evaluate_n_steps, None, target_velocity_traj, real_velocity_traj)

        # env.stop_video_recording()
        # env.turn_off_visualization()

        env.save_scaling(saver.data_dir, str(update))
    
    ### TRAINING ###

    start = time.time()
    env.reset()

    reward_sum = 0
    done_sum = 0

    for step in range(n_steps):
        if step % velocity_period == 0:
            # sample new velocity
            if cfg['environment']['single_velocity']:
                normalized_velocity = np.broadcast_to(np.random.uniform(low = 0, high=1, size=1)[:, np.newaxis], (env.num_envs, 1)).astype(np.float32)
            else:
                normalized_velocity = np.random.uniform(low = 0, high=1, size=env.num_envs)[:, np.newaxis].astype(np.float32)
            velocity = normalized_velocity * (max_vel - min_vel) + min_vel

            env.set_target_velocity(velocity)
            
        obs, non_obs = env.observe_logging()
        
        obs = np.concatenate([obs, velocity], axis=1, dtype=np.float32)
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

        reward, dones = env.step(action)

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum += sum(dones)
        reward_sum += sum(reward)

        # log reward for CPG policy
        if (update % 5 == 0) and (step % 25 == 0) and (1 < cfg['environment']['num_envs']):
            env.reward_logging()
            ppo.extra_log(env.reward_log, update * n_steps + step, type='reward')
            ppo.extra_log(action, update * n_steps + step, type='action')
            # pass
    
    velocity = np.zeros(env.num_envs)[:, np.newaxis].astype(np.float32)

    # update policy
    obs, _ = env.observe_logging()

    obs = np.concatenate([obs, velocity], axis=1, dtype=np.float32)
    ppo.update(actor_obs=obs, value_obs=obs,
            log_this_iteration=update % 10 == 0, update=update)
    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    # compute average performance
    average_performance = reward_sum / total_steps
    average_dones = done_sum / total_steps

    # plot joint value
    if update % 50 == 0:
        joint_angle_plotting(update, saver.plot_dir, np.arange(n_steps) * cfg['environment']['control_dt'], None,\
                                FR_thigh_joint_history, FL_thigh_joint_history, RR_thigh_joint_history, RL_thigh_joint_history,\
                                FR_calf_joint_history, FL_calf_joint_history, RR_calf_joint_history, RL_calf_joint_history)

    end = time.time()

    # increase cost (curriculum learning)
    env.increase_cost_scale()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average reward: ",
        '{:0.10f}'.format(average_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format(
        "time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format(
        "fps: ", '{:6.0f}'.format((total_steps) / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format((total_steps) / (end - start)
                                                                    * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
