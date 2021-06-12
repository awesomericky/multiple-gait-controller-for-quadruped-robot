from ruamel.yaml import YAML, dump, RoundTripDumper, tokens
from raisimGymTorch.env.bin import multigait_anymal_transfer
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
from raisimGymTorch.helper.utils import exp_contact_plotting, exp_CPG_and_velocity_plotting
import os
import math
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle


# configuration
# 
# ex) 
# --weight: /home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/data/anymal_locomotion/2021-05-04-20-57-12/full_400.pt

def sin(x, a, b):
    return np.sin(a*x + b)

target_gait_dict = np.array([[np.pi, 0, 0, np.pi], [np.pi, 0, np.pi, 0], [np.pi, np.pi, 0, 0]])
# 0 : trot, 1: pace, 2: bound

# target_gait_dict = np.array([[0, np.pi, 1.5 * np.pi, 0.5 * np.pi], [0, np.pi, 0, np.pi], [0, 0, np.pi, np.pi]])
# # 0 : walk, 1: pace, 2: bound

def compute_target_gait_phase(velocity):
    """
    Input:
        velocity: (num_envs, 1)
    Output:
        target_phase: (num_envs, 4)
    """
    velocity = np.squeeze(velocity)
    phase_idx = np.where(velocity < 0.5, 0, 1)
    phase_idx += np.where(velocity < 1.0, 0, 1)
    target_phase = target_gait_dict[phase_idx, :]

    gait_encoding = np.zeros((phase_idx.size, target_gait_dict.shape[0]))
    gait_encoding[np.arange(phase_idx.size), phase_idx] = 1
    target_phase = target_gait_dict[phase_idx, :]
    return target_phase, gait_encoding


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()
weight_path = args.weight

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
# cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
cfg = YAML().load(open(os.path.join(weight_path.rsplit('/', 1)[0], 'cfg.yaml'), 'r'))

# create environment from the configuration file
training_num_envs = cfg['environment']['num_envs']
cfg['environment']['num_envs'] = 1

env = VecEnv(multigait_anymal_transfer.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
act_dim = env.num_acts - 3  # 8 - 3 (w/ HAA joints fixed)
CPG_signal_dim = 1
CPG_signal_state_dim = 4
gait_encoding_dim = 3

min_vel = cfg['environment']['velocity']['min']
max_vel = cfg['environment']['velocity']['max']

CPG_period = int(cfg['environment']['CPG_control_dt'] / cfg['environment']['control_dt'])  # 5
wait_period = 200
velocity_period = 500

iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

task_specific_folder_name = f"{cfg['environment']['gait']}_{cfg['environment']['velocity']['min']}_{cfg['environment']['velocity']['max']}"

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    CPG_loaded_graph = ppo_module.MLP(cfg['architecture']['CPG_policy_net'], torch.nn.LeakyReLU, 1, CPG_signal_dim)
    CPG_loaded_graph.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))['CPG_actor_architecture_state_dict'])
    local_loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + CPG_signal_dim + CPG_signal_state_dim + gait_encoding_dim, act_dim)
    local_loaded_graph.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 2500 ## 12 secs

    count = 0

    # initialize value
    CPG_signal = np.zeros((training_num_envs, 4, max_steps), dtype=np.float32)
    env_action = np.zeros((training_num_envs, 8), dtype=np.float32)

    # initialize logging value
    contact_log = np.zeros((4, max_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
    CPG_signal_period_traj = np.zeros((max_steps, ), dtype=np.float32)
    target_velocity_traj = np.zeros((max_steps, ), dtype=np.float32)
    real_velocity_traj = np.zeros((max_steps, ), dtype=np.float32)

    # initialize exp2 data
    velocity_error_collection = dict()
    torque_collection = []
    power_collection = []

    for step in range(max_steps):
        frame_start = time.time()

        if step % CPG_period == 0:
            if step % velocity_period == 0:
                data_collection_start = step + wait_period
                velocity_error = []

                if count == 0:
                    velocity = np.broadcast_to(np.array([0.3])[:, np.newaxis], (training_num_envs, 1)).astype(np.float32)
                elif count == 1:
                    velocity = np.broadcast_to(np.array([0.6])[:, np.newaxis], (training_num_envs, 1)).astype(np.float32)
                elif count == 2:
                    velocity = np.broadcast_to(np.array([0.9])[:, np.newaxis], (training_num_envs, 1)).astype(np.float32)
                elif count == 3:
                    velocity = np.broadcast_to(np.array([1.2])[:, np.newaxis], (training_num_envs, 1)).astype(np.float32)
                else:
                    velocity = np.broadcast_to(np.array([1.5])[:, np.newaxis], (training_num_envs, 1)).astype(np.float32)

                env.set_target_velocity(velocity)
                count += 1

                if step == 0:
                    # initialize value
                    target_gait_phase, gait_encoding = compute_target_gait_phase(velocity)
                    previous_gait_phase = target_gait_phase.copy()
                    CPG_a_new = np.zeros((env.num_envs, 1))
                    CPG_b_new = target_gait_phase[:, :, np.newaxis]
                    CPG_a_old = np.zeros((env.num_envs, 1))
                    CPG_b_old = target_gait_phase[:, :, np.newaxis]
                else:
                    # update value
                    target_gait_phase, gait_encoding = compute_target_gait_phase(velocity)
                    CPG_b_extra = target_gait_phase[:, :, np.newaxis] - previous_gait_phase[:, :, np.newaxis]
                    previous_gait_phase = target_gait_phase.copy()
            
            # generate new CPG signal parameter
            CPG_a_old = CPG_a_new.copy()
            CPG_b_old = CPG_b_new.copy()
            with torch.no_grad():
                CPG_signal_period = CPG_loaded_graph.architecture(torch.from_numpy(velocity))
                CPG_signal_period = torch.clamp(CPG_signal_period, min=0.1, max=1.0).cpu().detach().numpy()

            CPG_a_new = 2 * np.pi / (CPG_signal_period + 1e-6)
            if (step % velocity_period == 0) and (step != 0):
                CPG_b_new = ((CPG_a_old - CPG_a_new) * (step * cfg['environment']['control_dt']))[:, np.newaxis, :] + CPG_b_old + CPG_b_extra
            else:
                CPG_b_new = ((CPG_a_old - CPG_a_new) * (step * cfg['environment']['control_dt']))[:, np.newaxis, :] + CPG_b_old

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
        
        obs = np.concatenate([obs, CPG_signal_period, CPG_phase, gait_encoding], axis=1, dtype=np.float32)

        with torch.no_grad():
            action_ll = local_loaded_graph.architecture(torch.from_numpy(obs))
            action_ll[:, 0] = torch.relu(action_ll[:, 0])
            action_ll = action_ll.cpu().detach().numpy()

        # env_action[:, [0, 2, 4, 6]] = action_ll[:, 0][:, np.newaxis] * CPG_signal[:, :, step] + action_ll[:, 1][:, np.newaxis]
        env_action[:, [0, 2, 4, 6]] = action_ll[:, 0][:, np.newaxis] * CPG_signal[:, :, step]
        env_action[:, [1, 3, 5, 7]] = action_ll[:, 1:]
        reward_ll, dones = env.step(env_action)
        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

        if step in range(data_collection_start, data_collection_start + (velocity_period - wait_period)):
            real_velocity = non_obs[0, 12]
            desired_velocity = velocity[0, 0]

            velocity_error.append(abs(real_velocity - desired_velocity))

            current_torque = env.get_torque()[0]
            current_power = env.get_power()[0]
            torque_collection.append([real_velocity, current_torque])
            power_collection.append([real_velocity, current_power])

            if step == data_collection_start + (velocity_period - wait_period) - 1:
                velocity_error = np.asarray(velocity_error)
                velocity_error_collection[desired_velocity] = [np.mean(velocity_error), np.std(velocity_error), np.quantile(velocity_error, .25), np.quantile(velocity_error, .50), np.quantile(velocity_error, .75)]
        
        # contact logging
        env.contact_logging()
        contact_log[:, step] = env.contact_log[0, :]

        # CPG_signal_param, target velocity, actual velocity logging
        CPG_signal_period_traj[step] = CPG_signal_period[0]
        target_velocity_traj[step] = velocity[0]
        real_velocity_traj[step] = non_obs[0, 12]
    
    # save exp2 data
    torque_collection = np.asarray(torque_collection)
    power_collection = np.asarray(power_collection)
    with open(f"raisimGymTorch/exp_result/exp2/vel_error_{cfg['environment']['gait']}.pkl", "wb") as f:
        pickle.dump(velocity_error_collection, f)
    np.savez_compressed(f"raisimGymTorch/exp_result/exp2/torque_{cfg['environment']['gait']}", torque=torque_collection)
    np.savez_compressed(f"raisimGymTorch/exp_result/exp2/power_{cfg['environment']['gait']}", power=power_collection)

    # save & plot contact log
    update = cfg['environment']['gait']
    exp_contact_plotting(update, "raisimGymTorch/exp_result/exp2", contact_log)
    exp_CPG_and_velocity_plotting(update, "raisimGymTorch/exp_result/exp2", max_steps, CPG_signal_period_traj, target_velocity_traj, real_velocity_traj)

    env.turn_off_visualization()
    env.reset()

    print("Finished at the maximum visualization steps")
