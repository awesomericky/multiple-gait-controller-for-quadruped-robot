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
act_dim = env.num_acts  # 8 - 3 (w/ HAA joints fixed)

min_vel = cfg['environment']['velocity']['min']
max_vel = cfg['environment']['velocity']['max']

wait_period = 200
velocity_period = 500  # sample velocity every 3 sec

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
    local_loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + 1, act_dim)
    local_loaded_graph.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 2500 ## 12 secs

    count = 0

    # initialize logging value
    contact_log = np.zeros((4, max_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
    target_velocity_traj = np.zeros((max_steps, ), dtype=np.float32)
    real_velocity_traj = np.zeros((max_steps, ), dtype=np.float32)

    # initialize exp2 data
    velocity_error_collection = dict()
    torque_collection = []
    power_collection = []

    for step in range(max_steps):
        frame_start = time.time()
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
            
        obs, non_obs = env.observe_logging(False)
        
        obs = np.concatenate([obs, velocity], axis=1, dtype=np.float32)

        with torch.no_grad():
            action_ll = local_loaded_graph.architecture(torch.from_numpy(obs))
            action_ll = action_ll.cpu().detach().numpy()

        reward_ll, dones = env.step(action_ll)
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
    exp_CPG_and_velocity_plotting(update, "raisimGymTorch/exp_result/exp2", max_steps, None, target_velocity_traj, real_velocity_traj, type='baseline')

    env.turn_off_visualization()
    env.reset()

    print("Finished at the maximum visualization steps")
