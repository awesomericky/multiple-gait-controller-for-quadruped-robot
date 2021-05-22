from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import multigait_anymal
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb


# configuration
# 
# ex) 
# --weight: /home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/data/anymal_locomotion/2021-05-04-20-57-12/full_400.pt

def sin(x, a, b, c, d):
    return a * np.sin(b*x + c) + d


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
training_num_envs = cfg['environment']['num_envs']
cfg['environment']['num_envs'] = 1

env = VecEnv(multigait_anymal.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
act_dim = env.num_acts - 2  # 8 - 2 (w/ HAA joints fixed)
CPG_signal_dim = 0

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + CPG_signal_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 3000 ## 10 secs

    t_range = np.arange(max_steps) * cfg['environment']['control_dt']
    CPG_signal = np.zeros((4, max_steps))
    period = 0.7  # [s]
    period_param = 2 * np.pi / period  # period:
    FR_target = np.pi
    FL_target = 0
    RR_target = 0
    RL_target = np.pi
    CPG_signal[0] = sin(t_range, 1, period_param, FR_target, 0.0)
    CPG_signal[1] = sin(t_range, 1, period_param, FL_target, 0.0)
    CPG_signal[2] = sin(t_range, 1, period_param, RR_target, 0.0)
    CPG_signal[3] = sin(t_range, 1, period_param, RL_target, 0.0)

    env_action = np.zeros((training_num_envs, 8), dtype=np.float32)
    contact_log = np.zeros((4, max_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL


    for step in range(max_steps):
        time.sleep(0.01)
        obs = env.observe(False)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs))
        action_ll[:, 0] = torch.relu(action_ll[:, 0])
        action_ll = action_ll.cpu().detach().numpy()
        env_action[:, [0, 2, 4, 6]] = action_ll[:, 0][:, np.newaxis] * CPG_signal[:, step] + action_ll[:, 1][:, np.newaxis]
        env_action[:, [1, 3, 5, 7]] = action_ll[:, 2:]
        reward_ll, dones = env.step(env_action)
        reward_ll_sum = reward_ll_sum + reward_ll[0]

        # contact logging
        env.contact_logging()
        contact_log[:, step] = env.contact_log[0, :]

        # if step % 10 == 0:
            # pdb.set_trace()

        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0
    
    # save & plot contact log
    np.savez_compressed(f'contact_plot/contact_test.npz', contact=contact_log)

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
    plt.savefig(f'contact_plot/contact_test.png')
    plt.close()

    env.turn_off_visualization()
    env.reset()

    print("Finished at the maximum visualization steps")
