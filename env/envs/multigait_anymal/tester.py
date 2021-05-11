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


# configuration
# 
# ex) 
# --weight: /home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/data/anymal_locomotion/2021-05-04-20-57-12/full_400.pt
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
act_dim = env.num_acts  # 8 (w/ HAA joints fixed)
target_signal_dim = 8

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
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + target_signal_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 3000 ## 10 secs

    LF_HFE_target = [1, 0]
    RF_HFE_target = [1, np.pi]
    LH_HFE_target = [1, np.pi]
    RH_HFE_target = [1, 0]

    target_signal = []    # [LF_HFE, RF_HFE, LH_HFE, RH_HFE]
    target_signal.extend(LF_HFE_target)
    target_signal.extend(RF_HFE_target)
    target_signal.extend(LH_HFE_target)
    target_signal.extend(RH_HFE_target)
    target_signal = np.asarray(target_signal)
    target_signal = np.broadcast_to(target_signal, (training_num_envs, target_signal.shape[0]))

    for step in range(max_steps):
        time.sleep(0.01)
        obs = env.observe(False)
        obs_and_target = np.concatenate((obs, target_signal), axis=1, dtype=np.float32)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs_and_target))
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    env.turn_off_visualization()
    env.reset()

    print("Finished at the maximum visualization steps")
