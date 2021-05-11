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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pdb


"""
# TODO

1. error of sin regression should be the closest point to either 0 or 2*pi (ex) if value is 2pi-1, then the mse should be calculated with 2pi, not 0
2. check whether value function is being updated
3. check curve fit model. Is it correct? could we improve it by bounding area or initialize better?
4. isn't there a way to not use curve fit model and just use other matric for reward?
"""

"""
# TODO

1. reward shaping: 1) low height (stability), 2) no orientation change 3) GRF entropy maximize
+) how coulb the 4 legs have similar load (currently just front two legs get the whole load)
"""


def sin(x, a, b, c, d):
    return a* np.sin(b*x + c) + d

def shift_sin_param(param1, param2, mean_param):
    new_param = (param2 - param1)/mean_param
    new_param -= (new_param // (2*np.pi)) * (2*np.pi)
    assert (0 <= new_param).all() and (new_param < 2*np.pi).all()
    return new_param

# def shift_sin_param(fix_param, move_param):
#     fix_param_0 = fix_param[:, 0]
#     fix_param_1 = fix_param[:, 1]
#     move_param_0 = move_param[:, 0]
#     move_param_1 = move_param[:, 1]
#     shift_param_0 = move_param_0 / (fix_param_0 + 1e-4)
#     shift_param_1 = move_param_1 - (fix_param_1 / (fix_param_0 + 1e-4)) * move_param_0
#     shift_param = np.concatenate((shift_param_0[:, np.newaxis], shift_param_1[:, np.newaxis]), axis=1)

#     return shift_param

# task specification
task_name = "multigait_anymal"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
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
env = VecEnv(multigait_anymal.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
act_dim = env.num_acts + 4  # 12 (w/ HAA joints fixed)
act_dim = env.num_acts
target_signal_dim = 0

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + target_signal_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0), #1.0
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

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
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

DESIRED_VELOCITY = cfg['environment']['velocity'] # m/s

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
period = .7 # [s]
period_param = 2 * np.pi / period  # period: 
LF_HFE_target = [1, np.pi]
RF_HFE_target = [1, 0]
LH_HFE_target = [1, 0]
RH_HFE_target = [1, np.pi]
target_signal[0] = sin(t_range, 1, period_param * LF_HFE_target[0], LF_HFE_target[1], 0.5)
target_signal[1] = sin(t_range, 1, period_param * RF_HFE_target[0], RF_HFE_target[1], 0.5)
target_signal[2] = sin(t_range, 1, period_param * LH_HFE_target[0], LH_HFE_target[1], 0.5)
target_signal[3] = sin(t_range, 1, period_param * RH_HFE_target[0], RH_HFE_target[1], 0.5)

env_action = np.zeros((cfg['environment']['num_envs'], 8), dtype=np.float32)

for update in range(1000000):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    
    # LF_HFE_history = []
    # RF_HFE_history = []
    # LH_HFE_history = []
    # RH_HFE_history = []

    # Target
    # Trot
    # LF_HFE_target = [1, 0]
    # RF_HFE_target = [1, np.pi]
    # LH_HFE_target = [1, np.pi]
    # RH_HFE_target = [1, 0]

    # Pace
    # LF_HFE_target = [1, 0]
    # RF_HFE_target = [1, np.pi]
    # LH_HFE_target = [1, 0]
    # RH_HFE_target = [1, np.pi]

    # target_signal = []    # [LF_HFE, RF_HFE, LH_HFE, RH_HFE]
    # target_signal.extend(LF_HFE_target)
    # target_signal.extend(RF_HFE_target)
    # target_signal.extend(LH_HFE_target)
    # target_signal.extend(RH_HFE_target)
    # target_signal = np.asarray(target_signal)
    # target_signal = np.broadcast_to(target_signal, (cfg['environment']['num_envs'], target_signal.shape[0]))
    """
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

        for step in range(n_steps*2):
            frame_start = time.time()
            obs = env.observe(False)
            # obs_and_target = np.concatenate((obs, target_signal), axis=1, dtype=np.float32)
            action_ll = loaded_graph.architecture(torch.from_numpy(obs))
            action_ll = action_ll.cpu().detach().numpy()
            env_action[:, [0, 2, 4, 6]] = action_ll[:, :4] * target_signal[:, step] + action_ll[:, 4:8]
            env_action[:, [1, 3, 5, 7]] = action_ll[:, 8:]
            reward_ll, dones = env.step(env_action)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))
    """
    # actual training
    for step in range(n_steps):
        obs = env.observe()
        # obs_and_target = np.concatenate((obs, target_signal), axis=1, dtype=np.float32)
        action = ppo.observe(obs)

        # env_action[:, [0, 2, 4, 6]] = action[:, 0][:, np.newaxis] * target_signal[:, step] + action[:, 1][:, np.newaxis]
        # env_action[:, [1, 3, 5, 7]] = action[:, 2:]

        # env_action[:, [0, 2, 4, 6]] = action[:, :4] * target_signal[:, step] + action[:, 4:8]
        # env_action[:, [1, 3, 5, 7]] = action[:, 8:]

        # action[:, [8, 10, 12, 14]] = action[:, :4] * target_signal[:, step] + action[:, 4:8] + action[:, [8, 10, 12, 14]]
        # action[:, [0, 2, 4, 6]] = np.tile(A[:, np.newaxis], (1, 4)) * target_signal[:, step] + action[:, [0, 2, 4, 6]]
        
        # reward, dones = env.step(env_action)
        reward, dones = env.step(action)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)

        if step % 50 == 0:
            env.reward_logging()
            ppo.extra_log(env.reward_log, update*n_steps + step)

        # LF_HFE_history.append(obs[:, 4])
        # RF_HFE_history.append(obs[:, 7])
        # LH_HFE_history.append(obs[:, 10])
        # RH_HFE_history.append(obs[:, 13])

    # Fitting sine model
    # t_range = np.arange(n_steps) * cfg['environment']['control_dt']
    # LF_HFE_history = np.asarray(LF_HFE_history).T  # (n_env, n_steps)
    # RF_HFE_history = np.asarray(RF_HFE_history).T
    # LH_HFE_history = np.asarray(LH_HFE_history).T
    # RH_HFE_history = np.asarray(RH_HFE_history).T

    # sin_fitting_loss = 0

    # LF_HFE_param = np.zeros((cfg['environment']['num_envs'], 2))
    # LF_HFE_param_cov = np.zeros((cfg['environment']['num_envs'], 2))
    # RF_HFE_param = np.zeros((cfg['environment']['num_envs'], 2))
    # RF_HFE_param_cov = np.zeros((cfg['environment']['num_envs'], 2))
    # LH_HFE_param = np.zeros((cfg['environment']['num_envs'], 2))
    # LH_HFE_param_cov = np.zeros((cfg['environment']['num_envs'], 2))
    # RH_HFE_param = np.zeros((cfg['environment']['num_envs'], 2))
    # RH_HFE_param_cov = np.zeros((cfg['environment']['num_envs'], 2))

    # for i in range(cfg['environment']['num_envs']):
    #     try:
    #         param, param_cov = curve_fit(sin, t_range, LF_HFE_history[i], p0=[1, 2*np.pi/DESIRED_VELOCITY, 0, 0])
    #         LF_HFE_param[i] = param[1:3]
    #         LF_HFE_param_cov[i] = np.diag(param_cov)[1:3]

    #         """
    #         # Plot Observation & Curve fit model

    #         plt.plot(t_range, LF_HFE_history[i], 'o')
    #         predict = sin(t_range, param[0], param[1], param[2], param[3])
    #         plt.plot(t_range, predict)
    #         plt.show()
    #         pdb.set_trace()
    #         plt.show()
    #         """
    #     except:
    #         print('curve fit error (LF_HFE)')
    #     try:
    #         param, param_cov = curve_fit(sin, t_range, RF_HFE_history[i], p0=[1, 2*np.pi/DESIRED_VELOCITY, 0, 0])
    #         RF_HFE_param[i] = param[1:3]
    #         RF_HFE_param_cov[i] = np.diag(param_cov)[1:3]
    #     except:
    #         print('curve fit error (RF_HFE)')
    #     try:
    #         param, param_cov = curve_fit(sin, t_range, LH_HFE_history[i], p0=[1, 2*np.pi/DESIRED_VELOCITY, 0, 0])
    #         LH_HFE_param[i] = param[1:3]
    #         LH_HFE_param_cov[i] = np.diag(param_cov)[1:3]
    #     except:
    #         print('curve fit error (LH_HFE)')
    #     try:
    #         param, param_cov = curve_fit(sin, t_range, RH_HFE_history[i], p0=[1, 2*np.pi/DESIRED_VELOCITY, 0, 0])
    #         RH_HFE_param[i] = param[1:3]
    #         RH_HFE_param_cov[i] = np.diag(param_cov)[1:3]
    #     except:
    #         print('curve fit error (RH_HFE)')
    
    # signal_freq = np.concatenate((LF_HFE_param[:, 0][:, np.newaxis], RF_HFE_param[:, 0][:, np.newaxis], \
    #                             LH_HFE_param[:, 0][:, np.newaxis], RH_HFE_param[:, 0][:, np.newaxis]), axis=1)
    # signal_freq_mean = np.mean(signal_freq, axis=1)
    # signal_freq_std = np.std(signal_freq, axis=1)

    # RF_HFE_param = shift_sin_param(RF_HFE_param[:, 1], LF_HFE_param[:, 1], signal_freq_mean)
    # LH_HFE_param = shift_sin_param(LH_HFE_param[:, 1], LF_HFE_param[:, 1], signal_freq_mean)
    # RH_HFE_param = shift_sin_param(RH_HFE_param[:, 1], LF_HFE_param[:, 1], signal_freq_mean)
    
    # sin_fitting_loss -= (RF_HFE_param - RF_HFE_target[-1])**2
    # sin_fitting_loss -= (LH_HFE_param - LH_HFE_target[-1])**2
    # sin_fitting_loss -= (RH_HFE_param - RH_HFE_target[-1])**2
    # sin_fitting_loss -= signal_freq_std
    # # sin_fitting_loss *= 10

    """

    print(f"[RF_HFE] True: ({RF_HFE_target[0]}, {RF_HFE_target[1]}) | Predict: ({RF_HFE_param[0, 0]}, {RF_HFE_param[0, 0]}) ")
    print(f"[LH_HFE] True: ({LH_HFE_target[0]}, {LH_HFE_target[1]}) | Predict: ({LH_HFE_param[0, 0]}, {LH_HFE_param[0, 0]}) ")
    print(f"[RF_HFE] True: ({RH_HFE_target[0]}, {RH_HFE_target[1]}) | Predict: ({RH_HFE_param[0, 0]}, {RH_HFE_param[0, 0]}) ")

    x_range = np.linspace(0, 30, 300)
    true_RF_HFE = sin(x_range, 1, RF_HFE_target[0], RF_HFE_target[1], 0)
    predict_RF_HFE = sin(x_range, 1, RF_HFE_param[0, 0], RF_HFE_param[0, 1], 0)
    true_LH_HFE = sin(x_range, 1, LH_HFE_target[0], LH_HFE_target[1], 0)
    predict_LH_HFE = sin(x_range, 1, LH_HFE_param[0, 0], LH_HFE_param[0, 1], 0)
    true_RH_HFE = sin(x_range, 1, RH_HFE_target[0], RH_HFE_target[1], 0)
    predict_RH_HFE = sin(x_range, 1, RH_HFE_param[0, 0], RH_HFE_param[0, 1], 0)

    fig = plt.figure()
    ax0 = fig.add_subplot(3, 1, 1)
    ax0.plot(x_range, true_RF_HFE, label='true')
    ax0.plot(x_range, predict_RF_HFE, label='predict')
    ax0.set_title('RF_HFE')
    ax1 = fig.add_subplot(3, 1, 2)
    ax1.plot(x_range, true_LH_HFE, label='true')
    ax1.plot(x_range, predict_LH_HFE, label='predict')
    ax1.set_title('LH_HFE')
    ax2 = fig.add_subplot(3, 1, 3)
    ax2.plot(x_range, true_RH_HFE, label='true')
    ax2.plot(x_range, predict_RH_HFE, label='predict')
    ax2.set_title('RH_HFE')
    plt.legend()
    plt.show()
    pdb.set_trace()
    plt.close()
    """

    # take st step to get value obs
    obs = env.observe()
    # obs_and_target = np.concatenate((obs, target_signal), axis=1, dtype=np.float32)
    # ppo.update(actor_obs=obs_and_target, value_obs=obs, log_this_iteration=update % 10 == 0, update=update, auxilory_value=sin_fitting_loss[:, np.newaxis])
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    # print('{:<40} {:>6}'.format("sin fitting reward: ", '{:0.10f}'.format(np.mean(sin_fitting_loss))))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
