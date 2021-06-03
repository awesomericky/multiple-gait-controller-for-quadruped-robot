import numpy as np
import matplotlib.pyplot as plt
from os.path import isdir
from os import makedirs

def check_saving_folder(folder_name):
    if not isdir(folder_name):
        makedirs(folder_name)

def joint_angle_plotting(update, folder_name, t_range, CPG_signal, \
                         FR_thigh_joint_history, FL_thigh_joint_history, RR_thigh_joint_history, RL_thigh_joint_history,\
                         FR_calf_joint_history, FL_calf_joint_history, RR_calf_joint_history, RL_calf_joint_history):

    final_folder_name = f'joint_plot/{folder_name}'
    check_saving_folder(final_folder_name)

    fig, ax = plt.subplots(2,2,figsize=(28, 15))

    # FR_thigh
    ax[0, 0].plot(t_range, FR_thigh_joint_history, 'o', label='joint angle [rad]')
    ax[0, 0].plot(t_range, CPG_signal[0], label='signal')
    ax[0, 0].set_xlabel('time [s]', fontsize=20)
    ax[0, 0].set_title('FR', fontsize=25)

    # FL_thigh
    ax[0, 1].plot(t_range, FL_thigh_joint_history, 'o', label='joint angle [rad]')
    ax[0, 1].plot(t_range, CPG_signal[1], label='signal')
    ax[0, 1].set_xlabel('time [s]', fontsize=20)
    ax[0, 1].set_title('FL', fontsize=25)

    # RR_thigh
    ax[1, 0].plot(t_range, RR_thigh_joint_history, 'o', label='joint angle [rad]')
    ax[1, 0].plot(t_range, CPG_signal[2], label='signal')
    ax[1, 0].set_xlabel('time [s]', fontsize=20)
    ax[1, 0].set_title('RR', fontsize=25)

    # RL_thigh
    ax[1, 1].plot(t_range, RL_thigh_joint_history, 'o', label='joint angle [rad]')
    ax[1, 1].plot(t_range, CPG_signal[3], label='signal')
    ax[1, 1].set_xlabel('time [s]', fontsize=20)
    ax[1, 1].set_title('RL', fontsize=25)

    plt.legend()
    plt.savefig(f'{final_folder_name}/Thigh_joint_angle_{update}.png')
    plt.close()


    fig, ax = plt.subplots(2,2,figsize=(28,15))

    # FR_calf
    ax[0, 0].plot(t_range, FR_calf_joint_history, 'o', label='joint angle [rad]')
    ax[0, 0].set_title('FR', fontsize=25)
    ax[0, 0].set_xlabel('time [s]', fontsize=20)

    # FL_calf
    ax[0, 1].plot(t_range, FL_calf_joint_history, 'o', label='joint angle [rad]')
    ax[0, 1].set_title('FL', fontsize=25)
    ax[0, 1].set_xlabel('time [s]', fontsize=20)

    # RR_calf
    ax[1, 0].plot(t_range, RR_calf_joint_history, 'o', label='joint angle [rad]')
    ax[1, 0].set_title('RR', fontsize=25)
    ax[1, 0].set_xlabel('time [s]', fontsize=20)

    # RL_calf
    ax[1, 1].plot(t_range, RL_calf_joint_history, 'o', label='joint angle [rad]')
    ax[1, 1].set_title('RL', fontsize=25)
    ax[1, 1].set_xlabel('time [s]', fontsize=20)

    plt.legend()
    plt.savefig(f'{final_folder_name}/Calf_joint_angle_{update}.png')
    plt.close()

def contact_plotting(update, folder_name, contact_log):
    final_folder_name = f'contact_plot/{folder_name}'
    check_saving_folder(final_folder_name)

    start = 100
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
    ax.set_title('contact', fontsize=20)
    ax.set_xlabel('time [s]')
    plt.savefig(f'{final_folder_name}/contact_{update}.png')
    plt.close()

def CPG_and_velocity_plotting(update, folder_name, n_steps, CPG_signal_period_traj, target_velocity_traj, real_velocity_traj):
    final_folder_name = f'CPG_and_velocity_plot/{folder_name}'
    check_saving_folder(final_folder_name)

    time_axis = np.arange(n_steps) * 0.01
    plt.plot(time_axis, CPG_signal_period_traj, label='CPG_period')
    plt.plot(time_axis, target_velocity_traj, label='target_velocity')
    plt.plot(time_axis, real_velocity_traj, label='real_velocity')
    plt.title('CPG and velocity', fontsize=20)
    plt.xlabel('time [s]')
    plt.legend()
    plt.savefig(f'{final_folder_name}/CPG_vel_{update}.png')
    plt.close()

check_saving_folder('contact_plot/ABC')