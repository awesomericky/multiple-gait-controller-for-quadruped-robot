import matplotlib.pyplot as plt


def joint_angle_plotting(update, t_range, CPG_signal, \
                         FR_thigh_joint_history, FL_thigh_joint_history, RR_thigh_joint_history, RL_thigh_joint_history,\
                         FR_calf_joint_history, FL_calf_joint_history, RR_calf_joint_history, RL_calf_joint_history):

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
    plt.savefig(f'joint_plot/Thigh_joint_angle_{update}.png')
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
    plt.savefig(f'joint_plot/Calf_joint_angle_{update}.png')
    plt.close()

def contact_plotting(update, contact_log):
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