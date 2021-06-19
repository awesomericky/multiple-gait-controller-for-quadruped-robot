import numpy as np
import matplotlib.pyplot as plt
import pickle

gaits = ['baseline', 'hierarchy']
color = ['black', 'red']

# Plot linear velocity error
for k, gait in enumerate(gaits):
    with open(f"raisimGymTorch/exp_result/exp2/vel_error_{gait}.pkl", "rb") as f:
        velocity_error_collection = pickle.load(f)
        command_vel = list(velocity_error_collection.keys())
        vel_error_mean = np.array(list(velocity_error_collection.values()))[:, 0]
        # vel_error_std = np.array(list(velocity_error_collection.values()))[:, 1]
        plt.plot(command_vel, vel_error_mean, color=color[k], label=gait)
plt.xlabel('Command velocity [m/s]')
plt.ylabel('Velocity error [m/s]')
plt.legend()
plt.grid(True)
plt.savefig('raisimGymTorch/exp_result/exp2/velocity_error.png')
plt.clf()
plt.close()

# Plot torque
for k, gait in enumerate(gaits):
    torque_collection = np.load(f"raisimGymTorch/exp_result/exp2/torque_{gait}.npz")['torque']

    range_1 = []; range_2 = []; range_3 = []; range_4 = []; range_5 = []; range_6 = []; range_7 = []; range_8 = [];

    for i in range(torque_collection.shape[0]):
        vel, torque = torque_collection[i]
        if vel < 0.2:
            range_1.append(torque)
        elif vel < 0.4:
            range_2.append(torque)
        elif vel < 0.6:
            range_3.append(torque)
        elif vel < 0.8:
            range_4.append(torque)
        elif vel < 1.0:
            range_5.append(torque)
        elif vel < 1.2:
            range_6.append(torque)
        elif vel < 1.4:
            range_7.append(torque)
        else:
            range_8.append(torque)

    total_range = [range_1, range_2, range_3, range_4, range_5, range_6, range_7, range_8]
    candidate_vel = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    torque_mean = []
    torque_std = []

    for i, single_range in enumerate(total_range):
        if len(single_range) == 0:
            candidate_vel[i] = None
        else:
            torque_mean.append(np.mean(single_range))
            torque_std.append(np.std(single_range))

    candidate_vel = list(set(candidate_vel))
    if None in candidate_vel:
        candidate_vel.remove(None)
    candidate_vel = sorted(candidate_vel)
    plt.plot(candidate_vel, torque_mean, color=color[k], label=gait)
plt.xlabel('Measured velocity [m/s]')
plt.ylabel('Torque [Nm]')
plt.legend()
plt.grid(True)
plt.savefig('raisimGymTorch/exp_result/exp2/torque.png')
plt.clf()
plt.close()

# Plot Energy
for k, gait in enumerate(gaits):
    power_collection = np.load(f"raisimGymTorch/exp_result/exp2/power_{gait}.npz")['power']

    range_1 = []; range_2 = []; range_3 = []; range_4 = []; range_5 = []; range_6 = []; range_7 = []; range_8 = [];

    for i in range(power_collection.shape[0]):
        vel, power = power_collection[i]
        if vel < 0.2:
            range_1.append(power)
        elif vel < 0.4:
            range_2.append(power)
        elif vel < 0.6:
            range_3.append(power)
        elif vel < 0.8:
            range_4.append(power)
        elif vel < 1.0:
            range_5.append(power)
        elif vel < 1.2:
            range_6.append(power)
        elif vel < 1.4:
            range_7.append(power)
        else:
            range_8.append(power)

    total_range = [range_1, range_2, range_3, range_4, range_5, range_6, range_7, range_8]
    candidate_vel = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    power_mean = []
    power_std = []

    for i, single_range in enumerate(total_range):
        if len(single_range) == 0:
            candidate_vel[i] = None
        else:
            power_mean.append(np.mean(single_range))
            power_std.append(np.std(single_range))

    candidate_vel = list(set(candidate_vel))
    if None in candidate_vel:
        candidate_vel.remove(None)
    candidate_vel = sorted(candidate_vel)
    plt.plot(candidate_vel, power_mean, color=color[k], label=gait)
plt.xlabel('Measured velocity [m/s]')
plt.ylabel('Energy [W]')
plt.legend()
plt.grid(True)
plt.savefig('raisimGymTorch/exp_result/exp2/energy.png')
plt.clf()
plt.close()