import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from collections import deque

import fym.logging as logging


class PostProcessing:
    def __init__(self, cfg):
        self.G = deque(maxlen=int(cfg.num_train/cfg.num_eval))
        self.num_epi = np.arange(
            cfg.num_eval,
            cfg.num_train+cfg.num_eval,
            cfg.num_eval
        )
        self.dir = cfg.dir

    def draw_plot(self, dir_save, dir_save_env):
        env_data, info = logging.load(dir_save_env, with_info=True)
        cfg = info['cfg']

        time = env_data['time']
        cmd_speed = env_data['cmd_speed']
        cmd_yaw = env_data['cmd_yaw']
        action = env_data['action']
        pos = env_data['quad']['pos'].squeeze()
        speed = env_data['quad']['speed']
        yaw = env_data['quad']['yaw']
        theta_min = env_data['theta_min']
        curvature_min = env_data['curvature_min']
        yaw_T_min = env_data['yaw_T_min']
        distance_min = env_data['distance_min']
        obs = env_data['obs']
        reward = env_data['reward']
        G = 0
        for r in reward[::-1]:
            G = r.item() + cfg.ddpg.discount*G
        self.G.append(G)

        traj = np.array(cfg.traj.trajectory)
        breakpoint()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        line1, = ax.plot(pos[:,0], pos[:,1], 'r')
        line2, = ax.plot(traj[:,0], traj[:,1], 'b--')
        ax.legend(handles=(line1, line2), labels=('true', 'des.'))
        ax.set_title("Trajectory")
        ax.set_ylabel("Y [m]")
        ax.set_xlabel("X [m]")
        ax.grid(True)
        fig.savefig(Path(dir_save, "trajectory.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax[0].plot(time, pos[:,0])
        ax[0].set_ylabel("X [m]")
        ax[0].set_title("System State")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, pos[:,1])
        ax[1].set_ylabel("Y [m]")
        ax[1].axes.xaxis.set_ticklabels([])
        line1, = ax[2].plot(time, speed, 'r')
        line2, = ax[2].plot(time, cmd_speed, 'b--')
        ax[2].legend(handles=(line1, line2), labels=('true', 'cmd'))
        ax[2].set_ylabel("V [m/s]")
        ax[2].axes.xaxis.set_ticklabels([])
        ax[3].plot(time, yaw, 'r', time, cmd_yaw, 'b--')
        ax[3].set_ylabel(r'$\psi$ [deg]')
        ax[3].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(4)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "system_state.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time, reward)
        ax.set_title("Reward")
        ax.set_ylabel('r')
        ax.set_xlabel('time [s]')
        ax.grid(True)
        fig.savefig(Path(dir_save, "reward.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax[0].plot(time, obs[:,0])
        ax[0].set_ylabel("$y_T$ [m]")
        ax[0].set_title("DDPG State")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, obs[:,1])
        ax[1].set_ylabel(r"$e_{\psi}$ [m]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, obs[:,2])
        ax[2].set_ylabel("$X_{curv.}$")
        ax[2].axes.xaxis.set_ticklabels([])
        ax[3].plot(time, obs[:,3])
        ax[3].set_ylabel('$Y_{curv.}$')
        ax[3].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(4)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "ddpg_state.png"), bbox_inches='tight')
        plt.close('all')

    def compare_eval(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.num_epi, list(self.G), "*")
        ax.set_title("Return")
        ax.set_ylabel("G")
        ax.set_xlabel("number of trained episode")
        ax.grid(True)
        fig.savefig(Path(self.dir, "return.png"), bbox_inches='tight')
        plt.close('all')




