import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

import fym.logging as logging
from fym.utils import rot


class PostProcessing:
    def __init__(self, cfg):
        self.G_avg = deque(maxlen=int(cfg.num_train/cfg.interval_validate))
        self.G_validate = deque(maxlen=cfg.num_validate)
        self.num_epi = np.arange(
            cfg.interval_validate,
            cfg.num_train+cfg.interval_validate,
            cfg.interval_validate
        )

    def path_follow(self, dir_save, dir_save_env):
        env_data, info = logging.load(dir_save_env, with_info=True)
        cfg = info['cfg']

        time = env_data['time']
        cmd_speed = env_data['cmd_speed']
        cmd_yaw = env_data['cmd_yaw']*180/np.pi
        action = env_data['action']
        pos = env_data['quad']['pos'].squeeze()
        speed = env_data['quad']['speed']
        yaw = env_data['quad']['yaw']*180/np.pi
        theta_min = env_data['theta_min']
        curvature_min = env_data['curvature_min']
        yaw_T_min = env_data['yaw_T_min']*180/np.pi
        distance_min = env_data['distance_min']
        obs = env_data['obs']
        reward = np.delete(env_data['reward'], 0)
        G = 0
        for r in reward[::-1]:
            G = r.item() + cfg.ddpg.discount*G
        G = G / len(reward)
        self.G_validate.append(G)
        # print(f"Return: {G}")

        traj = np.array(cfg.traj.trajectory)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        line1, = ax.plot(pos[:,0], pos[:,1], 'r')
        line2, = ax.plot(traj[:,0], traj[:,1], 'b--')
        ax.legend(handles=(line1, line2), labels=('true', 'des.'))
        ax.plot(pos[0,0], pos[0,1], '*')
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
        ax.plot(time[1::], reward)
        ax.set_title(f"Reward (Return: {int(G)})")
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
        ax[1].set_ylabel(r"$e_{\psi}$ [deg]")
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

        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax[0].plot(time, theta_min)
        ax[0].set_ylabel(r"$\Theta_{min}$")
        ax[0].set_title("Variables related to closest point")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, yaw_T_min)
        ax[1].set_ylabel(r"$\psi_{T,min}$ [deg]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, curvature_min[:,0])
        ax[2].set_ylabel("$X_{curv., min}$")
        ax[2].axes.xaxis.set_ticklabels([])
        ax[3].plot(time, curvature_min[:,1])
        ax[3].set_ylabel('$Y_{curv., min}$')
        ax[3].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(4)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "closest_point.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(time, action[:,0])
        ax[0].set_ylabel(f"$des. acc. [m/s^2]$")
        ax[0].set_title("DDPG Action")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, action[:,1])
        ax[1].set_ylabel(r"$des. \dot{\psi}$ [deg/s]")
        ax[1].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(2)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "ddpg_action.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        line1, = ax.plot(time, yaw, 'r')
        line2, = ax.plot(time, yaw_T_min, 'b--')
        ax.legend(handles=(line1, line2), labels=('true', 'des.'))
        ax.set_ylabel(r"$\psi$ [deg]")
        ax.set_title(" State")
        ax.grid(True)
        ax.set_xlabel("time [s]")
        fig.savefig(Path(dir_save, "yaw_comapre.png"), bbox_inches='tight')
        plt.close('all')

    def simple_guidance(self, dir_save, dir_save_env):
        env_data, info = logging.load(dir_save_env, with_info=True)
        cfg = info['cfg']

        time = env_data['time']
        cmd_euler = env_data['cmd_euler']*180/np.pi
        cmd_f = env_data['cmd_f']
        action = env_data['action']
        disturbance = env_data['disturbance']
        reward = np.delete(env_data['reward'], 0)
        obs = env_data['obs']
        pos = env_data['quad']['pos'].squeeze()
        vel = env_data['quad']['vel'].squeeze()
        euler = env_data['quad']['euler'].squeeze()*180/np.pi
        thrust = env_data['quad']['thrust']
        goal = env_data['goal']

        G = 0
        for r in reward[::-1]:
            G = r.item() + cfg.ddpg.discount*G
        self.G_validate.append(G)
        # print(f"Return: {G}")

        fig, ax = plt.subplots(nrows=3, ncols=1)
        line1, = ax[0].plot(time, pos[:,0], 'r')
        line2, = ax[0].plot(time, goal[:,0], 'b--')
        ax[0].legend(handles=(line1, line2), labels=('true', 'goal'))
        ax[0].set_ylabel("X [m]")
        ax[0].set_title("Position")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, pos[:,1], 'r', time, goal[:,1], 'b--')
        ax[1].set_ylabel("Y [m]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, pos[:,2], 'r', time, goal[:,2], 'b--')
        ax[2].set_ylabel('Z [m]')
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "position.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, vel[:,0])
        ax[0].set_ylabel("$V_x$ [m/s]")
        ax[0].set_title("Velocity")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, vel[:,1])
        ax[1].set_ylabel("$V_y$ [m/s]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, vel[:,2])
        ax[2].set_ylabel('$V_z$ [m/s]')
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "velocity.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        line1, = ax[0].plot(time, euler[:,0], 'r')
        line2, = ax[0].plot(time, cmd_euler[:,0], 'b--')
        ax[0].legend(handles=(line1, line2), labels=('true', 'cmd.'))
        ax[0].set_ylabel(r"$\phi$ [deg]")
        ax[0].set_title("Euler angle")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, euler[:,1], 'r', time, cmd_euler[:,1], 'b--')
        ax[1].set_ylabel(r"$\theta$ [deg]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, euler[:,2], 'r', time, cmd_euler[:,2], 'b--')
        ax[2].set_ylabel(r'$\psi$ [deg]')
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "euler.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        line1, = ax[0].plot(time, thrust, 'r')
        line2, = ax[0].plot(time, cmd_f, 'b--')
        ax[0].legend(handles=(line1, line2), labels=('true', 'cmd.'))
        ax[0].set_ylabel("Thrust [N]")
        ax[0].set_title("Action")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, action[:,1]*180/np.pi)
        ax[1].set_ylabel("azimuth [deg]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, action[:,2]*180/np.pi)
        ax[2].set_ylabel('elevation [deg]')
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "action.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, obs[:,0])
        ax[0].set_ylabel("$e_X$ [m]")
        ax[0].set_title("Position error")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, obs[:,1])
        ax[1].set_ylabel("$e_Y$ [m]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, obs[:,2])
        ax[2].set_ylabel('$e_Z$ [m]')
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "pos_error.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, disturbance[:,0])
        ax[0].set_ylabel("X [N]")
        ax[0].set_title("Disturbance")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, disturbance[:,1])
        ax[1].set_ylabel("Y [N]")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, disturbance[:,2])
        ax[2].set_ylabel('Z [N]')
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "disturbance.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time[1::], reward)
        ax.set_title(f"Reward (Return: {int(G)})")
        ax.set_ylabel('r')
        ax.set_xlabel('time [s]')
        ax.grid(True)
        fig.savefig(Path(dir_save, "reward.png"), bbox_inches='tight')
        plt.close('all')

        fig, _ = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
        ani = Animator(fig, [env_data])
        ani.animate()
        ani.save(Path(dir_save, "animation.mp4"))
        plt.close('all')

    def pointmass_linear(self, dir_save, dir_save_env):
        env_data, info = logging.load(dir_save_env, with_info=True)
        cfg = info['cfg']

        time = env_data['time']
        action = env_data['action']
        reward = env_data['reward']
        u = env_data['u']
        pos = env_data['quad']['pos'].squeeze()
        vel = env_data['quad']['vel'].squeeze()
        LQR_lambda = env_data['LQR_lambda']
        LQR_u = env_data['LQR_u']
        LQR_P = env_data['LQR_P']

        G = 0
        for r in reward[::-1]:
            G = r.item() + cfg.ddpg.discount*G
        self.G_validate.append(G)
        # print(f"Return: {G}")

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(pos[:,0], pos[:,1])
        ax.set_ylabel('Y [m]')
        ax.set_xlabel('X [m]')
        ax.set_title('Trajectory')
        ax.grid(True)
        fig.savefig(Path(dir_save, "trajectory.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(time, pos[:,0])
        ax[0].set_ylabel("X [m]")
        ax[0].set_title("Position")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, pos[:,1])
        ax[1].set_ylabel("Y [m]")
        ax[1].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(2)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "position.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(time, vel[:,0])
        ax[0].set_ylabel("$V_x$ [m/s]")
        ax[0].set_title("Velocity")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, vel[:,1])
        ax[1].set_ylabel("$V_y$ [m/s]")
        ax[1].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(2)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "velocity.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=4, ncols=1)
        line1, = ax[0].plot(time, action[:,0], 'r')
        line2, = ax[0].plot(time, LQR_lambda[:,0], 'b--')
        ax[0].legend(handles=(line1, line2), labels=('RL', 'LQR'))
        ax[0].set_ylabel(r"$\lambda_1$")
        ax[0].set_title("Costate")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, action[:,1], 'r', time, LQR_lambda[:,1], 'b--')
        ax[1].set_ylabel(r"$\lambda_2$")
        ax[1].axes.xaxis.set_ticklabels([])
        ax[2].plot(time, action[:,2], 'r', time, LQR_lambda[:,2], 'b--')
        ax[2].set_ylabel(r"$\lambda_3$")
        ax[2].axes.xaxis.set_ticklabels([])
        ax[3].plot(time, action[:,3], 'r', time, LQR_lambda[:,3], 'b--')
        ax[3].set_ylabel(r'$\lambda_4$')
        ax[3].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(4)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "costate.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=2, ncols=1)
        line1, = ax[0].plot(time, u[:,0], 'r')
        line2, = ax[0].plot(time, LQR_u[:,0], 'b--')
        ax[0].legend(handles=(line1, line2), labels=('RL', 'LQR'))
        ax[0].set_ylabel(r"$u_x$ [N]")
        ax[0].set_title("Input Force")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, u[:,1], 'r', time, LQR_u[:,1], 'b--')
        ax[1].set_ylabel(r'$u_y$ [N]')
        ax[1].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(2)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "input.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=4, ncols=4)
        for i in range(16):
            ax[i//4, i%4].plot(time, LQR_P[:, i//4, i%4])
            ax[i//4, i%4].grid(True)
        fig.savefig(Path(dir_save, "LQR_P.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time, reward)
        ax.set_title(f"Reward (Return: {int(G)})")
        ax.set_ylabel('r')
        ax.set_xlabel('time [s]')
        ax.grid(True)
        fig.savefig(Path(dir_save, "reward.png"), bbox_inches='tight')
        plt.close('all')

    def pendulum(self, dir_save, dir_save_env):
        env_data, info = logging.load(dir_save_env, with_info=True)
        cfg = info['cfg']

        time = env_data['time']
        action = env_data['action']
        reward = env_data['reward'].squeeze()
        theta = env_data['pendulum']['th'].squeeze()
        thetadot = env_data['pendulum']['thdot'].squeeze()

        G = 0
        for r in reward[::-1]:
            G = r.item() + cfg.ddpg.discount*G
        self.G_validate.append(G)
        # print(f"Return: {G}")

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(time, theta*180/np.pi)
        ax[0].set_ylabel(r"$\theta$ [deg]")
        ax[0].set_title("State")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].plot(time, thetadot*180/np.pi)
        ax[1].set_ylabel(r"$\dot{\theta}$ [deg/s]")
        ax[1].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(2)]
        fig.align_ylabels(ax)
        fig.savefig(Path(dir_save, "state.png"), bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time, reward)
        ax.set_title(f"Reward (Return: {int(G)})")
        ax.set_ylabel('r')
        ax.set_xlabel('time [s]')
        ax.grid(True)
        fig.savefig(Path(dir_save, "reward.png"), bbox_inches='tight')
        plt.close('all')

    def compare_validate(self, dir_save):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.num_epi, list(self.G_avg), "*")
        ax.set_title("Average Return")
        ax.set_ylabel("G")
        ax.set_xlabel("number of trained episode")
        ax.grid(True)
        fig.savefig(Path(dir_save, "return.png"), bbox_inches='tight')
        plt.close('all')

    def average_return(self):
        self.G_avg.append(np.array(self.G_validate).mean())


class Quad_ani:
    def __init__(self, ax):
        d = 1.
        r = 0.5

        body_segs = np.array([
            [[d, 0, 0], [0, 0, 0]],
            [[-d, 0, 0], [0, 0, 0]],
            [[0, d, 0], [0, 0, 0]],
            [[0, -d, 0], [0, 0, 0]]
        ])
        colors = (
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
        )
        self.body = art3d.Line3DCollection(
            body_segs,
            colors=colors,
            linewidths=2
        )

        kwargs = dict(radius=r, ec="k", fc="k", alpha=0.3)
        self.rotors = [
            Circle((d, 0), **kwargs),
            Circle((0, d), **kwargs),
            Circle((-d, 0), **kwargs),
            Circle((0, -d), **kwargs),
        ]

        ax.add_collection3d(self.body)
        for rotor in self.rotors:
            ax.add_patch(rotor)
            art3d.pathpatch_2d_to_3d(rotor, z=0)

        self.body._base = self.body._segments3d
        for rotor in self.rotors:
            rotor._segment3d = np.array(rotor._segment3d)
            rotor._center = np.array(rotor._center + (0,))
            rotor._base = rotor._segment3d

    def set(self, pos, dcm=np.eye(3)):
        self.body._segments3d = np.array([
            dcm @ point for point in self.body._base.reshape(-1, 3)
        ]).reshape(self.body._base.shape)

        for rotor in self.rotors:
            rotor._segment3d = np.array([
                dcm @ point for point in rotor._base
            ])

        self.body._segments3d = self.body._segments3d + pos
        for rotor in self.rotors:
            rotor._segment3d += pos


class Animator:
    def __init__(self, fig, datalist):
        self.offsets = ['collections', 'patches', 'lines', 'texts',
                        'artists', 'images']
        self.fig = fig
        self.axes = fig.axes
        self.datalist  = datalist

    def init(self):
        self.frame_artists = []
        for ax in self.axes:
            ax.quad = Quad_ani(ax)

            ax.set_xlim3d(-10, 10)
            ax.set_ylim3d(-10, 10)
            ax.set_zlim3d(-10, 10)

            for name in self.offsets:
                self.frame_artists += getattr(ax, name)

        self.fig.tight_layout()
        return self.frame_artists

    def get_sample(self, frame):
        self.init()
        self.update(frame)
        self.fig.show()

    def update(self, frame):
        for data, ax in zip(self.datalist, self.axes):
            pos = data['quad']['pos'][frame].squeeze()
            dcm = data['dcm'][frame].squeeze()
            ax.quad.set(pos, dcm)
        return self.frame_artists

    def animate(self, *args, **kwargs):
        data_len = len(self.datalist[0]['time'])
        frames = range(0, data_len, 1)
        self.anim = FuncAnimation(
            self.fig, self.update, init_func=self.init,
            frames=frames, interval=10, blit=True,
            *args, **kwargs
        )

    def save(self, path, *args, **kwargs):
        self.anim.save(path, writer="ffmpeg", fps=30, *args, **kwargs)


