import numpy as np
from types import SimpleNamespace as SN
from pathlib import Path
from datetime import datetime

import torch


def path_follow():
    cfg = SN()
    cfg.dt = 0.1
    cfg.max_t = 10.
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))
    cfg.num_train = 20000
    cfg.num_validate = 10
    cfg.interval_validate = 1000

    cfg.quad = SN()
    cfg.quad.tau_speed = 0.1
    cfg.quad.tau_yaw = 0.1
    cfg.quad.init_pos = np.vstack((-8., 5.))
    cfg.quad.init_speed = 0.5
    cfg.quad.init_yaw = -np.pi/4
    cfg.quad.rand_pos = [[-12., 12.], [-8., 8.]]
    cfg.quad.rand_speed = [-15., 15.]
    cfg.quad.rand_yaw = [-np.pi, np.pi]

    cfg.traj = SN()
    cfg.traj.A = 4.
    cfg.traj.split_num  = 354

    cfg.ddpg = SN()
    cfg.ddpg.dim_state = 4
    cfg.ddpg.dim_action = 2
    cfg.ddpg.action_max = np.array([3., np.pi/2])
    cfg.ddpg.action_min = np.array([-3., -np.pi/2])
    cfg.ddpg.action_scaling = torch.from_numpy(cfg.ddpg.action_max).float()
    cfg.ddpg.memory_size = 300000
    cfg.ddpg.actor_lr = 0.0001
    cfg.ddpg.critic_lr = 0.001
    cfg.ddpg.batch_size = 64
    cfg.ddpg.discount = 0.999
    cfg.ddpg.softupdate_rate = 0.001
    cfg.ddpg.terminate_condition = 10
    cfg.ddpg.reward_weight = 2
    cfg.ddpg.reward_max = 21
    cfg.ddpg.reward_scaling = 10
    cfg.ddpg.reward_bias = 11
    # cfg.ddpg.actor_node = 64
    # cfg.ddpg.critic_node = 64
    cfg.ddpg.node_set = [64, 128, 256]

    cfg.noise = SN()
    cfg.noise.rho = 0.01
    cfg.noise.mu = np.zeros(cfg.ddpg.dim_action)
    cfg.noise.sigma = 1 / cfg.ddpg.action_max
    cfg.noise.size = cfg.ddpg.dim_action
    cfg.noise.dt = cfg.dt
    cfg.noise.x0 = None
    cfg.noise.tau = 200
    return cfg

def simple_guidance():
    cfg = SN()
    cfg.dt = 0.1
    cfg.max_t = 10.
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))
    cfg.num_train = 1
    cfg.num_validate = 1
    cfg.interval_validate = 1

    cfg.quad = SN()
    cfg.quad.tau_euler = 0.1
    cfg.quad.tau_thrust = 0.1
    cfg.quad.init_pos = np.vstack((0., 0., 0.))
    cfg.quad.init_vel = np.vstack((0., 0., 0.))
    cfg.quad.init_euler = np.vstack((0., 0., 0.))
    cfg.quad.mass = 1.
    cfg.quad.rand_pos = [[-10., 10.], [-10., 10.], [0., 20.]]
    cfg.quad.rand_vel = [[-3., 3.], [-3., 3.], [3., 3.]]
    cfg.quad.rand_euler = [[-15*np.pi/180, 15*np.pi/180],
                           [-15*np.pi/180, 15*np.pi/180],
                           [-15*np.pi/180, 15*np.pi/180]]
    cfg.quad.goal = np.vstack((0., 0., 10.))
    cfg.quad.cmd_psi = 0

    cfg.ddpg = SN()
    cfg.ddpg.dim_state = 9
    cfg.ddpg.dim_action = 3
    cfg.ddpg.action_max = np.array([15., np.pi, np.pi/6])
    cfg.ddpg.action_min = np.array([0., -np.pi, 0.])
    cfg.ddpg.action_scaling = torch.FloatTensor([15/2, np.pi, np.pi/12])
    cfg.ddpg.action_bias = torch.FloatTensor([1., 0., 1.])
    cfg.ddpg.memory_size = 300000
    cfg.ddpg.actor_lr = 0.0001
    cfg.ddpg.critic_lr = 0.001
    cfg.ddpg.batch_size = 64
    cfg.ddpg.discount = 0.999
    cfg.ddpg.softupdate_rate = 0.001
    cfg.ddpg.node_set = [64, 128, 256]

    cfg.noise = SN()
    cfg.noise.rho = 0.01
    cfg.noise.mu = np.zeros(cfg.ddpg.dim_action)
    cfg.noise.sigma = 1 / cfg.ddpg.action_max
    cfg.noise.size = cfg.ddpg.dim_action
    cfg.noise.dt = cfg.dt
    cfg.noise.x0 = None
    cfg.noise.tau = 500
    return cfg
