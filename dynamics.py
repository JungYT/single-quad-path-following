import numpy as np
import numpy.random as random

from fym.core import BaseEnv, BaseSystem
import fym.core as core
from fym.utils import rot

class PointMass2D(BaseEnv):
    def __init__(self, cfg):
        super().__init__()
        self.pos = BaseSystem(cfg.init_pos)
        self.speed = BaseSystem(cfg.init_speed)
        self.yaw = BaseSystem(cfg.init_yaw)
        self.tau_speed = cfg.tau_speed
        self.tau_yaw = cfg.tau_yaw

    def set_dot(self, cmd_speed, cmd_yaw):
        speed = self.speed.state
        yaw = self.yaw.state
        self.pos.dot = np.vstack((speed*np.cos(yaw), speed*np.sin(yaw)))
        self.speed.dot = -speed/self.tau_speed + cmd_speed/self.tau_speed
        self.yaw.dot = -yaw/self.tau_yaw + cmd_yaw/self.tau_yaw

class PointMass2DPathFollowing(BaseEnv):
    def __init__(self, cfg):
        super().__init__(dt=cfg.dt, max_t=cfg.max_t)
        self.quad = PointMass2D(cfg.quad)
        self.cfg = cfg
        self.trajectory, self.theta_set, self.curvature_set, self.yaw_T_set = \
            self.make_trajectory()
        self.theta_min = None
        self.curvature_min = None
        self.yaw_T_min = None
        self.reward = None
        self.distance_min = None
        self.obs = None

    def reset(self, random_init=True):
        super().reset()
        if random_init:
            self.quad.pos.state = np.vstack((
                random.uniform(
                    low=self.cfg.quad.rand_pos[0][0],
                    high=self.cfg.quad.rand_pos[0][1]
                ),
                random.uniform(
                    low=self.cfg.quad.rand_pos[1][0],
                    high=self.cfg.quad.rand_pos[1][1]
                )
            ))
            self.quad.speed.state = random.uniform(
                low=self.cfg.quad.rand_speed[0],
                high=self.cfg.quad.rand_speed[1]
            )
            self.quad.yaw.state = random.uniform(
                low=self.cfg.quad.rand_yaw[0],
                high=self.cfg.quad.rand_yaw[1]
            )
        obs = self.observe()
        return obs

    def set_dot(self, t, action):
        del_speed, del_yaw = action
        speed = self.quad.speed.state
        yaw = self.quad.yaw.state

        cmd_speed = speed + del_speed*self.cfg.dt
        cmd_yaw = yaw + del_yaw*self.cfg.dt
        self.quad.set_dot(cmd_speed, cmd_yaw)

        return dict(time=t, **self.observe_dict(), cmd_speed=cmd_speed,
                    cmd_yaw=cmd_yaw, action=action)

    def step(self, action):
        *_, time_out = self.update(action=action)
        obs = self.observe()
        reward = self.get_reward(obs)
        done = self.terminate(time_out, obs[0][0])
        return obs, reward, done

    def observe(self):
        theta, curvature, yaw_T, distance = self.compute_closest()
        x, y = self.quad.pos.state.squeeze()
        yaw = self.quad.yaw.state
        A = self.cfg.traj.A
        y_T = -np.sin(yaw_T)*(x - 2*A*np.cos(theta)) \
            + np.cos(yaw_T)*(y - A*np.sin(2*theta))
        e_yaw = wrap(yaw - yaw_T)
        obs = np.hstack([y_T, e_yaw, curvature]).reshape(1,-1)

        self.obs = obs.squeeze()
        self.theta_min = theta
        self.curvature_min = curvature
        self.yaw_T_min = yaw_T
        self.distance_min = distance
        return obs

    def logger_callback(self, t, action):
        return dict(theta_min=self.theta_min, curvature_min=self.curvature_min,
                    distance_min=self.distance_min, yaw_T_min=self.yaw_T_min,
                    reward=self.reward, obs=self.obs)

    def terminate(self, time_out, y_T):
        done = 1. if (
            abs(y_T) > self.cfg.ddpg.terminate_condition or time_out
        ) else 0.
        return done

    def get_reward(self, obs):
        y_T, e_yaw, *_ = obs.squeeze()
        if abs(y_T) > self.cfg.ddpg.terminate_condition:
            r = -self.cfg.ddpg.reward_max
        else:
            r = -self.cfg.ddpg.reward_weight*abs(y_T) - e_yaw**2
        self.reward = r
        return r

    def compute_closest(self):
        x, y = self.quad.pos.state
        distance = [
            (x-traj[0])**2 + (y-traj[1])**2 for traj in self.trajectory
        ]
        distance_min = min(distance)
        min_index = distance.index(distance_min)
        theta_min = self.theta_set[min_index]
        curvature_min = self.curvature_set[min_index]
        yaw_T_min = self.yaw_T_set[min_index]
        return theta_min, curvature_min, yaw_T_min, distance_min
        
    def make_trajectory(self):
        theta_set = list(range(-180, 180))
        delete_set = [-180, -135, -45, 0, 45, 135]
        for item in delete_set:
            theta_set.remove(item)
        theta_set = np.array(theta_set)*np.pi/180
        trajectory = [None]*self.cfg.traj.split_num
        curvature_set = [None]*self.cfg.traj.split_num
        yaw_T_set = [None]*self.cfg.traj.split_num
        A = self.cfg.traj.A
        for i, theta in enumerate(theta_set):
            trajectory[i] = [2*A*np.cos(theta), A*np.sin(2*theta)]
            cos1 = np.cos(theta)
            cos2 = np.cos(2*theta)
            sin1 = np.sin(theta)
            sin2 = np.sin(theta)
            den = sin1**2 + cos2**2
            curvature_x = cos1*(1 - sin1**2*(1-4*cos2)/den) \
                / (2*A*sin1*np.sqrt(den))
            curvature_y = -sin2*(4 + cos2*(1-4*cos2)/den) \
                / (4*A*cos2*np.sqrt(den))
            curvature_set[i] = [curvature_x, curvature_y]
            yaw_T_set[i] = np.arctan2(cos2, -sin1)
        return trajectory, theta_set, curvature_set, yaw_T_set

def wrap(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi
