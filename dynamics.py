import numpy as np
import numpy.random as random

from fym.core import BaseEnv, BaseSystem
from fym.utils import rot
from fym.agents import LQR

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


class PointMass2DLinear(BaseEnv):
    def __init__(self, cfg):
        super().__init__()
        # self.pos = BaseSystem(cfg.init_pos)
        # self.vel = BaseSystem(cfg.init_vel)
        self.pos = BaseSystem(shape=(2,1))
        self.vel = BaseSystem(shape=(2,1))
        self.m = cfg.mass

    def set_dot(self, force):
        self.pos.dot = self.vel.state
        self.vel.dot = force / self.m


class Pendulum(BaseEnv):
    def __init__(self):
        super().__init__()
        # self.th = BaseSystem(np.vstack([0.]))
        # self.thdot = BaseSystem(np.vstack([0.]))
        self.th = BaseSystem(np.array([0.]))
        self.thdot = BaseSystem(np.array([0.]))
        self.g = 10
        self.m = 1.
        self.l = 1.
        self.max_u = 2.
        self.max_speed = 8.

    def set_dot(self, u):
        th = self.th.state
        thdot = self.thdot.state

        thdot_tmp = -3 * self.g / (2 * self.l) * np.sin(th + np.pi) \
            + 3. / (self.m * self.l ** 2) * u

        self.th.dot = thdot
        self.thdot.dot = np.clip(thdot_tmp, -self.max_speed, self.max_speed)


class InvertedPendulum(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.05, max_t=20)
        self.pendulum = Pendulum()

    def reset(self):
        super().reset()
        self.pendulum.th.state = randomize([[-np.pi, np.pi]])
        self.pendulum.thdot.state = randomize([[-1, 1]])
        obs = self.observe()
        return obs

    def step(self, u):
        *_, done = self.update(u=u)
        r = self.get_reward(u)
        obs = self.observe()
        return obs, r, done

    def set_dot(self, t, u):
        self.pendulum.set_dot(u)
        reward = self.get_reward(u)
        return dict(time=t, **self.observe_dict(), reward=reward, action=u)

    def get_reward(self, u):
        th = self.pendulum.th.state
        thdot = self.pendulum.thdot.state
        th = wrap(th)
        r = -th ** 2 - 0.1 * thdot ** 2 - 0.001 * u ** 2
        return r

    def observe(self):
        th = self.pendulum.th.state
        thdot = self.pendulum.thdot.state
        obs = np.hstack([ np.cos(th), np.sin(th), thdot ]).reshape(1, -1)
        return obs


class PointMass2DPathFollowing(BaseEnv):
    def __init__(self, cfg):
        super().__init__(dt=cfg.dt, max_t=cfg.max_t)
        self.quad = PointMass2D(cfg.quad)
        self.cfg = cfg
        self.trajectory, self.theta_set, self.curvature_set, self.yaw_T_set, \
            self.tangent_set= self.make_trajectory()
        self.theta_min = None
        self.curvature_min = None
        self.yaw_T_min = None
        self.reward = None
        self.distance_min = None
        self.obs = None

    def reset(self):
        super().reset()
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
        theta, curvature, yaw_T, distance, tangent, traj \
            = self.compute_closest()
        x, y = self.quad.pos.state.squeeze()
        yaw = self.quad.yaw.state
        body_x = np.array([np.cos(yaw), np.sin(yaw)]).squeeze()
        e_yaw = -np.dot(body_x, np.array(tangent)) \
            / np.sqrt(tangent[0]**2 + tangent[1]**2)
        # e_yaw = wrap(yaw - yaw_T)
        # e_yaw = np.linalg.norm(np.cross(body_x, np.array(tangent)))

        tmp = tangent[0]*(y - traj[1]) - tangent[1]*(x - traj[0])
        y_T = distance.item() if tmp > 0 else -distance.item()
        # y_T = -np.sin(yaw_T)*(x - traj[0]) + np.cos(yaw_T)*(y - traj[1])

        obs = np.vstack([y_T, e_yaw, curvature]).reshape(1,-1)

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
            r = -self.cfg.ddpg.reward_weight*abs(y_T) - e_yaw
        self.reward = r
        return r

    def compute_closest(self):
        x, y = self.quad.pos.state
        distance = [
            np.sqrt((x-traj[0])**2 + (y-traj[1])**2) for traj in self.trajectory
        ]
        distance_min = min(distance)
        min_index = distance.index(distance_min)
        theta_min = self.theta_set[min_index]
        curvature_min = self.curvature_set[min_index]
        yaw_T_min = self.yaw_T_set[min_index]
        tangent_min = self.tangent_set[min_index]
        traj_min = self.trajectory[min_index]
        return theta_min, curvature_min, yaw_T_min, distance_min, tangent_min,\
            traj_min
        
    def make_trajectory(self):
        theta_set = list(range(-180, 180))
        delete_set = [-180, -135, -45, 0, 45, 135]
        for item in delete_set:
            theta_set.remove(item)
        theta_set = np.array(theta_set)*np.pi/180
        trajectory = [None]*self.cfg.traj.split_num
        curvature_set = [None]*self.cfg.traj.split_num
        tangent_set = [None]*self.cfg.traj.split_num
        yaw_T_set = [None]*self.cfg.traj.split_num
        A = self.cfg.traj.A
        for i, theta in enumerate(theta_set):
            cos1 = np.cos(theta)
            cos2 = np.cos(2*theta)
            sin1 = np.sin(theta)
            sin2 = np.sin(2*theta)
            trajectory[i] = [2*A*cos1, A*sin2]
            den = sin1**2 + cos2**2
            curvature_x = cos1*(1 - sin1**2*(1-4*cos2)/den) \
                / (2*A*sin1*np.sqrt(den))
            curvature_y = -sin2*(4 + cos2*(1-4*cos2)/den) \
                / (4*A*cos2*np.sqrt(den))
            tangent_x = -2*A*sin1 
            tangent_y = 2*A*cos2
            curvature_set[i] = [curvature_x, curvature_y]
            yaw_T_set[i] = np.arctan2(cos2, -sin1)
            tangent_set[i] = [tangent_x, tangent_y]
        return trajectory, theta_set, curvature_set, yaw_T_set, tangent_set


class SimpleQuad(BaseEnv):
    def __init__(self, cfg):
        super().__init__()
        self.pos = BaseSystem(cfg.init_pos)
        self.vel = BaseSystem(cfg.init_vel)
        self.euler = BaseSystem(cfg.init_euler)
        self.thrust = BaseSystem(cfg.mass*9.81)
        self.tau_euler = cfg.tau_euler
        self.tau_thrust = cfg.tau_thrust
        self.mass = cfg.mass
        self.z = np.vstack((0., 0., 1.))
        self.g = -9.81*self.z

    def set_dot(self, cmd_f, cmd_euler, disturbance):
        self.pos.dot = self.vel.state
        phi, theta, psi = self.euler.state.squeeze()
        R = rot.angle2dcm(psi, theta, phi)
        force = self.thrust.state*R.dot(self.z)
        self.vel.dot = (force + disturbance)/self.mass + self.g
        self.euler.dot = (-self.euler.state + cmd_euler) / self.tau_euler
        self.thrust.dot = (-self.thrust.state + cmd_f) / self.tau_thrust


class SimpleQuadGuidance(BaseEnv):
    def __init__(self, cfg):
        super().__init__(dt=cfg.dt, max_t=cfg.max_t)
        self.quad = SimpleQuad(cfg.quad)
        self.cfg = cfg
        self.reward = None
        self.obs = None

    def reset(self, goal):
        super().reset()
        self.quad.pos.state = randomize(self.cfg.quad.rand_pos)
        self.quad.vel.state = randomize(self.cfg.quad.rand_vel)
        self.quad.euler.state = randomize(self.cfg.quad.rand_euler)
        obs = self.observe(goal)
        return obs

    def set_dot(self, t, action, cmd_psi, disturbance, goal):
        cmd_f, cmd_euler = self.convert_action2cmd(action, cmd_psi)
        self.quad.set_dot(cmd_f, cmd_euler, disturbance)

        return dict(time=t, **self.observe_dict(), cmd_euler=cmd_euler, 
                    cmd_f=cmd_f, action=action, disturbance=disturbance, 
                    goal=goal)

    def step(self, action, goal, cmd_psi):
        disturbance = np.vstack((0., 0., 0.))
        *_, done = self.update(
            action=action,
            cmd_psi=cmd_psi,
            disturbance=disturbance,
            goal=goal
        )
        obs = self.observe(goal)
        reward = self.get_reward(obs)
        return obs, reward, done

    def convert_action2cmd(self, action, cmd_psi):
        cmd_f, cmd_chi, cmd_gamma = action.squeeze()
        z_des = rot.sph2cart2(1, cmd_chi, cmd_gamma)
        cmd_phi, cmd_theta = self.convert_z2euler(z_des, cmd_psi)
        cmd_euler = np.vstack((cmd_phi, cmd_theta, cmd_psi))
        return cmd_f, cmd_euler

    def convert_z2euler(self, z_des, psi):
        z_n = rot.angle2dcm(psi, 0, 0).dot(z_des)
        theta = np.arctan2(z_n[0], z_n[2]).item()
        phi = np.arctan2(-z_n[1]*z_n[2], np.cos(theta)*(1-z_n[1]**2)).item()
        return phi, theta

    def observe(self, goal):
        pos = self.quad.pos.state
        vel = self.quad.vel.state
        euler = self.quad.euler.state
        e_pos = pos - goal
        obs = np.vstack([e_pos, vel, euler]).reshape(1,-1)
        return obs

    def get_reward(self, obs):
        e_pos = obs[0][0:3]
        r = -np.sqrt(e_pos[0]**2 + e_pos[1]**2 + e_pos[2]**2)
        return r

    def logger_callback(self, t, action, cmd_psi, disturbance, goal):
        obs = self.observe(goal)
        reward = self.get_reward(obs)
        phi, theta, psi = self.quad.euler.state.squeeze()
        R = rot.angle2dcm(psi, theta, phi)
        return dict(reward=reward, obs=obs.squeeze(), dcm=R)


class PointMass2DLinearOptimal(BaseEnv):
    def __init__(self, cfg):
        super().__init__(dt=cfg.dt, max_t=cfg.max_t)
        self.quad = PointMass2DLinear(cfg.quad)
        self.cfg = cfg
        self.B = cfg.quad.B
        self.K, self.P = LQR.clqr(
            cfg.quad.A,
            cfg.quad.B,
            np.zeros((4,4)),
            np.eye(2)
        )

    def reset(self):
        super().reset()
        self.quad.pos.state = self.randomize(self.cfg.quad.rand_pos)
        self.quad.vel.state = self.randomize(self.cfg.quad.rand_vel)
        obs = self.observe()
        return obs

    def set_dot(self, t, action):
        u = -self.B.T.dot(action)
        self.quad.set_dot(u.reshape(-1, 1))

        obs = self.observe()
        LQR_lambda = self.P.dot(obs.squeeze())
        LQR_u = -self.K.dot(obs.squeeze())
        reward = self.get_reward(obs, action)
        return dict(time=t, **self.observe_dict(), action=action, u=u,
                    reward=reward, LQR_lambda=LQR_lambda, LQR_u=LQR_u,
                    LQR_P=self.P)

    def step(self, action):
        *_, done = self.update(action=action)
        obs = self.observe()
        reward = self.get_reward(obs, action)
        return obs, reward, done

    def observe(self):
        pos = self.quad.pos.state
        vel = self.quad.vel.state
        obs = np.vstack([pos, vel]).reshape(1,-1)
        return obs

    def get_reward(self, obs, action):
        pos = obs[0][0:2]
        vel = obs[0][2:4]
        u = -self.B.T.dot(action)
        r = -self.cfg.ddpg.reward_weight*pos.dot(vel) - u.dot(u)
        return r

    def randomize(self, bound):
        length = len(bound)
        vec = np.ones((length, 1))
        for i in range(len(bound)):
            vec[i] = random.uniform(low=bound[i][0], high=bound[i][1])
        return vec


def wrap(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

def randomize(bound):
    length = len(bound)
    vec = np.ones((length, 1))
    for i in range(len(bound)):
        vec[i] = random.uniform(low=bound[i][0], high=bound[i][1])
    return vec
