import numpy as np
import random
from types import SimpleNamespace as SN
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import fym.logging as logging

from dynamics import PointMass2DPathFollowing
from postProcessing import PostProcessing

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def load_config():
    cfg = SN()
    cfg.dt = 0.1
    cfg.max_t = 10.
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))
    cfg.num_train = 10
    cfg.num_eval = 5

    cfg.quad = SN()
    cfg.quad.tau_speed = 0.1
    cfg.quad.tau_yaw = 0.3
    cfg.quad.init_pos = np.vstack((-8., 5.))
    cfg.quad.init_speed = 1
    cfg.quad.init_yaw = np.pi/4
    cfg.quad.rand_pos = [[-12., 12.], [-8., 8.]]
    cfg.quad.rand_speed = [-3., 3.]
    cfg.quad.rand_yaw = [-np.pi, np.pi]

    cfg.traj = SN()
    cfg.traj.A = 4.
    cfg.traj.split_num  = 354

    cfg.ddpg = SN()
    cfg.ddpg.dim_state = 4
    cfg.ddpg.dim_action = 2
    cfg.ddpg.action_max = np.array([3., np.pi])
    cfg.ddpg.action_min = np.array([-3., -np.pi])
    cfg.ddpg.action_scaling = torch.from_numpy(cfg.ddpg.action_max).float()
    cfg.ddpg.memory_size = 20000
    cfg.ddpg.actor_lr = 0.0001
    cfg.ddpg.critic_lr = 0.001
    cfg.ddpg.batch_size = 64
    cfg.ddpg.discount = 0.999
    cfg.ddpg.softupdate_rate = 0.01

    cfg.noise = SN()
    cfg.noise.rho = 0.01
    cfg.noise.mu = np.zeros(cfg.ddpg.dim_action)
    cfg.noise.sigma = 1 / cfg.ddpg.action_max
    cfg.noise.size = cfg.ddpg.dim_action
    cfg.noise.dt = cfg.dt
    cfg.noise.x0 = None
    cfg.noise.tau = 200
    return cfg


class ActorNet(nn.Module):
    def __init__(self, cfg):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(cfg.dim_state, 8)
        self.lin2 = nn.Linear(8, 16)
        self.lin3 = nn.Linear(16, 8)
        self.lin4 = nn.Linear(8, cfg.dim_action)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)
        self.cfg = cfg

    def forward(self, state):
        x1 = self.relu(self.bn1(self.lin1(state)))
        x2 = self.relu(self.bn2(self.lin2(x1)))
        x3 = self.relu(self.bn3(self.lin3(x2)))
        x4 = self.tanh(self.lin4(x3))
        x_scaled = x4 * self.cfg.action_scaling
        return x_scaled

class CriticNet(nn.Module):
    def __init__(self, cfg):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(cfg.dim_state, 8)
        self.lin2 = nn.Linear(cfg.dim_action+8, 16)
        self.lin3 = nn.Linear(16, 8)
        self.lin4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)

    def forward(self, state, action):
        x1 = self.relu(self.bn1(self.lin1(state)))
        x_cat = torch.cat((x1, action), dim=1)
        x2 = self.relu(self.bn2(self.lin2(x_cat)))
        x3 = self.relu(self.bn3(self.lin3(x2)))
        x4 = self.lin4(x3)
        return x4

class DDPG:
    def __init__(self, cfg):
        self.memory = deque(maxlen=cfg.memory_size)
        self.behavior_actor = ActorNet(cfg).float()
        self.behavior_critic = CriticNet(cfg).float()
        self.target_actor = ActorNet(cfg).float()
        self.target_critic = CriticNet(cfg).float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=cfg.critic_lr
        )
        self.mse = nn.MSELoss()
        self.hardupdate(self.target_actor, self.behavior_actor)
        self.hardupdate(self.target_critic, self.behavior_critic)
        self.cfg = cfg

    def get_action(self, state, net='behavior'):
        with torch.no_grad():
            action = self.behavior_actor(torch.FloatTensor(state)) \
                if net == "behavior" \
                else self.target_actor(torch.FloatTensor(state))
        return np.array(np.squeeze(action))

    def memorize(self, item):
        self.memory.append(item)

    def get_sample(self):
        sample = random.sample(self.memory, self.cfg.batch_size)
        state, action, reward, state_next, epi_done = zip(*sample)
        x = torch.FloatTensor(state)
        u = torch.FloatTensor(action)
        r = torch.FloatTensor(reward).view(-1,1)
        xn = torch.FloatTensor(state_next)
        done = torch.FloatTensor(epi_done).view(-1,1)
        return x, u, r, xn, done

    def train(self):
        x, u, r, xn, done = self.get_sample()
        with torch.no_grad():
            action = self.target_actor(xn)
            Qn = self.target_critic(xn, action)
            target = r + (1-done)*self.cfg.discount*Qn
        Q_w_noise_action = self.behavior_critic(x, u)
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        action_wo_noise = self.behavior_actor(x)
        Q = self.behavior_critic(x, action_wo_noise)
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        self.softupdate(self.target_actor, self.behavior_actor)
        self.softupdate(self.target_critic, self.behavior_critic)

    def save_params(self, path_save):
        torch.save({
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'behavior_actor': self.behavior_actor.state_dict(),
            'behavior_critic': self.behavior_critic.state_dict()
        }, path_save)

    def set_train_mode(self):
        self.behavior_actor.train()
        self.behavior_critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def set_eval_mode(self):
        self.behavior_actor.eval()
        self.behavior_critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def hardupdate(self, target, behavior):
        target.load_state_dict(behavior.state_dict())
        for target_param, behavior_param in zip(
            target.parameters(),
            behavior.parameters()
        ):
            target_param.data.copy_(behavior_param.data)

    def softupdate(self, target, behavior):
        for target_param, behavior_param in zip(
            target.parameters(),
            behavior.parameters()
        ):
            target_param.data.copy_(
                target_param.data*(1. - self.cfg.softupdate_rate) \
                + behavior_param.data*self.cfg.softupdate_rate
            )


class OrnsteinUhlenbeckNoise:
    def __init__(self, cfg):
        self.rho = cfg.rho
        self.mu = cfg.mu
        self.sigma = cfg.sigma
        self.size = cfg.size
        self.dt = cfg.dt
        self.x0 = cfg.x0
        self.reset()

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros(self.size)

    def get_noise(self):
        self.x = self.x + self.rho*(self.mu-self.x)*self.dt \
            + np.sqrt(self.dt)*self.sigma*np.random.normal(size=self.size)
        return self.x


def train(cfg, env, agent, noise, epi):
    x = env.reset()
    noise.reset()
    while True:
        agent.set_eval_mode()
        action = np.clip(
            agent.get_action(x) + noise.get_noise()/(epi/cfg.noise.tau + 1),
            cfg.ddpg.action_max,
            cfg.ddpg.action_min
        )
        xn, r, done  = env.step(action)
        item = (x.squeeze(), action, r, xn.squeeze(), done)
        agent.memorize(item)
        x = xn
        if len(agent.memory) > 5 * cfg.ddpg.batch_size:
            agent.set_train_mode()
            agent.train()
        if done:
            break

def evaluate(cfg, env, agent, dir_save_env):
    agent.set_eval_mode()
    env.logger = logging.Logger(dir_save_env)
    env.logger.set_info(cfg=cfg)

    x = env.reset(random_init=False)
    while True:
        action = agent.get_action(x)
        xn, _, done = env.step(action)
        x = xn
        if done:
            break
    env.logger.close()


def main():
    cfg = load_config()
    env = PointMass2DPathFollowing(cfg)
    agent = DDPG(cfg.ddpg)
    noise = OrnsteinUhlenbeckNoise(cfg.noise)
    cfg.traj.trajectory = env.trajectory
    cfg.traj.theta_set = env.theta_set
    cfg.traj.curvature_set = env.curvature_set
    cfg.traj.yaw_T_set = env.yaw_T_set
    post_processing = PostProcessing(cfg)

    for epi in tqdm(range(cfg.num_train)):
        train(cfg, env, agent, noise, epi)

        if (epi+1) % cfg.num_eval == 0:
            dir_save = Path(cfg.dir, f"epi_after_{epi+1:05d}")
            dir_save_env = Path(dir_save, "env_data.h5")
            dir_save_agent = Path(dir_save, "agent_params.h5")

            evaluate(cfg, env, agent, dir_save_env)
            agent.save_params(dir_save_agent)
            post_processing.draw_plot(dir_save, dir_save_env)
    env.close()
    post_processing.compare_eval()
    

if __name__ == "__main__":
    main()
