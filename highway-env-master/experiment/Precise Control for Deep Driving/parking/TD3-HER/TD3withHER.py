import collections
import math
import random

import numpy as np
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Trajectory:
    """ 用来记录一条完整轨迹 """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_ = []
        self.dones = []
        self.length = 0

    def store_step(self, state, action, reward, state_, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_.append(state_)
        self.dones.append(done)
        self.length += 1


# Replay Buffer的代码实现
class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.buffer = collections.deque(maxlen=max_size)
        self.batch_size = batch_size

    def store_transition(self, trajectory):
        self.buffer.append(trajectory)

    def sample_buffer(self, use_her, dis_threshold, her_ratio, memory_size):
        batch = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for _ in range(self.batch_size):
            traj = random.sample(self.buffer, 1)[0]  # 从buffer中随机抽样一个轨迹
            step_state = np.random.randint(traj.length)  # 从抽样的轨迹中随机抽样出一个transition
            state = traj.states[step_state]  # 抽样出的transition的state
            next_state = traj.states_[step_state]  # 抽样出的transition的next_state
            action = traj.actions[step_state]  # 抽样出的transition的动作
            reward = traj.rewards[step_state]  # 抽样出的transition的奖励
            done = traj.dones[step_state]  # 抽样出的transition是否done
            if use_her and np.random.uniform() <= her_ratio and step_state + 1 < traj.length and memory_size > 250:
                step_goal = np.random.randint(step_state + 1, traj.length)  # 从上面transition之后的轨迹选择一个设置之后的goal
                goal = traj.states[step_goal][6:12]  # 使用HER算法的future方案设置目标，选择此transition的当前位置作为goal

                dis = np.sqrt(np.sum(np.square(next_state[6:12] - goal)))
                if dis >= dis_threshold:
                    reward = -5
                    done = False
                else:
                    reward = 5
                    done = True
                # reward = -50.0 if dis > dis_threshold else 10
                # done = False if dis > dis_threshold else True
                state = np.hstack((state[:12], goal))  # 将原来的初始位置和后来挑选的goal拼接
                next_state = np.hstack((next_state[:12], goal))  # 将原来的下一个transition的初始位置和goal拼接
            batch['states'].append(state)
            batch['next_states'].append(next_state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        batch['states'] = np.array(batch['states'])  # 256*18
        batch['next_states'] = np.array(batch['next_states'])  # 256*18
        batch['actions'] = np.array(batch['actions'])  # 256*?
        return batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']

    def size(self):
        return len(self.buffer)


# Actor和Critic网络的代码实现
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        """
        1.n.linear是一个线性全连接层，它接收输入张量，将其与权重矩阵相乘并加上偏差项以完成转换。它通常用于实现神经网络中的前向传递步骤，
          可以将输入张量映射到输出张量。这个操作在很多类型的深度学习任务，例如图像分类和语言建模中都非常有用。
        2.nn.layernorm是正则化技术之一，它能够对每个mini-batch的数据进行同样的规范化处理。具体来说，它可以对给定输入张量的最后一个维度进行标准化，
          并对结果进行缩放和位移。它通常用于避免神经网络中的内部协变量偏移（internal covariate shift）问题，从而提高模型的训练效果和泛化能力。
          以自然语言处理任务为例，在该领域应用广泛。
        """
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)

        """
        torch.optim.adam 是 pytorch 中一个优化器(optimizer)类，用于训练神经网络模型。adam是一种自适应学习率算法，在深度学习中广泛使用，
        能够在处理非平稳目标、梯度稀疏的问题时表现出色。
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.action(x))

        return action

    """
    torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=false)是一个pytorch函数，
    用于将神经网络模型的参数保存在指定文件路径中。具体来说，这个函数使用state_dict()方法从神经网络模型中提取各层的权重和偏置等参数，
    并把它们以python字典的形式存储。该字典可以被后续的代码用来重新初始化该神经网络。其中，checkpoint_file是将要保存参数的文件路径。
    _use_new_zipfile_serialization是一个可选参数，默认为false，用于控制是否采用新的zip文件序列化方式。
    """

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    """
    self.load_state_dict(torch.load(checkpoint_file)) 是一个 pytorch 中用于加载模型权重的函数。
    它需要传入一个包含了预训练权重的 .pt 或 .pth 文件的路径，然后读取文件并将权重加载到当前的神经网络中。
    这个函数会返回一个字典对象，其中包含了预训练权重的名称及其对应的 tensor 值。
    在加载预训练权重之后，你可以使用新数据集 fine-tune 该网络或者直接进行预测。
    """

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        """
        这是一段使用pytorch框架编写的神经网络代码，这段代码完成以下操作：
        1.把张量state和action在最后一个维度进行拼接，形成一个新的张量x。
        2.将张量x输入到一个全连接层（fc1）中，并对其进行线性变换。
        3.将线性变换后的结果输入到一个layernorm层（ln1）中，并对其进行标准化。
        4.将标准化的结果通过relu激活函数进行处理。
        ……
        这段代码实现了一个神经网络的前向传播过程。
        """
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


# TD3算法的代码实现
"""
TD3算法在DDPG算法的基础上，提出了三个关键技术：
（1）双重网络 (Double network)：采用两套Critic网络，计算目标值时取二者中的较小值，从而抑制网络过估计问题。
（2）目标策略平滑正则化 (Target policy smoothing regularization)：计算目标值时，在下一个状态的动作上加入扰动，从而使得价值评估更准确。
（3）延迟更新 (Delayed update)：Critic网络更新多次后，再更新Actor网络，从而保证Actor网络的训练更加稳定。
"""


class TD3:
    """
    alpha、beta ———— 学习率（分别对应ActorNetwork和CriticNetwork）
    """

    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir, gamma, tau, action_noise,
                 policy_noise, policy_noise_clip, delay_time, max_size, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = math.sqrt(action_noise)
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, traj):
        self.memory.store_transition(traj)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        # state = torch.tensor([observation], dtype=torch.float).to(device)

        # """转换observation的维度"""
        # observation = gym.spaces.utils.flatten(self.env.observation_space, observation)
        # np.reshape(observation,[-1])

        state = torch.tensor(observation, dtype=torch.float).to(device)
        action = self.actor.forward(state)

        if train:
            """
            torch.tensor(np.random.normal(loc=0.0, scale=self.policy_noise), dtype=torch.float)
            是一个将NumPy数组转换为PyTorch张量的操作。具体来说：
            np.random.normal(loc=0.0, scale=self.policy_noise)生成一个均值为0，标准差为self.policy_noise的正态分布随机数。
            torch.tensor()将这个随机数数组转换为PyTorch张量，并指定数据类型为torch.float。
            """
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                                 dtype=torch.float).to(device)
            action = torch.clamp(action + noise, -1, 1)

        self.actor.train()

        return action.squeeze().detach().cpu().numpy()

    def learn(self):
        # 没有达到一个batch
        if self.memory.size() < 20:
            return

        states, actions, rewards, states_, terminals = self.memory.sample_buffer(True, 0.3, 0.8, self.memory.size())
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states_tensor = torch.tensor(states_, dtype=torch.float).to(device)
        terminals_tensor = torch.tensor(terminals).to(device)

        with torch.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)

            action_noise = torch.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                        dtype=torch.float).to(device)
            # smooth noise

            action_noise = torch.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = torch.clamp(next_actions_tensor + action_noise, -1, 1)
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            critic_val = torch.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val

        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return

        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic1.forward(states_tensor, new_actions_tensor)
        actor_loss = -torch.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Saving critic1 network successfully!')
        self.target_critic1.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Saving target critic1 network successfully!')
        self.critic2.save_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Saving critic2 network successfully!')
        self.target_critic2.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Saving target critic2 network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Loading critic1 network successfully!')
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Loading target critic1 network successfully!')
        self.critic2.load_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Loading critic2 network successfully!')
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Loading target critic2 network successfully!')
