import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer:
    """
    经验回放缓冲区 (Replay Buffer)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q 网络结构
    支持标准 DQN 以及 Dueling DQN 结构
    """
    def __init__(self, state_dim, action_dim, dueling=False):
        super(QNetwork, self).__init__()
        self.dueling = dueling
        
        # 共享特征提取网络
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        if self.dueling:
            # 状态价值分支 V(s)
            self.value_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            # 动作优势分支 A(s, a)
            self.advantage_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
        else:
            # 标准 Q 值输出分支 Q(s, a)
            self.q_head = nn.Sequential(
                nn.Linear(128, action_dim)
            )

    def forward(self, x):
        features = self.feature(x)
        if self.dueling:
            values = self.value_head(features)
            advantages = self.advantage_head(features)
            # Dueling DQN 的结合公式，减去均值保证可辨识性
            q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
            return q_values
        else:
            q_values = self.q_head(features)
            return q_values


class DQNAgent:
    """
    DQN / Double DQN / Dueling DQN 控制器智能体
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 buffer_size=100000, batch_size=64, double_dqn=True, dueling_dqn=True, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.tau = tau
        
        # 创建在线网络和目标网络
        self.q_net = QNetwork(state_dim, action_dim, dueling=dueling_dqn)
        self.target_net = QNetwork(state_dim, action_dim, dueling=dueling_dqn)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 记录训练指标，用于分析过估偏差 (Overestimation Bias)
        self.loss_history = []
        self.q_value_history = []

    def select_action(self, state, epsilon=0.0):
        """
        epsilon-greedy 动作选择
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = q_values.argmax(dim=-1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        核心网络参数更新
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 1. 计算当前的 Q(s, a)
        q_values = self.q_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. 计算目标 $y_i$
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 在线网络选择动作
                next_state_actions = self.q_net(next_states).argmax(dim=-1, keepdim=True)
                # 目标网络计算该动作的 Q 值
                next_q_values = self.target_net(next_states).gather(1, next_state_actions).squeeze(1)
            else:
                # 标准 DQN: 目标网络选择并评估最大动作 Q 值
                next_q_values = self.target_net(next_states).max(dim=-1)[0]
                
            expected_state_action_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        # 3. 计算 Huber 损失 (lec-8 推荐用于平滑大梯度)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # 4. 反向传播更新
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 5. 软更新目标网络参数 (Polyak 更新): $\bar{\theta} \leftarrow \tau \theta + (1-\tau)\bar{\theta}$
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        # 记录当前 batch 估计的平均 Q 值以分析过估倾向
        avg_q_estimation = state_action_values.mean().item()
        
        return loss.item(), avg_q_estimation
