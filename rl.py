import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# 环境模拟
class TrafficEnv:
    def __init__(self, traffic_data, model, window_size):
        self.traffic_data = traffic_data
        self.model = model
        self.window_size = window_size
        self.cache = None

    def reset(self):
        # Reset environment, e.g., set the initial state
        self.cache = None
        return self.get_state(0)

    def get_state(self, t):
        # Get state at time t
        if t + self.window_size >= len(self.traffic_data):
            return None
        data_window = self.traffic_data[t:t + self.window_size]
        return np.concatenate([data_window.flatten(), [0]])  # 加上预测误差占位符

    def step(self, t, action):
        # If action is 1, use cache, else re-run model
        if action == 0:
            prediction = self.model(self.traffic_data[t:t + self.window_size])
            error = np.random.random()  # 模拟误差
            self.cache = prediction
        else:
            prediction = self.cache
            error = np.random.random() * 0.5  # 模拟较小的误差
        
        # 奖励设计: 根据误差的反向
        reward = -error

        next_state = self.get_state(t + 1)
        done = (next_state is None)

        return next_state, reward, done

# 强化学习代理
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.qnetwork.state_dict())

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice([0, 1])
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.qnetwork(state)
            return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # 计算当前 Q 值
        q_values = self.qnetwork(states).gather(1, actions)

        # 计算目标 Q 值
        next_q_values = self.target_network(next_states).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 训练代理
def train_dqn(agent, env, episodes=500, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    epsilon = epsilon_start
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        t = 0

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(t, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            t += 1

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        if episode % 10 == 0:
            agent.update_target()

# 示例: 假设有一个简单的流量预测模型
class SimpleTrafficModel:
    def __call__(self, data_window):
        # 模拟简单的流量预测
        return np.random.random(data_window.shape)

# 示例流量数据
traffic_data = np.random.random((1000, 10))

# 实例化强化学习环境和代理
model = SimpleTrafficModel()
env = TrafficEnv(traffic_data, model, window_size=24)
agent = DQNAgent(state_size=241, action_size=2)  # 状态大小取决于数据窗口和误差

# 训练强化学习代理
train_dqn(agent, env, episodes=500)