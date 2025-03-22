import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
from collections import deque
from utils.mdtp import MyDataset_rl, set_seed, metric1
from torch.utils.data.dataloader import DataLoader
from models.model import Net_timesnet_sample_onetimesnet
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
import functools
from utils import mdtp
from models.expert_s import Generator_rl
import time
import yaml
import torch.nn.functional as F



parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:4', help='GPU setting')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--window_size', type=int, default=24, help='window size')
parser.add_argument('--pred_size', type=int, default=4, help='pred size')
parser.add_argument('--node_num', type=int, default=231, help='number of node to predict')
parser.add_argument('--in_features', type=int, default=2, help='GCN input dimension')
parser.add_argument('--out_features', type=int, default=16, help='GCN output dimension')
parser.add_argument('--lstm_features', type=int, default=256, help='LSTM hidden feature size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1000, help='epoch')
parser.add_argument('--gradient_clip', type=int, default=5, help='gradient clip')
parser.add_argument('--pad', type=bool, default=False, help='whether padding with last batch sample')
parser.add_argument('--bike_base_path', type=str, default='./data/bike', help='bike data path')
parser.add_argument('--taxi_base_path', type=str, default='./data/taxi', help='taxi data path')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='/home_nfs/haitao/data/web_host_network/mdtp_moe/whnbest/best_model.pth', help='save path')
parser.add_argument('--rlsave', type=str, default='./rlbest/', help='save path')
parser.add_argument('--smoe_start_epoch', type=int, default=99, help='smoe start epoch')
parser.add_argument('--gpus', type=str, default='4', help='gpu')
parser.add_argument('--log', type=str, default='0.log', help='log name')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def custom_collate_fn(batch):
    # 因为 batch 是一个长度为 1 的列表，直接取出这个列表的第一个元素
    return batch

class Newdata():
    def __init__(self, bike_node, bike_adj, taxi_node, taxi_adj,cache):
        self.bike_adj = bike_adj
        self.bike_node = bike_node
        self.taxi_adj = taxi_adj
        self.taxi_node = taxi_node
        self.batch_size = bike_node.size(0)
        self.cache = cache
    
    def get_value(self):
        return self.bike_node, self.bike_adj, self.taxi_node, self.taxi_adj, self.cache
    
    def to(self, device):
        """Moves the data to the specified device (CPU or GPU)."""
        self.bike_node = self.bike_node.to(device)
        self.bike_adj = self.bike_adj.to(device)
        self.taxi_node = self.taxi_node.to(device)
        self.taxi_adj = self.taxi_adj.to(device)
        self.cache = self.cache.to(device)
        return self  # 返回自己以支持链式调用
    
    def slice(self):
        """Slices the data into batches based on the given batch size."""
        bike_batches = [b for b in torch.split(self.bike_node, 1)]
        bike_adj_batches = [b for b in torch.split(self.bike_adj, 1)]
        taxi_batches = [t for t in torch.split(self.taxi_node, 1)]
        taxi_adj_batches = [t for t in torch.split(self.taxi_adj, 1)]
        cache_batches = [c for c in torch.split(self.cache, 1)]
        
        # 返回以元组为元素的列表，每个元组包含bike和taxi数据的切片
        return list(zip(bike_batches, bike_adj_batches, taxi_batches, taxi_adj_batches, cache_batches))

    @staticmethod
    def concat(states, dim=0):
        """Concatenate a list of Newdata objects along the given dimension."""
        bike_nodes = torch.cat([state[0] for state in states], dim=dim)
        bike_adjs = torch.cat([state[1] for state in states], dim=dim)
        taxi_nodes = torch.cat([state[2] for state in states], dim=dim)
        taxi_adjs = torch.cat([state[3] for state in states], dim=dim)
        cache = torch.cat([state[4] for state in states], dim=dim)
        
        return Newdata(bike_nodes, bike_adjs, taxi_nodes, taxi_adjs, cache)

def euclidean_distance(A, B):
    # A 和 B 需要有相同的形状
    return torch.sqrt(torch.sum((A - B) ** 2, dim=-1))

def cosine_similarity(A, B):
    return torch.nn.functional.cosine_similarity(A, B, dim=1)

# Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, batch_size):
        super(QNetwork, self).__init__()
        self.generator = Generator_rl(
            window_size=1,
            node_num=231,
            in_features=2,
            out_features=4
        )
        self.fc = nn.Linear(462*4, 2)

        # self.tensor0 = nn.Parameter(torch.zeros
        # (231 * 4))  # 随机初始化 tensor0
        # self.tensor1 = nn.Parameter(self.tensor0.clone())  # 创建 tensor0 的独立副本，并取反
        self.tensor0 = nn.Parameter(torch.zeros(231 * 4, 100) * 0.001)
        self.tensor1 = nn.Parameter(torch.zeros(231 * 4, 100) * 0.001)
        self.sigmoid = nn.Sigmoid()

        # self.tensor1 = nn.Parameter(torch.zeros(231 * 4))

        # self.tensor1 = nn.Parameter(torch.full((231 * 4,), 0.005))      

        # self.fc1 = nn.Linear(231*4, 128)  # 输入层到隐藏层
        # self.relu = nn.ReLU()  # ReLU激活函数
        # self.fc2 = nn.Linear(128, 2)

        # self.tmpfc = nn.Linear(2, 2)
        self.sim_time = 0
        self.batch_size = batch_size
    
    # def forward(self, state):

    #     bike_node, bike_adj, taxi_node, taxi_adj, cache = state.get_value()
    #     if torch.all(cache[:,1, :, :].contiguous().view(1, -1)==0, dim=1):
    #         q_output = torch.tensor([0, 1]).unsqueeze(0).to(bike_node.dtype).to(device)
    #         return q_output
    #     pred = cache[:, 0, :, :].view(self.batch_size, -1)
    #     node = torch.cat((bike_node, taxi_node), dim=-1).view(self.batch_size, -1)
    #     node = pred - node
    #     dis1 = torch.norm(node.unsqueeze(-1) - self.tensor0.unsqueeze(0), dim=1)
    #     dis2 = torch.norm(node.unsqueeze(-1) - self.tensor1.unsqueeze(0), dim=1)


    #     dis1 = torch.sum(dis1, dim = 1)
    #     dis2 = torch.sum(dis2, dim = 1)

    #     # cos_sim_1 = cosine_similarity(node, self.tensor0)
    #     # cos_sim_2 = cosine_similarity(node, self.tensor1)
        
        
    #     # 将两个相似度拼接起来，形状为 (batch_size, 2)
    #     q_output = torch.stack([dis1, dis2], dim=1)
    #     q_output = q_output / q_output.sum(dim=1, keepdim=True)
        
    #     return q_output

    def forward(self, state):

        bike_node, bike_adj, taxi_node, taxi_adj, cache = state.get_value()

        
        # Unpacking new_data and cache from the input state
        # new_data, cache = state
        
        # Create a mask for entries where the cache is valid (assuming non-zero is valid)
        cache_empty_mask = torch.all(cache[:,1, :, :].contiguous().view(cache.size(0), -1)==0, dim=1)  
        
        # print(cache_empty_mask)
        # cache_empty_mask = torch.all(cache_empty_mask, dim=(1, 2))  # 生成 [batch_size] 大小的布尔掩码

        # Initialize an output tensor with the correct size (batch_size, action_size)
        batch_size = bike_node.size(0)
        action_size = 2
        output = torch.zeros(batch_size, action_size).to(device)
        
        # Process the states where the cache is invalid (empty/zero)
        # t1 = time.time()
        # if torch.any(torch.logical_not(cache_empty_mask)):
            # # 第一步：取出 dim=1 上的第 0 个
        pred = cache[:, 0, :, :].reshape(batch_size, -1)

        node = torch.cat((bike_node, taxi_node), dim=-1).view(batch_size, -1)
        # node = bike_node.view(batch_size, -1)
        # pred = torch.rand(32,231, 4).to(device)
        # 第二步：将最后一个维度分成两个张量
        # pred_bike, pred_taxi = torch.chunk(pred, 2, dim=-1)
        # bike_node = bike_node - pred_bike
        # taxi_node = taxi_node - pred_taxi

        # node = pred - node
        # node = pred

        # print(node)

        # bike_node, bike_adj, taxi_node, taxi_adj = \
        #         bike_node.unsqueeze(1), bike_adj.unsqueeze(1), \
        #         taxi_node.unsqueeze(1), taxi_adj.unsqueeze(1)
        
        
        # gcn_output = gcn_output.reshape(gcn_output.size(0), -1)
        # node = torch.cat((bike_node, taxi_node), dim=-1).reshape(batch_size, -1)

        # q_output = self.fc(node)
        # bike_node = bike_node.reshape(batch_size, -1)
        

        # cos_sim_1 = torch.nn.functional.cosine_similarity(node.unsqueeze(-1), self.tensor0.unsqueeze(0), dim=1)
        # cos_sim_2 = torch.nn.functional.cosine_similarity(node.unsqueeze(-1), self.tensor1.unsqueeze(0), dim=1)

        # gcn_output = self.generator(bike_node.unsqueeze(1), 
        #                             bike_adj.unsqueeze(1), 
        #                             taxi_node.unsqueeze(1), 
        #                             taxi_adj.unsqueeze(1))
        # gcn_output = gcn_output.reshape(self.batch_size, -1)
        # q_output = self.fc(gcn_output)

        dis1 = torch.norm(node.unsqueeze(-1) - self.tensor0.unsqueeze(0), dim=1)
        dis2 = torch.norm(node.unsqueeze(-1) - self.tensor1.unsqueeze(0), dim=1)


        dis1 = torch.sum(dis1, dim = 1)
        dis2 = torch.sum(dis2, dim = 1)

        # cos_sim_1 = cosine_similarity(node, self.tensor0)
        # cos_sim_2 = cosine_similarity(node, self.tensor1)
        
        
        # 将两个相似度拼接起来，形状为 (batch_size, 2)
        q_output = torch.stack([dis1, dis2], dim=1)
        q_output = q_output / q_output.sum(dim=1, keepdim=True)
        
        # out = self.fc1(node)
        # out = self.relu(out)
        # q_output = self.fc2(out)
        # print(torch.max(node[0], dim=0))
        # Store the computed values into the output where the cache is invalid
        # output[torch.logical_not(cache_empty_mask)] = q_output[torch.logical_not(cache_empty_mask)]
        output = q_output
        
        # t2 = time.time()
        # self.sim_time += t2- t1
        # print('sim:',self.sim_time)


                # q_output = torch.rand(32, 2).to(device)
                # q_output = self.tmpfc(q_output)
                # output[torch.logical_not(cache_empty_mask)] = q_output[torch.logical_not(cache_empty_mask)]
        
        # For the states where the cache is valid, set the output to [0, 1]
        specified_output = torch.tensor([0, 1]).to(output.dtype).to(device)  # Specify the fixed output for valid cache entries
        specified_output = torch.mean(q_output, dim=-1, keepdim=True)  # 保持均值后的维度为 (batch_size, 1)
        specified_output = specified_output.expand_as(q_output)
        # Broadcasting the specified output to match the shape of the valid cache entries
        output[cache_empty_mask] = specified_output[cache_empty_mask]

        # print(output)
        # output = F.softmax(output, dim=1)  # 按列进行 softmax，确保每个样本的两个相似度转换为概率
        return output

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
    def __init__(self, bike_adj, bike_node, taxi_adj, taxi_node, model, device, window_size, pred_size):
        self.bike_adj = bike_adj.to(device)
        self.bike_node = bike_node.to(device)
        self.taxi_adj = taxi_adj.to(device)
        self.taxi_node = taxi_node.to(device)
        self.total_length = bike_adj.size(1)
        self.model = model
        self.window_size = window_size
        self.batch_size = bike_adj.size(0)
        self.pred_size = pred_size
        self.device = device

        # self.cache = []
        # # 添加多个空的队列到列表中
        # for _ in range(self.batch_size):
        #     self.cache.append([0, None])
        self.cache = torch.zeros(self.batch_size, self.pred_size, 231, 4)

        self.loss = mdtp.mae_rlerror

    def reset(self):
        # Reset environment, e.g., set the initial state
        # self.cache = []
        # # 添加多个空的队列到列表中
        # for _ in range(self.batch_size):
        #     self.cache.append([])
        self.cache = torch.zeros((self.batch_size, self.pred_size, 231, 4), device=device)

        return self.get_state(0)

    def get_state(self, t):
        # Get state at time t

        if t + self.window_size >= self.total_length:
            return None
        
        new_data = Newdata( bike_node=self.bike_node[:,t + self.window_size - 1, :, :],\
                            bike_adj=self.bike_adj[:,t + self.window_size - 1, :, :],\
                            taxi_node=self.taxi_node[:,t + self.window_size - 1, :, :],\
                            taxi_adj=self.taxi_adj[:,t + self.window_size - 1, :, :],\
                            cache=self.cache[:, t:t+2].clone() )
        new_data = new_data.to(device)
        # bike_flow = self.bike_node[:,t + self.window_size - 1, :, :]


        # batch_size = new_data.size(0)
        # new_data = new_data.reshape(batch_size, -1)
        # cache = self.cache[:, t:t+2]
  
        return new_data
        # return np.concatenate([data_window.flatten(), [0]])  # 加上预测误差占位符
    
    def step_best(self, t, action):

        data = (self.bike_node[:, t:t + self.window_size, :, :].to(device), self.bike_adj[:, t:t + self.window_size, :, :].to(device), \
                            self.taxi_node[:, t:t + self.window_size, :, :].to(device), self.taxi_adj[:, t:t + self.window_size, :, :].to(device))
        bike_start, bike_end, taxi_start, taxi_end = self.model(data)
        bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(-1), bike_end.unsqueeze(-1), \
                                                        taxi_start.unsqueeze(-1), taxi_end.unsqueeze(-1)
        prediction = torch.cat((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
        model_pred = prediction[:, 0]
        cache_pred = self.cache[:, t + 1]
        action = action
        newcache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
        newcache = torch.cat((self.cache[:, :t+1, :, :], prediction[:, :, :, :]), dim=1)
        self.cache = newcache.detach()
        

        predloss, _, _, _ = self.get_acc(t, model_pred)
        cacheloss, _, _, _ = self.get_acc(t, cache_pred)

        if predloss[0] > cacheloss[0]:
            pred_final = cache_pred
        else:
            pred_final = model_pred


        next_state = self.get_state(t + 1)
        done = (next_state is None)


        t1 = time.time()
        t2 = time.time()
        model_time = t2 -t1
        return next_state, pred_final, done, model_time

    def steptest(self, t, action):
        # if torch.all(self.cache[:,t+1, :, :].view(1, -1)==0, dim=1):
        #     action = torch.zeros((1), device=device)
        if action == 0:
            data = (self.bike_node[:, t:t + self.window_size, :, :].contiguous(), 
                    self.bike_adj[:, t:t + self.window_size, :, :].contiguous(), 
                    self.taxi_node[:, t:t + self.window_size, :, :].contiguous(), 
                    self.taxi_adj[:, t:t + self.window_size, :, :].contiguous())
            
            t1 = time.time()
            with torch.no_grad():
                bike_start, bike_end, taxi_start, taxi_end = self.model(data)
            t2 = time.time()
            model_time = t2 - t1
            bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(-1), bike_end.unsqueeze(-1), \
                                                        taxi_start.unsqueeze(-1), taxi_end.unsqueeze(-1)
            prediction = torch.cat((bike_start, bike_end, taxi_start, taxi_end), dim=-1)

            # prediction = torch.cat((self.bike_node[:, t + self.window_size:t + self.window_size + 4, :, :].to(device),  \
            #                 self.taxi_node[:, t + self.window_size:t + self.window_size + 4, :, :].to(device)), dim=-1)
            
            cache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
            cache[action==0, :, :, :] = torch.cat((self.cache[action==0, :t+1, :, :], prediction[action==0, :, :, :]), dim=1)
            # t2 = t1 =0
        elif action == 1:
            # t1 = time.time()
            cache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
            zero = torch.zeros((self.batch_size, 1, 231, 4), device=device)
            cache[action==1] = torch.cat((self.cache[action==1, :, :, :], zero[action==1, :, :, :]), dim=1)
            model_time = 0
            # t2 = time.time()

        self.cache = cache.clone()
        pred_final = self.cache[:, t + 1]   
        # pred_final = prediction[:, 0, :, :]
        next_state = self.get_state(t + 1)
        done = (next_state is None)
        
        return next_state, pred_final, done, model_time


    def step(self, t, action):

        data = (self.bike_node[:, t:t + self.window_size, :, :].to(device), self.bike_adj[:, t:t + self.window_size, :, :].to(device), \
                            self.taxi_node[:, t:t + self.window_size, :, :].to(device), self.taxi_adj[:, t:t + self.window_size, :, :].to(device))
        bike_start, bike_end, taxi_start, taxi_end = self.model(data)
        bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(-1), bike_end.unsqueeze(-1), \
                                                        taxi_start.unsqueeze(-1), taxi_end.unsqueeze(-1)
        prediction = torch.cat((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
        action = action
        newcache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
        newcache[action==0] = torch.cat((self.cache[action==0, :t+1, :, :], prediction[action==0, :, :, :]), dim=1)
        zero = torch.zeros((self.batch_size, 1, 231, 4), device=device)
        newcache[action==1] = torch.cat((self.cache[action==1, :, :, :], zero[action==1, :, :, :]), dim=1)
        self.cache = newcache.detach()
        pred_final = self.cache[:, t + 1]
        
        # 奖励设计: 根据误差的反向
        labels = torch.cat((self.bike_node[:, t + self.window_size, :, :].to(device), self.taxi_node[:, t + self.window_size, :, :].to(device)), dim=-1)
        reward = self._get_reward(pred_final, prediction[:, 0, :, :], labels, action)

        next_state = self.get_state(t + 1)
        done = (next_state is None)

        return next_state, reward, done
    
    def _get_reward(self, pred_final, pred_model, lables, action):
        # acc reward
        pred_final = pred_final.to(device)
        pred_model = pred_model.to(device)
        lables = lables.to(device)

        # final_loss = self.loss(pred_final[..., :2], lables[..., :2])
        # pred_loss = self.loss(pred_model[..., :2], lables[..., :2])

        final_loss = self.loss(pred_final[:, :1], lables[:, :1])
        pred_loss = self.loss(pred_model[:, :1], lables[:, :1])
        # error = pred_loss - final_loss
        error = final_loss - pred_loss
        # eff reward
        # error_1 = error
        # error_1[error_1>0] = error_1[error_1>0] * 1
        # error_1[error_1<0] = error_1[error_1<0] / 1
         
        # error_2 = error
        # error_2[error_2>0] = error_2[error_2>0] / 1
        # error_2[error_2<0] = error_2[error_2<0] * 1

        # error[action==1]=error_1[action==1]  
        # error[action==0]=error_2[action==0]

        # error[action==1] += 0.005 
        # print(error)

        # positive = error > 0
        # error[positive] = error[positive] * 10

        # mean = error.mean()
        # std = error.std()
        # error = (error - mean) / std 
        # error[error<0] = 0

        return error
    
    def get_acc(self, t, pred_final):

        bike_start_real = self.bike_node[:, t+self.window_size, :, 0]
        bike_end_real = self.bike_node[:, t+self.window_size, :, 1]
        taxi_start_real = self.taxi_node[:, t+self.window_size, :, 0]
        taxi_end_real = self.taxi_node[:, t+self.window_size, :, 1]

        bike_start_pred = pred_final[..., 0]
        bike_end_pred = pred_final[..., 1]
        taxi_start_pred = pred_final[..., 2]
        taxi_end_pred = pred_final[..., 3]

        bk_start_mask = bike_start_real != bike_start_pred
        bk_end_mask = bike_end_real != bike_end_pred
        tx_start_mask = taxi_start_real != taxi_start_pred
        tx_end_mask = taxi_end_real != taxi_end_pred
        
        bike_start_metrics = metric1(bike_start_pred, bike_start_real, bk_start_mask)
        bike_end_metrics = metric1(bike_end_pred, bike_end_real, bk_end_mask)
        taxi_start_metrics = metric1(taxi_start_pred, taxi_start_real, tx_start_mask)
        taxi_end_metrics = metric1(taxi_end_pred, taxi_end_real, tx_end_mask)

        return bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics


# 强化学习代理
class DQNAgent:
    def __init__(self, state_size, action_size, device, buffer_size=64, batch_size=32, gamma=1.0, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork = QNetwork(state_size, action_size, batch_size).to(device)
        self.target_network = QNetwork(state_size, action_size, batch_size).to(device)
        self.target_network.eval()
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.qnetwork.state_dict())

    # def act(self, state, epsilon=1.0):
    #     if random.random() < epsilon:

    #         bike_node, bike_adj, taxi_node, taxi_adj, cache = state.get_value()

        
    #         # Create a mask for entries where the cache is valid (assuming non-zero is valid)
    #         cache_empty_mask = torch.all(cache[:,1, :, :].view(cache.size(0), -1)==0, dim=1)  
        
    #         # Initialize an output tensor with the correct size (batch_size, action_size)
    #         batch_size = bike_node.size(0)
    #         action_size = 2
    #         output = torch.zeros(batch_size, action_size).to(device)
        
    #         # Process the states where the cache is invalid (empty/zero)
    #         if torch.any(torch.logical_not(cache_empty_mask)):
    #             q_output = torch.randint(0, self.action_size, (self.batch_size, self.action_size)).to(output.dtype).to(device)
    #             # Store the computed values into the output where the cache is invalid
    #             output[torch.logical_not(cache_empty_mask)] = q_output[torch.logical_not(cache_empty_mask)]
            
    #         # For the states where the cache is valid, set the output to [0, 1]
    #         specified_output = torch.tensor([0, 1]).to(output.dtype).to(device)  # Specify the fixed output for valid cache entries
            
    #         # Broadcasting the specified output to match the shape of the valid cache entries
    #         output[cache_empty_mask] = specified_output            
    #         # return torch.argmax(output, dim=1)
    #         return torch.argmin(output, dim=1)
    #     else:
    #         # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #         # new_data, cache = state
    #         # state = (new_data.to(self.device), cache.to(self.device))
    #         state = state.to(device)
    #         with torch.no_grad():
    #             q_values = self.qnetwork(state).detach()
    #         # return torch.argmax(q_values).item()
    #         # return torch.argmax(q_values, dim=1)
    #         return torch.argmin(q_values, dim=1)

    def act(self, state, epsilon=1.0):
    
        state = state.to(device)
        with torch.no_grad():
            q_values = self.qnetwork(state).detach()
        if random.random() < epsilon:

            _, _, _, _, cache = state.get_value()

            # Create a mask for entries where the cache is valid (assuming non-zero is valid)
            cache_empty_mask = torch.all(cache[:,1, :, :].view(cache.size(0), -1)==0, dim=1)  
        
            output = torch.tensor([[1, 0]]).repeat(self.batch_size, 1).to(device)# 0 是闭区间，2 是开区间
            
            # Broadcasting the specified output to match the shape of the valid cache entries
            pecified_output = torch.tensor([0, 1]).to(device)  # Specify the fixed output for valid cache entries
            output[cache_empty_mask] = pecified_output         
            # return torch.argmax(output, dim=1)
            return torch.argmin(output, dim=1)

        return torch.argmin(q_values, dim=1)

    def step(self, state, action, reward, next_state, done):
        state_slices = state.slice()
        if next_state is not None:
            next_state_slices = next_state.slice()
        for i in range(self.batch_size):
            # 对于元组 state 中的每个元素，分别取第 i 个样本
            # state_slices = tuple(s[i] for s in state)  # 对元组中的每个元素切片
            
            # 对 next_state 也做同样的处理，假设 next_state 也是一个元组
            if next_state is None:
                continue

            # if reward[i] == 0:
            #     continue
            # if reward[i] == 0:
            #     continue
            # else:
            #     next_state_slices = tuple(ns[i] for ns in next_state)
            self.memory.add(state_slices[i], action[i], reward[i], next_state_slices[i], done)

        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states_concat = Newdata.concat(states).to(device)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards).unsqueeze(-1).to(device)
        next_states_concat = Newdata.concat(next_states).to(device)
        dones = torch.tensor(dones).float().unsqueeze(-1).to(device)
        
        q_values = self.qnetwork(states_concat).gather(1, actions.unsqueeze(-1))
        next_q_values = self.target_network(next_states_concat).min(1)[0].detach().unsqueeze(1)
        # next_q_values = self.target_network(next_states_concat).detach().gather(1, actions.unsqueeze(-1))
        
        # print(rewards)
        target_q_values = (rewards * 10000 + (self.gamma * next_q_values * (1 - dones))).float()
        # print("q_values:", q_values)
        # print("rewards:",rewards)
        # print("target_q_values:", target_q_values)
        loss = nn.MSELoss()(q_values, target_q_values).float()
        
        # 反向传播
        if loss.requires_grad == True:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 训练代理
def train_dqn(agent, env, episodes=50, epsilon_start=0, epsilon_end=0.0, epsilon_decay=0.95):
    epsilon = epsilon_start
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        t = 0

        while not done:
            # print(f"Current time step: {t}")

            # action = agent.act(state, epsilon)
            action = agent.act(state, epsilon)

            next_state, reward, done = env.step(t, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            t += 1

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        # print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        if episode % 5 == 0:
            agent.update_target()

        # Save the model at regular intervals or after training
        if episode % 10 == 0 or episode == episodes - 1:
            save_model(agent.qnetwork, args.rlsave, episode)

    torch.save(agent.qnetwork.tensor0, 'tensor0.pt')
    torch.save(agent.qnetwork.tensor1, 'tensor1.pt')

def save_model(model, save_path, episode):
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    model_save_path = os.path.join(save_path, f"qnetwork_episode_{episode}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == '__main__':
    set_seed(args.seed)

    # load data
    bikevolume_train_save_path = os.path.join(args.bike_base_path, 'BV_train.npy')
    bikeflow_train_save_path = os.path.join(args.bike_base_path, 'BF_train.npy')
    taxivolume_train_save_path = os.path.join(args.taxi_base_path, 'TV_train.npy')
    taxiflow_train_save_path = os.path.join(args.taxi_base_path, 'TF_train.npy')

    bike_train_data = MyDataset_rl(bikevolume_train_save_path, args.window_size, args.batch_size)
    taxi_train_data = MyDataset_rl(taxivolume_train_save_path, args.window_size, args.batch_size)
    bike_adj_data = MyDataset_rl(bikeflow_train_save_path, args.window_size, args.batch_size)
    taxi_adj_data = MyDataset_rl(taxiflow_train_save_path, args.window_size, args.batch_size)

    bike_train_loader = DataLoader(
        dataset=bike_train_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    taxi_train_loader = DataLoader(
        dataset=taxi_train_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    bike_adj_loader = DataLoader(
        dataset=bike_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )
    taxi_adj_loader = DataLoader(
        dataset=taxi_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )


    base_smoe_config = SpatialMoEConfig(
            in_planes=2,
            out_planes=16,
            num_experts=32,
            gate_block=functools.partial(SpatialLatentTensorGate2d,
                                node_num = 231),
            save_error_signal=True,
            dampen_expert_error=False,
            unweighted=False,
            block_gate_grad=True,
        )
    model = Net_timesnet_sample_onetimesnet(args.batch_size, args.window_size, args.node_num, args.in_features, args.out_features, args.lstm_features, base_smoe_config, args.pred_size)
    model.to(device)
    model.load_state_dict(torch.load(args.save))
    model.eval()

    for iter, (bike_adj, bike_node, taxi_adj, taxi_node) in enumerate(zip(bike_adj_loader, bike_train_loader,
                                                                              taxi_adj_loader, taxi_train_loader)):

        # bike_adj, bike_node, taxi_adj, taxi_node = bike_adj.to(device), bike_node.to(device), taxi_adj.to(device), taxi_node.to(device)
        env = TrafficEnv(bike_adj, bike_node, taxi_adj, taxi_node, model, device, window_size=24, pred_size=args.pred_size)
        # 状态大小取决于数据窗口和误差
        agent = DQNAgent(device=device, state_size=462, batch_size=args.batch_size, action_size=2)
        # 训练强化学习代理        
        train_dqn(agent, env, episodes=50)