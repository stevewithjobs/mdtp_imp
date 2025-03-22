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
from models.expert_t import LstmAttention

from models.expert_s import Generator
from models.TimesNet import TimesBlock, Model, Model_moe, Model_withoutmoe, Model_moeconv, Model_moenew, Model_onetimenet





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
    def __init__(self, bike_node, bike_adj, taxi_node, taxi_adj, cache, action):
        self.bike_adj = bike_adj
        self.bike_node = bike_node
        self.taxi_adj = taxi_adj
        self.taxi_node = taxi_node
        self.batch_size = bike_node.size(0)
        self.cache = cache
        self.action = action
    
    def get_value(self):
        return self.bike_node, self.bike_adj, self.taxi_node, self.taxi_adj, self.cache, self.action
    
    def to(self, device):
        """Moves the data to the specified device (CPU or GPU)."""
        self.bike_node = self.bike_node.to(device)
        self.bike_adj = self.bike_adj.to(device)
        self.taxi_node = self.taxi_node.to(device)
        self.taxi_adj = self.taxi_adj.to(device)
        self.cache = self.cache.to(device)
        self.action = self.action.to(device)
        return self  # 返回自己以支持链式调用
    
    def slice(self):
        """Slices the data into batches based on the given batch size."""
        bike_batches = [b for b in torch.split(self.bike_node, 1)]
        bike_adj_batches = [b for b in torch.split(self.bike_adj, 1)]
        taxi_batches = [t for t in torch.split(self.taxi_node, 1)]
        taxi_adj_batches = [t for t in torch.split(self.taxi_adj, 1)]
        cache_batches = [c for c in torch.split(self.cache, 1)]
        action_batches = [c for c in torch.split(self.action, 1)]

        
        # 返回以元组为元素的列表，每个元组包含bike和taxi数据的切片
        return list(zip(bike_batches, bike_adj_batches, taxi_batches, taxi_adj_batches, cache_batches, action_batches))

    @staticmethod
    def concat(states, dim=0):
        """Concatenate a list of Newdata objects along the given dimension."""
        bike_nodes = torch.cat([state[0] for state in states], dim=dim)
        bike_adjs = torch.cat([state[1] for state in states], dim=dim)
        taxi_nodes = torch.cat([state[2] for state in states], dim=dim)
        taxi_adjs = torch.cat([state[3] for state in states], dim=dim)
        cache = torch.cat([state[4] for state in states], dim=dim)
        action = torch.cat([state[5] for state in states], dim=dim)
        
        return Newdata(bike_nodes, bike_adjs, taxi_nodes, taxi_adjs, cache, action)

def euclidean_distance(A, B):
    # A 和 B 需要有相同的形状
    return torch.sqrt(torch.sum((A - B) ** 2, dim=-1))

def cosine_similarity(A, B):
    return torch.nn.functional.cosine_similarity(A, B, dim=1)

class MultiTensorAttention(nn.Module):
    def __init__(self, feature_dim):
        super(MultiTensorAttention, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)  # Query 映射
        self.key_proj = nn.Linear(feature_dim, feature_dim)    # Key 映射
        self.value_proj = nn.Linear(feature_dim, feature_dim)  # Value 映射
        self.scale = feature_dim ** 0.5  # 缩放因子，防止数值过大

    def forward(self, tensors):
        # tensors: (B, N, F)，B是batch size，N是Tensor个数，F是特征维度
        batch_size, num_tensors, feature_dim = tensors.size()

        # Step 1: 投影 Query, Key, Value
        queries = self.query_proj(tensors)  # (B, N, F)
        keys = self.key_proj(tensors)      # (B, N, F)
        values = self.value_proj(tensors)  # (B, N, F)

        # Step 2: 计算注意力分数
        # 点乘计算：每个 Query 和 Key 的注意力分数 (B, N, N)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, N, N)
        attention_scores = attention_scores / self.scale  # 缩放因子

        # Step 3: 归一化注意力分数
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, N)
        # weights = torch.sum(attention_weights, dim=-1)

        # Step 4: 加权融合 Value
        output = torch.matmul(attention_weights, values)  # (B, N, F)
        # 如果需要最终输出为每个 Tensor 的加权和，可以进一步对 N 维度求和
        fused_output = torch.sum(output, dim=1)  # (B, F)

        return attention_scores

class Config:
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels, e_layers, c_out, batch_size):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.e_layers = e_layers
        self.c_out = c_out
        self.batch_size = batch_size
# Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, batch_size):
        super(QNetwork, self).__init__()

        self.rnn = LstmAttention(
            node_size=4,
            input_size=1,
            hidden_dim=16,
            n_layers=2,
        )

        self.generator1 = Generator(
            batch_size=batch_size,
            window_size=24,
            node_num=231,
            in_features=2,
            out_features=8
        )
        timesnetconfig = Config(
            seq_len=24, 
            pred_len=4,
            top_k=1, 
            d_model=231 * 16, 
            d_ff=231 * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=231 * 4,
            batch_size=batch_size
        )
        self.timesnet = Model_onetimenet(
             timesnetconfig,
            #  smoe_config,
        )
        self.fc1 = nn.Sequential(
            nn.Linear(231 * 4, 231),
            nn.ReLU(),
            nn.Linear(231, 32),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(36, 2)
        )
        self.batch_size = batch_size
        self.window_size = 24


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
        self.model = Net_timesnet_sample_onetimesnet(self.batch_size, 24, args.node_num, args.in_features, args.out_features, args.lstm_features, base_smoe_config, args.pred_size)
        self.model.to(device)
        self.model.load_state_dict(torch.load(args.save))
        self.model.eval()

        # self.intermediate_outputs = []
        # def hook(module, input, output):
        #     self.intermediate_outputs.append(output)
        # # 假设你要获取模型中 `timesnet` 的输出
        # self.model.timesnet.register_forward_hook(hook)
        # self.last_hook = None

        self.attention = MultiTensorAttention(231*4)

    def forward(self, state):

        bike_node, bike_adj, taxi_node, taxi_adj, cache, action = state.get_value()

        data = (bike_node, bike_adj, taxi_node, taxi_adj)

        # 执行一次前向传播
        bike_start, bike_end, taxi_start, taxi_end = self.model(data)
        bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(-1), bike_end.unsqueeze(-1), \
                                                        taxi_start.unsqueeze(-1), taxi_end.unsqueeze(-1)
        new_predict = torch.cat((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
        new_predict = new_predict[:, 1, :, :].view(self.batch_size, 1, -1)
        old_predict = cache.view(self.batch_size, 4, -1)
        # timesnetout = self.intermediate_outputs[0][0]
        # self.intermediate_outputs.clear()
        # print(intermediate_outputs[0])
        # gcn_output1, gcn_output2 = self.generator1(bike_node, bike_adj, taxi_node, taxi_adj, 0)
        # gcn_output = torch.cat((gcn_output1, gcn_output2), dim=-1)
        # gcn_output = gcn_output.view(self.batch_size, self.window_size, -1)
        # timesnetout, _ = self.timesnet(bike_node, gcn_output)
        # if self.last_hook is None:
        #     self.last_hook = timesnetout
        #     return torch.zeros((self.batch_size,2),device=device)

        # output1 = self.fc1(self.last_hook[:, 1, :] + timesnetout[:, 0, :])
        # att_input = torch.cat((self.last_hook[:, 1, :].unsqueeze(1), timesnetout[:, 0, :].unsqueeze(1)), dim=1)
        att_input = torch.cat((old_predict, new_predict), dim=1)
        attention_weights = self.attention(att_input)
        
        output = attention_weights[:, -1, :]
        # # Step 1: 生成非对角掩码 (n, n)
        # non_diag_mask = 1 - torch.eye(2)  # 创建一个对角元素为 0 的掩码矩阵
        # non_diag_mask = non_diag_mask.unsqueeze(0).expand(self.batch_size, -1, -1).to(device)  # 扩展到 (B, n, n)

        # # Step 2: 应用掩码，保留非对角元素
        # non_diag_attention = attention_weights * non_diag_mask  # (B, n, n)

        # # Step 3: 沿最后一个维度求和，得到每个 Tensor 的相对重要性 (B, n)
        # output = non_diag_attention.sum(dim=2)

        # self.last_hook = timesnetout5
        # timesnetout = torch.sum(timesnetout, dim = 1)
        # output1 = self.fc1(timesnetout)

        # action = action.unsqueeze(-1)
        # output2 = self.rnn(action)


        # output = torch.cat((output1, output2), dim=1)
        # output = self.fc2(output)

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

# class ReplayBuffer:
#     def __init__(self, buffer_size):
#         # 分别存储 action=0 和 action=1 的经验
#         self.memory_0 = deque(maxlen=buffer_size // 2)  # 存储 action=0
#         self.memory_1 = deque(maxlen=buffer_size // 2)  # 存储 action=1

#     def add(self, state, action, reward, next_state, done):
#         # 根据 action 值将经验存入对应的队列
#         if action == 0:
#             self.memory_0.append((state, action, reward, next_state, done))
#         elif action == 1:
#             self.memory_1.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         half_batch = batch_size // 2

#         # 先从两个队列中各采样一半的数量
#         experiences_0 = random.sample(self.memory_0, min(len(self.memory_0), half_batch))
#         experiences_1 = random.sample(self.memory_1, min(len(self.memory_1), half_batch))

#         # 如果其中一个队列不足 half_batch，则从另一个队列补足
#         if len(experiences_0) < half_batch:
#             experiences_1 += random.sample(self.memory_1, half_batch - len(experiences_0))
#         elif len(experiences_1) < half_batch:
#             experiences_0 += random.sample(self.memory_0, half_batch - len(experiences_1))

#         # 合并并打乱采样的经验
#         experiences = experiences_0 + experiences_1
#         random.shuffle(experiences)

#         # 返回采样的经验，解包为 states, actions, rewards, next_states, dones
#         states, actions, rewards, next_states, dones = zip(*experiences)
#         return states, actions, rewards, next_states, dones

#     def __len__(self):
#         # 总的样本数为两个队列的长度之和
#         return len(self.memory_0) + len(self.memory_1)

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
        self.cache = torch.zeros(self.batch_size, self.pred_size, self.pred_size, 231, 4)
        self.action = torch.zeros(self.batch_size, 4).to(device)

        self.loss = mdtp.mae_rlerror

        self.action_count = 0

    def reset(self):
        # Reset environment, e.g., set the initial state
        # self.cache = []
        # # 添加多个空的队列到列表中
        # for _ in range(self.batch_size):
        #     self.cache.append([])
        self.cache = torch.zeros((self.batch_size, self.pred_size, self.pred_size,  231, 4), device=device)
        self.action = torch.zeros((self.batch_size, 4), device=device)
        return self.get_state(0)

    def get_state(self, t):
        # Get state at time t

        if t + self.window_size >= self.total_length:   
            return None
        
        new_data = Newdata( bike_node=self.bike_node[:, t:t + self.window_size, :, :],\
                            bike_adj=self.bike_adj[:, t:t + self.window_size, :, :],\
                            taxi_node=self.taxi_node[:, t:t + self.window_size, :, :],\
                            taxi_adj=self.taxi_adj[:, t:t + self.window_size, :, :],\
                            cache=self.cache[:, :, t].clone(),\
                            action=self.action[:, -4:])
        # new_data = Newdata( bike_node=self.bike_node[:, t + self.window_size - 4:t + self.window_size, :, :],\
        #                     bike_adj=self.bike_adj[:, t + self.window_size - 4:t + self.window_size, :, :],\
        #                     taxi_node=self.taxi_node[:, t + self.window_size - 4:t + self.window_size, :, :],\
        #                     taxi_adj=self.taxi_adj[:, t + self.window_size - 4:t + self.window_size, :, :],\
                            # cache=self.cache[:, t-3:t+2].clone() )    
        # new_data = Newdata( bike_node=self.bike_node[:, t:t + self.window_size, :, :],\
        #                     bike_adj=self.bike_adj[:, t:t + self.window_size, :, :],\
        #                     taxi_node=self.taxi_node[:, t:t + self.window_size, :, :],\
        #                     taxi_adj=self.taxi_adj[:, t:t + self.window_size, :, :],\
        #                     cache=self.cache[:, t-23:t+2].clone() )
        # new_data = new_data.to(device)
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
        

        predloss1, _, predloss2, _ = self.get_acc(t, model_pred)
        cacheloss1, _, cacheloss2, _ = self.get_acc(t, cache_pred)

        # if predloss1[0] > cacheloss1[0] and predloss2[0] > cacheloss2[0]:
        if predloss2[1] > cacheloss2[1]:
            pred_final = cache_pred
            self.action_count += 1
            
        else:
            pred_final = model_pred
        print(self.action_count)

        next_state = self.get_state(t + 1)
        done = (next_state is None)


        t1 = time.time()
        t2 = time.time()
        model_time = t2 -t1
        return next_state, pred_final, done, model_time

    # def steptest(self, t, action):
    #     # if torch.all(self.cache[:,t+1, :, :].view(1, -1)==0, dim=1):
    #     #     action = torch.zeros((1), device=device)
    #     if action == 0:
    #         data = (self.bike_node[:, t:t + self.window_size, :, :].contiguous(), 
    #                 self.bike_adj[:, t:t + self.window_size, :, :].contiguous(), 
    #                 self.taxi_node[:, t:t + self.window_size, :, :].contiguous(), 
    #                 self.taxi_adj[:, t:t + self.window_size, :, :].contiguous())
            
    #         t1 = time.time()
    #         with torch.no_grad():
    #             bike_start, bike_end, taxi_start, taxi_end = self.model(data)
    #         t2 = time.time()
    #         model_time = t2 - t1
    #         bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(-1), bike_end.unsqueeze(-1), \
    #                                                     taxi_start.unsqueeze(-1), taxi_end.unsqueeze(-1)
    #         prediction = torch.cat((bike_start, bike_end, taxi_start, taxi_end), dim=-1)

    #         # prediction = torch.cat((self.bike_node[:, t + self.window_size:t + self.window_size + 4, :, :].to(device),  \
    #         #                 self.taxi_node[:, t + self.window_size:t + self.window_size + 4, :, :].to(device)), dim=-1)
            
    #         cache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
    #         cache[action==0, :, :, :] = torch.cat((self.cache[action==0, :t+1, :, :], prediction[action==0, :, :, :]), dim=1)
    #         # t2 = t1 =0
    #     elif action == 1:
    #         # t1 = time.time()
    #         cache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
    #         zero = torch.zeros((self.batch_size, 1, 231, 4), device=device)
    #         cache[action==1] = torch.cat((self.cache[action==1, :, :, :], zero[action==1, :, :, :]), dim=1)
    #         model_time = 0
    #         # t2 = time.time()
    
    #     self.cache = cache.clone()
    #     pred_final = self.cache[:, t + 1]   
    #     # pred_final = prediction[:, 0, :, :]
    #     next_state = self.get_state(t + 1)
    #     done = (next_state is None)
        
    #     return next_state, pred_final, done, model_time

    def steptest(self, t, action):
        # if torch.all(self.cache[:,t+1, :, :].view(1, -1)==0, dim=1):
        #     action = torch.zeros((1), device=device)
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

        cache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
        cache[:, :, :, :] = torch.cat((self.cache[:, :t+1, :, :], prediction[:, :, :, :]), dim=1)
        
        if action == 0:
           pred_final = prediction[:, 0, :, :]

        elif action == 1:
            # t1 = time.time()
            pred_final = self.cache[:, t + 1]
            # t2 = time.time()
        # get right action
        model_pred = prediction[:, 0, :, :]
        cache_pred = self.cache[:, t + 1]
        labels = torch.cat((self.bike_node[:, t + self.window_size, :, :].to(device), self.taxi_node[:, t + self.window_size, :, :].to(device)), dim=-1)
        _, right_action = self._get_reward(cache_pred, model_pred, labels, action)
        self.action = torch.cat((self.action, right_action.unsqueeze(1)), dim=1)
        # pred_final = prediction[:, 0, :, :]
        
        self.cache = cache.clone()

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
        action = action.to(torch.int64)

        model_pred = prediction[:, 0, :, :]
        index = action.view(self.batch_size, 1, 1, 1).expand(-1, 1, 231, 4)
        cache_pred = torch.gather(self.cache[:, :, t, :], dim=1, index=index).squeeze(1)

        # newcache = torch.zeros((self.batch_size, t + self.pred_size + 1, 231, 4), device=device)
        # # newcache[action==0] = torch.cat((self.cache[action==0, :t+1, :, :], prediction[action==0, :, :, :]), dim=1)
        # # newcache[action==1] = torch.cat((self.cache[action==1, :t+2, :, :], prediction[action==1, 1:, :, :]), dim=1)
        # newcache = torch.cat((self.cache[:, :t+1, :, :], prediction[:, :, :, :]), dim=1)


        # newcache[action==0] = torch.cat((self.cache[action==0, :t+1, :, :], prediction[action==0, :, :, :]), dim=1)
        # zero = torch.zeros((self.batch_size, 1, 231, 4), device=device)
        # newcache[action==1] = torch.cat((self.cache[action==1, :, :, :], zero[action==1, :, :, :]), dim=1)
        
        emptyfill = torch.zeros((self.batch_size, self.pred_size, 1, 231, 4), device=device)
        self.cache = torch.cat((self.cache, emptyfill), dim=2)
        self.cache[:, t%4, t:t+4, :, :] = prediction 
        
        # pred_final = self.cache[:, t + 1]
        
        # 奖励设计: 根据误差的反向5
        labels = torch.cat((self.bike_node[:, t + self.window_size, :, :].to(device), self.taxi_node[:, t + self.window_size, :, :].to(device)), dim=-1)
        # reward = self._get_reward(pred_final, prediction[:, 0, :, :], labels, action)
        reward, right_action = self._get_reward(cache_pred, model_pred, labels, action)

        self.action = torch.cat((self.action, right_action.unsqueeze(1)), dim = 1)


        next_state = self.get_state(t + 1)
        done = (next_state is None)

        return next_state, reward, done
    
    # def _get_reward(self, pred_final, pred_model, lables, action):
    #     # acc reward
    #     pred_final = pred_final.to(device)
    #     pred_model = pred_model.to(device)
    #     lables = lables.to(device)

    #     # final_loss = self.loss(pred_final[..., :2], lables[..., :2])
    #     # pred_loss = self.loss(pred_model[..., :2], lables[..., :2])

    #     final_loss = self.loss(pred_final[:, :, :4], lables[:, :, :4])
    #     pred_loss = self.loss(pred_model[:, :, :4], lables[:, : ,:4])
    #     error = pred_loss - final_loss
    #     # error = final_loss
    #     # error = final_loss - pred_loss
    #     # eff reward
    #     error_0 = error.clone()
    #     error_0[:] = -0.0001
         
    #     error_1 = error.clone()
    #     error_1[error_1>0] = error_1[error_1>0] * 10
    #     # error_1[error_1<0] = error_1[error_1<0]

    #     error[action==0]=error_0[action==0]
    #     error[action==1]=error_1[action==1]  

    #     return error

    def _get_reward(self, cache_pred, model_pred, lables, action):
        # acc reward
        cache_pred = cache_pred.to(device)
        model_pred = model_pred.to(device)
        lables = lables.to(device)

        # final_loss = self.loss(pred_final[..., :2], lables[..., :2])
        # pred_loss = self.loss(pred_model[..., :2], lables[..., :2])

        cache_loss = self.loss(cache_pred[:, :, :1], lables[:, :, :1])
        model_loss = self.loss(model_pred[:, :, :1], lables[:, : ,:1])

        # cache_empty_mask = torch.all(cache_pred.view(cache_pred.size(0), -1)==0, dim=1)  

        error = cache_loss - model_loss
        # error[error < 0] *= 2
        # error *= 100
        # error = torch.where((error > 10**-4), 10**-4, error)
        # error = torch.where((error < 0) & (error > -10**-4), -10**-4, error)


        # negative_values = error[error < 0]  # 筛选出大于 0 的元素
        # median_value = torch.median(negative_values) 
        # # median_value = torch.median(error_1)
        # min_value = torch.min(error)
        # 使用条件替换所有小于中位数的元素
        
        # error[error < 0] = error[error < 0] 
        # error[cache_empty_mask] = 0
        # count_neg = (error < 0).sum().item()
        # count_pos = (error > 0).sum().item()
        # error[error < 0]
        # error[error < 0] = error[error < 0] * count_pos
        # error[error > 0] = error[error > 0] * count_neg
        # error[error < 0] = error[error < 0] * count_pos
        # error[error > 0] = error[error > 0] * count_neg

        
        error_0 = error.clone()
        # error_0 = torch.where((error_0 > 10**-4), 10**-4, error_0)
        # error_0 = torch.where((error_0 < 0) & (error_0 > -10**-3), -10**-3, error_0)


        # error_0[error_0<0] = -2
        # error_0[error_0>0] = 1
        # error_0[error_0 > 0] = error_0[error_0 > 0]
        # error_0[error_0 < 0] = error_0[error_0 < 0] * 100
        # error_0[error_0 > 0] = 1
        # error_0[error_0 < 0] = -1
    
        error_1 = -error.clone()
        # error_1 = torch.where((error_1 > 0) & (error_1 < 10**-3), 10**-3, error_1)
        # error_0 = torch.where((error_0 < -10**-4), -10**-4, error_0)
        # error_1[error_1<0] = -1
        # error_1[error_1>0] = 1
        # error_1[error_1 > 0] = error_1[error_1 > 0] * 10
        # error_1[error_1 < 0] = error_1[error_1 < 0]
        # error_1[error_1 > 0] = 1
        # error_1[error_1 < 0] = -1

        error = torch.zeros_like(error_0)
        right_action = torch.where(error_0 < 0, torch.tensor(1, device=device), torch.tensor(0, device=device))
        error[action == 1] = error_1[action == 1]
        error[action == 0] = error_0[action == 0]

        return error, right_action
        # return error, action
    
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
    def __init__(self, state_size, action_size, device, buffer_size=320, batch_size=32, gamma=1, lr=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork = QNetwork(state_size, action_size, batch_size).to(device)
        self.target_network = QNetwork(state_size, action_size, batch_size).to(device)
        self.target_network.eval()
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.memory = ReplayBuffer(int(buffer_size))
        # self.memory_1 = ReplayBuffer(int(buffer_size/2))

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
        # if random.random() < epsilon:

        #     _, _, _, _, cache = state.get_value()

        #     # Create a mask for entries where the cache is valid (assuming non-zero is valid)
        #     cache_empty_mask = torch.all(cache[:,1, :, :].view(cache.size(0), -1)==0, dim=1)  
        
        #     output = torch.tensor([[1, 0]]).repeat(self.batch_size, 1).to(device)# 0 是闭区间，2 是开区间
            
        #     # Broadcasting the specified output to match the shape of the valid cache entries
        #     pecified_output = torch.tensor([0, 1]).to(device)  # Specify the fixed output for valid cache entries
        #     output[cache_empty_mask] = pecified_output         
        #     # return torch.argmax(output, dim=1)
        #     return torch.argmin(output, dim=1)

        return torch.argmax(q_values, dim=1)

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
            # if action[i] == 0:
            #     self.memory_0.add(state_slices[i], action[i], reward[i], next_state_slices[i], done)
            # if action[i] == 1:
            self.memory.add(state_slices[i], action[i], reward[i], next_state_slices[i], done)

        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)

            # # 将两个采样的列表合并
            # experiences = experiences_0 + experiences_1
            
            # # 打乱合并后的列表顺序
            # random.shuffle(experiences)
            
            # 进行学习
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states_concat = Newdata.concat(states).to(device)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards).unsqueeze(-1).to(device)
        next_states_concat = Newdata.concat(next_states).to(device)
        dones = torch.tensor(dones).float().unsqueeze(-1).to(device)
        
        q_values = self.qnetwork(states_concat).gather(1, actions.unsqueeze(-1))
        next_q_values = self.target_network(next_states_concat).max(1)[0].detach().unsqueeze(1)
        # next_q_values = self.target_network(next_states_concat).detach().gather(1, actions.unsqueeze(-1))
        
        # print(rewards)
        with torch.no_grad():
            target_q_values = (rewards + (self.gamma * next_q_values * (1 - dones))).float()
        # print("q_values:", q_values)
        # print("rewards:",rewards)
        # print("target_q_values:", target_q_values)
        loss = nn.MSELoss()(q_values, target_q_values).float()
        # print(loss)
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
            if t > 3 :
                action = agent.act(state, epsilon)
            else:
                 action = torch.zeros((32), device=device)

            next_state, reward, done = env.step(t, action)
            # if t > 4 :
            #     agent.step(state, action, reward, next_state, done)
            if t > 3:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            t += 1

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")  

        if episode % 5 == 0:
            agent.update_target()

        # Save the model at regular intervals or after training
        if episode % 1 == 0 or episode == episodes - 1:
            save_model(agent.qnetwork, args.rlsave, episode)
    print(agent.qnetwork.time)
    # torch.save(agent.qnetwork.tensor0, 'tensor0.pt')
    # torch.save(agent.qnetwork.tensor1, 'tensor1.pt')

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
        train_dqn(agent, env, episodes=500)