import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
from collections import deque
from utils.mdtp import MyDataset_mmsm, set_seed, metric1
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

import torch.distributions as distributions





parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:4', help='GPU setting')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
# parser.add_argument('--save', type=str, default='/home/haitao/data/web_host_network/mdtp_moe/whnbest/best_model.pth', help='save path')
parser.add_argument('--save', type=str, default='/home_nfs/haitao/data/web_host_network/mdtp_moe/log_2024-12-22_22-54-12/best_model.pth', help='save path')
parser.add_argument('--rlsave', type=str, default='./mmsm_model/', help='save path')
parser.add_argument('--smoe_start_epoch', type=int, default=99, help='smoe start epoch')
parser.add_argument('--gpus', type=str, default='4', help='gpu')
parser.add_argument('--log', type=str, default='0.log', help='log name')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def custom_collate_fn(batch):
    # 因为 batch 是一个长度为 1 的列表，直接取出这个列表的第一个元素
    return batch

class MultiArmedBanditEnv:
    def __init__(self, num_arms, batch_size, pred_size, data_type='val'):
        """
        初始化多臂老虎机环境。

        :param num_arms: 动作数量（臂的数量）
        :param data_type: 数据类型，'train' 或 'test'
        """
        self.num_arms = num_arms
        self.batch_size = batch_size
        self.pred_size = pred_size

        # 检查 data_type 是否有效
        assert data_type in ['train', 'val', 'test'], "data_type 必须为 'train' 或 'test'"

        # 根据 data_type 动态加载数据
        bikevolume_save_path = os.path.join(args.bike_base_path, f'BV_{data_type}.npy')
        bikeflow_save_path = os.path.join(args.bike_base_path, f'BF_{data_type}.npy')
        taxivolume_save_path = os.path.join(args.taxi_base_path, f'TV_{data_type}.npy')
        taxiflow_save_path = os.path.join(args.taxi_base_path, f'TF_{data_type}.npy')

        bike_train_data = MyDataset_mmsm(bikevolume_save_path, args.window_size, batch_size)
        taxi_train_data = MyDataset_mmsm(taxivolume_save_path, args.window_size, batch_size)
        bike_adj_data = MyDataset_mmsm(bikeflow_save_path, args.window_size, batch_size)
        taxi_adj_data = MyDataset_mmsm(taxiflow_save_path, args.window_size, batch_size)

        self.bike_train_loader = DataLoader(
            dataset=bike_train_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        self.taxi_train_loader = DataLoader(
            dataset=taxi_train_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        self.bike_adj_loader = DataLoader(
            dataset=bike_adj_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        self.taxi_adj_loader = DataLoader(
            dataset=taxi_adj_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        # cache
        self.cache = torch.zeros((batch_size, pred_size, pred_size, 231, 4),device=device)
        self.timesnet_cache = torch.zeros((batch_size, pred_size, pred_size, 231 * 4),device=device)

        # out_node
        self.out_node = None

        # loss
        self.loss = mdtp.mae_rlerror

    def _getdata(self):
        try:
            # 如果当前迭代器耗尽，重新创建迭代器
            if not hasattr(self, 'bike_adj_loader_iter'):
                self.bike_adj_loader_iter = iter(self.bike_adj_loader)
                self.bike_train_loader_iter = iter(self.bike_train_loader)
                self.taxi_adj_loader_iter = iter(self.taxi_adj_loader)
                self.taxi_train_loader_iter = iter(self.taxi_train_loader)

            # 从每个迭代器中获取数据
            bike_in_adj, bike_out_adj = next(self.bike_adj_loader_iter)
            bike_in_node, bike_out_node = next(self.bike_train_loader_iter)
            taxi_in_adj, taxi_out_adj = next(self.taxi_adj_loader_iter)
            taxi_in_node, taxi_out_node = next(self.taxi_train_loader_iter)
            self.out_node = torch.cat((bike_out_node.to(device), taxi_out_node.to(device)), dim=-1)
            return (bike_in_node.to(device), bike_in_adj.to(device), taxi_in_node.to(device), taxi_in_adj.to(device))
        except StopIteration:
            # 如果任何一个迭代器耗尽，重置所有迭代器
            self.bike_adj_loader_iter = iter(self.bike_adj_loader)
            self.bike_train_loader_iter = iter(self.bike_train_loader)
            self.taxi_adj_loader_iter = iter(self.taxi_adj_loader)
            self.taxi_train_loader_iter = iter(self.taxi_train_loader)
            return None
    
    def _getcache(self):
        return self.cache.detach()
    
    def _gettimesnetcache(self):
        return self.timesnet_cache.detach()
    
    def reset(self):
        data = self._getdata()
        self.cache = torch.zeros((self.batch_size, self.pred_size, self.pred_size, 231, 4),device=device)
        cache = self._getcache()
        return data, cache
    
    def _getreward(self, action, newcache):
        cache_loss1 = self.loss(newcache[:, :, 0, :, 0], self.out_node[:, :, 0].unsqueeze(1).expand(-1, 4, -1))      
        cache_loss2 = self.loss(newcache[:, :, 0, :, 2], self.out_node[:, :, 2].unsqueeze(1).expand(-1, 4, -1))      
        # # mean = torch.mean(cache_loss, dim=1, keepdim=True)
        # mean = cache_loss[:,0].unsqueeze(1)
        # std = torch.sqrt(((cache_loss - mean) ** 2).mean(dim=1, keepdim=True))
        # cache_loss = (cache_loss - mean)/std
        # reward = nn.Softmax(dim=-1)(-cache_loss)
        # # reward -= 0.25
        # # # reward = torch.where(cache_loss < cache_loss[:, 0].unsqueeze(-1), 1.0, 0.0)
        # # # reward = (cache_loss - cache_loss[:, 0].unsqueeze(-1)) < 0
        # # # reward = reward.float()
        # # # reward[:, 0] = 0.5
        # return reward

        # mean = torch.mean(cache_loss, dim=1, keepdim=True)
        # std = torch.sqrt(((cache_loss - mean) ** 2).mean(dim=1, keepdim=True))
        # cache_loss = (cache_loss - mean)/std
        # reward = nn.Softmax(dim=1)(-cache_loss)
        # return reward

        # weights = torch.tensor([1, 1, 1.5, 1.5], device=device)
        # reward = torch.where(reward > 0, reward * weights, reward)
        min_indices1 = torch.argmin(cache_loss1, dim=1)
        min_indices2 = torch.argmin(cache_loss2, dim=1)

        return min_indices1, min_indices2
        # reward = min_indices == action
        # reward = torch.zeros_like(cache_loss)  # 初始化为 0
        # reward[torch.arange(self.batch_size), min_indices] = 1 
        # 计算newcache中每个的loss，找最小，如果和action相同，reward=1 否则=0
        # return reward.float()
        
        # return min_indices

    
    def _updatecache(self, newcache):
        newcache = torch.roll(newcache, 1, dims=1)
        newcache = torch.roll(newcache, -1, dims=2)
        self.cache = newcache
        return 

    # def step(self, action, newcache):
    #     # 获取reward
    #     reward1, reward2 = self._getreward(action, newcache)
    #     # 更新cache
    #     self._updatecache(newcache)
    #     cache = self._getcache()
    #     # 获取data
    #     data = self._getdata()
    #     return reward1, reward2, data, cache
    def step(self, new_cache):
        # update cache
        self._updatecache(new_cache)
        cache = self._getcache()

        # 获取data
        data = self._getdata()
        return data, cache

    # def steptest(self, action, newcache):
    #     # 更新cache
    #     self._updatecache(newcache)
    #     cache = self._getcache()
    #     # 获取data
    #     data = self._getdata()
    #     return data, cache
    
    def getrightaction(self, action, newcache):
        cache_loss = self.loss(newcache[:, :, 0, :, 0], self.out_node[:, :, 0].unsqueeze(1).expand(-1, 4, -1))      
        rightaction = torch.argmin(cache_loss, dim=1)
        _, sortaction = torch.sort(cache_loss, dim=-1, descending=False)
        return sortaction
    
    def get_acc(self, pred):
        # pred = pred[:, :, 0, :, :].view(231, 4)
        # pred = pred[:, 0, :, :].view(231, 4)
        pred = pred.view(16, 231, 4)
        # pred = pred[0, 0, :, :]
        out_node = self.out_node.view(16, 231, 4)
        bk_start_mask = pred[:, :, 0]!= out_node[:, :, 0]
        bk_end_mask = pred[:, :, 1] != out_node[:, :, 1]
        tx_start_mask = pred[:, :, 2] != out_node[:, :, 2]
        tx_end_mask = pred[:, :, 3] != out_node[:, :, 3]
        
        bike_start_metrics = metric1(pred[:, :, 0], out_node[:, :, 0], bk_start_mask)
        bike_end_metrics = metric1(pred[:, :, 1], out_node[:, :, 1], bk_end_mask)
        taxi_start_metrics = metric1(pred[:, :, 2], out_node[:, :, 2], tx_start_mask)
        taxi_end_metrics = metric1(pred[:, :, 3], out_node[:, :, 3], tx_end_mask)

        return bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics
    
    def get_acc_train(self, pred):
        # pred = pred[:, :, 0, :, :].view(231, 4)
        # pred = pred[:, 0, :, :].view(self.batch_size, 231, 4)
        # pred = pred[0, 0, :, :]
        out_node = self.out_node.view(self.batch_size, 231, 4)
        bk_start_mask = pred[:, :, 0]!= out_node[:, :, 0]
        bk_end_mask = pred[:,:,1] != out_node[:,:, 1]
        tx_start_mask = pred[:,:, 2] != out_node[:,:, 2]
        tx_end_mask = pred[:,:, 3] != out_node[:,:, 3]
        
        bike_start_metrics = mdtp.rmse(pred[:,:, 0], out_node[:,:, 0], bk_start_mask)
        bike_end_metrics = mdtp.rmse(pred[:,:, 1], out_node[:,:, 1], bk_end_mask)
        taxi_start_metrics = mdtp.rmse(pred[:,:, 2], out_node[:,:, 2], tx_start_mask)
        taxi_end_metrics = mdtp.rmse(pred[:,:, 3], out_node[:,:, 3], tx_end_mask)

        return bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics

    
class MultiTensorAttention(nn.Module):
    def __init__(self, batch_size):
        super(MultiTensorAttention, self).__init__()
        self.batch_size = batch_size
        # self.query_proj = nn.Linear(feature_dim, feature_dim)  # Query 映射
        # self.key_proj = nn.Linear(feature_dim, feature_dim)    # Key 映射
        # self.value_proj = nn.Linear(feature_dim, feature_dim)  # Value 映射

        # self.query_proj = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5)
        # self.key_proj = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5)
        # self.value_proj = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)

        # 使用 nn.Sequential 将多层组合到一起
        self.query_proj = nn.Sequential(
            # 第一层：通道数从 4 -> 16
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            
            # # 第二层：通道数从 16 -> 4
            # nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(4)
        )
        self.key_proj = nn.Sequential(
            # 第一层：通道数从 4 -> 16
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            
            # # 第二层：通道数从 16 -> 4
            # nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(4)
        )
        self.q_k_proj = nn.Sequential(
            # 第一层：通道数从 4 -> 16
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
        self.tt = 0
    # def forward(self, q, k, v):
    def forward(self, q, k, v):
        t1= time.time()
        # bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = q
        # bike_node = bike_node_ori[:,-4:,:,:]
        # bike_adj = bike_adj_ori[:,-4:,:,:]
        # taxi_node = taxi_node_ori[:,-4:,:,:]
        # taxi_adj = taxi_adj_ori[:,-4:,:,:]

        # tensors: (B, N, F)，B是batch size，N是Tensor个数，F是特征维度
        # batch_size, _, _ = q.size()
        q = q.reshape(self.batch_size, 4, 11, 21)
        k = k.reshape(self.batch_size, 4, 11, 21)
        
        
        # Step 1: 投影 Query, Key, Value
        # queries = self.query_proj(q)  # (B, N, F)
        # keys = self.key_proj(k)      # (B, N, F)
        # queries = queries.reshape(self.batch_size, 4, -1)
        # keys = keys.reshape(self.batch_size, 4, -1)
        q_k= torch.cat((q, k), dim=1)
        queries_keys = self.q_k_proj(q_k)
        queries = queries_keys[:,:4,:,:].reshape(self.batch_size, 4, -1)
        keys = queries_keys[:,4:,:,:].reshape(self.batch_size, 4, -1)
        t2 = time.time()
        self.tt += t2 - t1
        print("tt111:", self.tt)
        
        # Step 2: 计算注意力分数
        # 点乘计算：每个 Query 和 Key 的注意力分数 (B, N, N)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, N, N)
       
        d_k = queries.size(-1)
        attention_scores /= torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # 缩放
        
        # scores = torch.mean(attention_scores, dim=1)
        # scores = self.fc(scores)
        # scores = nn.Sigmoid()(scores)
        # # Step 3: 归一化注意力分数
        attention_weights = nn.Softmax(dim=-1)(attention_scores)  # (B, N, N)
        weights = torch.mean(attention_weights, dim=1)
        
        
        return weights

class BanditNet(nn.Module):
    def __init__(self, num_arms, model_list, batch_size, pred_size):
        super(BanditNet, self).__init__()
        self.batch_size = batch_size
        self.pred_size = pred_size
        self.lstm = nn.LSTM(input_size=231, hidden_size=32, num_layers=1, batch_first=True)
        self.decide = nn.Linear(in_features=32, out_features=6)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_state = None
        self.model_list = model_list

        self.fc = nn.Linear(num_arms, num_arms)  # 简单的线性网络
        self.attention1 = MultiTensorAttention(batch_size)
        self.attention2 = MultiTensorAttention(batch_size)
        # self.attention = nn.MultiheadAttention(embed_dim=231, num_heads=3, dropout=0.1, batch_first=True)
        # self.decisionT = DecisionTransformer(seq_features=231, 
        #                                      pred_features=231, 
        #                                      model_dim=231*8, 
        #                                      num_heads=8, 
        #                                      num_layers=2)
        self.tt1 = 0
        self.tt2 = 0
        self.t = 0
        self.tt = 0

    # def forward(self, data, cache, epsilon):
        
    #     lstm_input = torch.cat((data[0], data[2]), dim=-1)
    #     lstm_input = torch.sum(lstm_input, dim=-1)
    #     lstm_input = lstm_input[:, -1, :].reshape(self. batch_size, -1)
    #     lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
    #     decisoin_q = torch.mean(self.decide(lstm_out), dim=0)
    #     action_probs = self.softmax(decisoin_q)

    #     # if random.random() < epsilon:  # 以 ε 的概率进行随机探索
    #     #     decision = torch.randint(0, decisoin_q.size(0), (1,))  # 随机选择动作
    #     # else:

    #     # action_dist = distributions.Categorical(action_probs)
    
    #     # # 根据概率分布采样一个动作
    #     # action = action_dist.sample()
    #     action = torch.argmax(action_probs)

    #     self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())

    #     with torch.no_grad():
    #         bike_start, bike_end, taxi_start, taxi_end = self.model_list[action](data, (action+1)*4)
    #     new_predict = torch.stack((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
    #     updated_cache = torch.cat((new_predict.unsqueeze(1), cache[:, -3: :, : ,:]), dim=1)


    #     with torch.no_grad():
    #         bike_start, bike_end, taxi_start, taxi_end = self.model_list[-1](data, 24)
    #     st_predict = torch.stack((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
    #     return new_predict[:, 0, :, :], st_predict[:, 0, :, :], action_probs[action], updated_cache

    def forward(self, data, cache):
        t1 = time.time()
        t1 = time.time()
        lstm_input = torch.cat((data[0], data[2]), dim=-1)
        lstm_input = torch.sum(lstm_input, dim=-1)
        lstm_input = lstm_input[:, -1, :].reshape(self. batch_size, -1)
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        
        self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())
        t2 = time.time()
        if self.t > 1:
            self.tt1 += t2-t1
        print("tt1:", self.tt1)
        
        
        decisoin_q = torch.mean(self.decide(lstm_out), dim=0)

        action_probs = self.softmax(decisoin_q)

        
        # print(action_probs)
        
        action = torch.argmax(action_probs).item()
        # action = 2
        

        t1 = time.time()
        with torch.no_grad():
            bike_start, bike_end, taxi_start, taxi_end = self.model_list[action](data, (action+1)*4)
        new_predict = torch.stack((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
        t2 = time.time()
        if self.t > 1:
            self.tt2 += t2-t1
        print("tt2:", self.tt2)

        updated_cache = torch.cat((new_predict.unsqueeze(1), cache[:, -3: :, : ,:]), dim=1)

        self.t += 1
        return new_predict[:, 0, :, :], updated_cache

        
    # def forward(self, data, cache):
    #     # 模型预测
    #     t1 = time.time()
    #     with torch.no_grad():
    #         bike_start, bike_end, taxi_start, taxi_end = self.pred_model(data)
    #     t2 = time.time()
    #     if self.t > 2:
    #         self.tt1 += t2 - t1
    #     print("tt:",self.tt1)

    #     bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(-1), bike_end.unsqueeze(-1), \
    #                                                     taxi_start.unsqueeze(-1), taxi_end.unsqueeze(-1)
        

    #     new_predict = torch.cat((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
    #     # cache处理
    #     # updated_cache = cache.clone()
    #     updated_cache = torch.cat((new_predict.unsqueeze(1), cache[:, -3: :, : ,:]), dim=1)
    #     # results = updated_cache[:, :, 0, :, :2]
    #     # interoutput 处理
    #     # print(intermediate_output)
    #     # 移除 hook
    #     # qvalu
    #     # attention_scores = self.attention(tensors)
    #     q = data[0][:, -4:, :, 0].reshape(self.batch_size, 4, -1)
    #     # q = intermediate_output[:, -8:-4:,:]
    #     k = updated_cache[:, :, 0, :, 0].reshape(self.batch_size, 4, -1)
    #     # k = new_timesnet_cache[:, :, 0, :].reshape(self.batch_size, 4, -1)
    #     v = q
        
    #     t3 = time.time()
    #     # _, attn_weights = self.attention(q, k, v)
    #     attn_weights1 = self.attention1(q, k, v)
    #     t4 = time.time()
    #     self.tt2 += t4 - t3
    #     print("tt22222:",self.tt2)
    #     self.t += 1


    #     # attn_weights = self.attention(q, k, v)
    #     # logits = self.decisionT(q, k)
    #     # attn_weights = torch.mean(attn_weights, dim=1)
    #     q = data[2][:, -4:, :, 0].reshape(self.batch_size, 4, -1)
    #     # # q = intermediate_output[:, -8:-4:,:]
    #     k = updated_cache[:, :, 0, :, 2].reshape(self.batch_size, 4, -1)
    #     attn_weights2 = self.attention2(q, k, v)

    #     return attn_weights1, attn_weights2, updated_cache, new_predict
    
def save_model(model, save_path, episode):
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    model_save_path = os.path.join(save_path, f"mmsm_{episode}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == '__main__':
    set_seed(args.seed)

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



        # 初始化模型列表
    model_list = nn.ModuleList()

    # 加载多个不同权重的模型
    for i, model_path in enumerate([
                    '/home_nfs/haitao/data/web_host_network/mdtp_moe/log_2024-12-27_01-43-55/best_model.pth',# 4
                    '/home_nfs/haitao/data/web_host_network/mdtp_moe/log_2024-12-27_01-49-22/best_model.pth',# 8
                    '/home_nfs/haitao/data/web_host_network/mdtp_moe/log_2024-12-23_12-50-02/best_model.pth',# 12
                    '/home_nfs/haitao/data/web_host_network/mdtp_moe/log_2025-01-01_03-05-24/best_model.pth',# 16
                    '/home_nfs/haitao/data/web_host_network/mdtp_moe/log_2025-01-01_03-25-10/best_model.pth',# 20
                    '/home_nfs/haitao/data/web_host_network/mdtp_moe/log_2024-12-22_22-54-12/best_model.pth'# 24
                    ]):
        # 初始化一个新模型
        pred_model = Net_timesnet_sample_onetimesnet(
                args.batch_size, 
                args.window_size, 
                args.node_num, 
                args.in_features, 
                args.out_features, 
                args.lstm_features, 
                base_smoe_config, 
                args.pred_size)
        pred_model.to(device)
        pred_model.load_state_dict(torch.load(model_path))
        for param in pred_model.parameters():
            param.requires_grad = False
        # 将模型添加到列表中
        model_list.append(pred_model)


    # for param in pred_model.parameters():
    #     param.requires_grad = False
    # BanditNet
    bandit_Net = BanditNet(
        num_arms=4, 
        model_list=model_list, 
        batch_size=args.batch_size,
        pred_size=args.pred_size)
    bandit_Net.to(device)
    bandit_Net.train()
    optimizer = optim.Adam(bandit_Net.parameters(), lr=0.0001)
    
    # env
    env = MultiArmedBanditEnv(num_arms=4, batch_size=args.batch_size, pred_size=args.pred_size)

    num_episodes = 400
    epsilon_start = 1.0  # 初始探索率
    epsilon_end = 0.01   # 最低探索率
    epsilon_decay = 0.995  # 衰减因子
    for episode in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        data, cache = env.reset()
        total_loss = 0
        t = 0
        while data != None:
            # 做决策
            # q_values1, q_values2, cache, new_predict= bandit_Net(data, cache)  # Q值估计
            new_predict, st_predict, decisoin_q, newcache = bandit_Net(data, cache, epsilon)  # Q值估计
            # # 自由探索 (ε-greedy 策略)
            # if random.random() < epsilon:  # 以 ε 的概率进行随机探索
            #     decision = torch.randint(0, decisoin_q.size(0), (1,))  # 随机选择动作
            # else:
            #     decision = torch.argmax(decisoin_q, dim=0)  # 选择最大Q值对应的动作

            # # 在环境中执行动作
            # final_pred1 = cache[:,:,0,:,:2] * q_values1.unsqueeze(-1).unsqueeze(-1)
            # final_pred2 = cache[:,:,0,:,2:] * q_values2.unsqueeze(-1).unsqueeze(-1)
            # final_pred = torch.cat((final_pred1, final_pred2), dim=-1)
            # final_pred = torch.sum(final_pred,dim=1)
            data, cache = env.step(newcache=newcache)
            bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics = env.get_acc_train(new_predict)
            st_bike_start_metrics, st_bike_end_metrics, st_taxi_start_metrics, st_taxi_end_metrics = env.get_acc_train(st_predict)
            reward = (st_bike_start_metrics - bike_start_metrics) + (st_bike_end_metrics - bike_end_metrics) + \
                        (st_taxi_start_metrics - taxi_start_metrics) + (st_taxi_end_metrics - taxi_end_metrics)

            

            # if t < 4:
            #     t += 1
            #     continue
            # t += 1

            # 计算当前Q值并生成目标值
            # target1 = q_values1.clone()  # 克隆当前Q值作为目标值
            # target1 = target1.detach()  # 确保目标值不会参与梯度计算
            # target[torch.arange(args.batch_size), action] = reward[torch.arange(args.batch_size), action]  # 仅更新当前动作的目标值
            # target[torch.arange(args.batch_size), action] = reward  # 仅更新当前动作的目标值
            # target1 = reward1
            # target2 = reward2
            # epsilon = 1e-9
            # target = torch.clamp(target, min=epsilon)
            # q_values = torch.clamp(q_values, min=epsilon)

            # 前向传播和优化
            optimizer.zero_grad()

            # target = decisoin_q.clone().detach()
            # target += reward
            # loss = nn.MSELoss()(decisoin_q, target).float()  
            loss = -torch.log(decisoin_q) * reward

            # log_probs = torch.log(q_values)
            # weight = torch.tensor((1.0, 2.0, 3.0, 4.0), device=device)
            # loss = nn.NLLLoss(weight=weight)(log_probs, reward)
            # loss = F.kl_div(q_values.log(), target, reduction='batchmean') 
            # loss = nn.BCEWithLogitsLoss()(q_values, target).float()
            # loss = nn.L1Loss()(q_values, target).float()  # 使用第一次模型运行的q_values
            # loss = nn.MSELoss()(q_values, target).float()  # 使用第一次模型运行的q_values
            # loss = nn.MSELoss(reduction='none')(q_values, target).float()  # 使用第一次模型运行的q_values
            # weight = torch.tensor((1, 2, 3, 4), device=device)
            # loss = (loss * weight).mean()
            # weights = torch.tensor([0.1, 0.2, 0.3, 0.4],device=device)
            # loss = nn.CrossEntropyLoss()(q_values1, target1)
            # loss += nn.CrossEntropyLoss()(q_values2, target2)
            # print(f"Loss1: {loss:.4f}")
            
            # loss = bike_start_metrics + bike_end_metrics + taxi_start_metrics + taxi_end_metrics
            # print(f"Loss1: {bike_start_metrics:.4f}")
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()
            # print("e;t", episode, t)
            # print("q_values", q_values)
            # print("target", target)
        print(f"Loss: {total_loss:.4f}")
        # 打印训练进度
        print(f"Episode {episode + 1}/{num_episodes}")
        save_model(bandit_Net, args.rlsave, episode)