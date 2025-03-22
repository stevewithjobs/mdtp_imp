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
parser.add_argument('--save', type=str, default='/home/haitao/data/web_host_network/mdtp_moe/whnbest/best_model.pth', help='save path')
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
    def __init__(self, num_arms, batch_size, pred_size, data_type='train'):
        """
        初始化多臂老虎机环境。

        :param num_arms: 动作数量（臂的数量）
        :param data_type: 数据类型，'train' 或 'test'
        """
        self.num_arms = num_arms
        self.batch_size = batch_size
        self.pred_size = pred_size

        # 检查 data_type 是否有效
        assert data_type in ['train', 'test'], "data_type 必须为 'train' 或 'test'"

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
        timesnet_cache = self._gettimesnetcache()
        return data, cache, timesnet_cache
    
    def _getreward(self, action, newcache):
        cache_loss0 = self.loss(newcache[:, :, 0, :, 0], self.out_node[:, :, 0].unsqueeze(1).expand(-1, 4, -1))      
        cache_loss1 = self.loss(newcache[:, :, 0, :, 1], self.out_node[:, :, 1].unsqueeze(1).expand(-1, 4, -1))      
        cache_loss2 = self.loss(newcache[:, :, 0, :, 2], self.out_node[:, :, 2].unsqueeze(1).expand(-1, 4, -1))      
        cache_loss3 = self.loss(newcache[:, :, 0, :, 3], self.out_node[:, :, 3].unsqueeze(1).expand(-1, 4, -1))         # # mean = torch.mean(cache_loss, dim=1, keepdim=True)
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
        min_indices0 = torch.argmin(cache_loss0, dim=1).unsqueeze(-1)
        min_indices1 = torch.argmin(cache_loss1, dim=1).unsqueeze(-1)
        min_indices2 = torch.argmin(cache_loss2, dim=1).unsqueeze(-1)
        min_indices3 = torch.argmin(cache_loss3, dim=1).unsqueeze(-1) 
        min_indices = torch.cat((min_indices0, min_indices1, min_indices2, min_indices3), dim=-1)
        return min_indices
        # reward = min_indices == action
        # reward = torch.zeros_like(cache_loss)  # 初始化为 0
        # reward[torch.arange(self.batch_size), min_indices] = 1 
        # 计算newcache中每个的loss，找最小，如果和action相同，reward=1 否则=0
        # return reward.float()
        
        # return min_indices

    
    def _updatecache(self, newcache):
        self.cache = newcache
        self.cache = torch.roll(self.cache, 1, dims=1)
        self.cache = torch.roll(self.cache, -1, dims=2)
        return 

    def step(self, action, newcache, new_timesnet_cache):
        # 获取reward
        reward = self._getreward(action, newcache)
        # 更新cache
        self._updatecache(newcache)
        cache = self._getcache()
        # 更新cache
        self._updatecache(new_timesnet_cache)
        timesnet_cache = self._getcache()
        # 获取data
        data = self._getdata()
        return reward, data, cache, timesnet_cache

    def steptest(self, action, newcache, new_timesnet_cache):
        # 更新cache
        self._updatecache(newcache)
        cache = self._getcache()
        # 更新cache
        self._updatecache(new_timesnet_cache)
        timesnet_cache = self._getcache()
        # 获取data
        data = self._getdata()
        return data, cache, timesnet_cache
    
    def getrightaction(self, action, newcache):
        cache_loss = self.loss(newcache[:, :, 0, :, 0], self.out_node[:, :, 0].unsqueeze(1).expand(-1, 4, -1))      
        rightaction = torch.argmin(cache_loss, dim=1)
        _, sortaction = torch.sort(cache_loss, dim=-1, descending=False)
        return sortaction
    
    def get_acc(self, pred):
        # pred = pred[:, :, 0, :, :].view(231, 4)
        # pred = pred[:, 0, :, :].view(231, 4)
        pred = pred.view(231, 4)
        # pred = pred[0, 0, :, :]
        out_node = self.out_node.view(231, 4)
        bk_start_mask = pred[:, 0]!= out_node[:, 0]
        bk_end_mask = pred[:, 1] != out_node[:, 1]
        tx_start_mask = pred[:, 2] != out_node[:, 2]
        tx_end_mask = pred[:, 3] != out_node[:, 3]
        
        bike_start_metrics = metric1(pred[:, 0], out_node[:, 0], bk_start_mask)
        bike_end_metrics = metric1(pred[:, 1], out_node[:, 1], bk_end_mask)
        taxi_start_metrics = metric1(pred[:, 2], out_node[:, 2], tx_start_mask)
        taxi_end_metrics = metric1(pred[:, 3], out_node[:, 3], tx_end_mask)

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
        
        bike_start_metrics = mdtp.mae(pred[:,:, 0], out_node[:,:, 0], bk_start_mask)
        bike_end_metrics = mdtp.mae(pred[:,:, 1], out_node[:,:, 1], bk_end_mask)
        taxi_start_metrics = mdtp.mae(pred[:,:, 2], out_node[:,:, 2], tx_start_mask)
        taxi_end_metrics = mdtp.mae(pred[:,:, 3], out_node[:,:, 3], tx_end_mask)

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
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 第二层：通道数从 16 -> 4
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.key_proj = nn.Sequential(
            # 第一层：通道数从 4 -> 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 第二层：通道数从 16 -> 4
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.fc = nn.Linear(32,4)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=231, num_heads=11, batch_first=True)

    # def forward(self, q, k, v):
    #     _, attn_weights = self.multihead_attn(q, k, v)
    #     # attn_scores = torch.log(attn_weights) 
    #     weights = torch.mean(attn_weights, dim=1)
    #     return weights

    # def forward(self, q, k, v):
    def forward(self, q, k, v):
        # bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = q
        # bike_node = bike_node_ori[:,-4:,:,:]
        # bike_adj = bike_adj_ori[:,-4:,:,:]
        # taxi_node = taxi_node_ori[:,-4:,:,:]
        # taxi_adj = taxi_adj_ori[:,-4:,:,:]

        # tensors: (B, N, F)，B是batch size，N是Tensor个数，F是特征维度
        # batch_size, _, _ = q.size()
        q = q.reshape(self.batch_size, 16, 11, 21)
        k = k.reshape(self.batch_size, 16, 11, 21)

        # Step 1: 投影 Query, Key, Value
        # gcn_output1, gcn_output2 = self.query_proj(bike_node, bike_adj, taxi_node, taxi_adj, 0)
        queries = self.query_proj(q)  # (B, N, F)
        keys = self.key_proj(k)      # (B, N, F)
        queries = queries.reshape(self.batch_size, 16, -1)
        keys = keys.reshape(self.batch_size, 16, -1)

        # self.multihead_attn(queries, keys, keys)
        
        # Step 2: 计算注意力分数
        # 点乘计算：每个 Query 和 Key 的注意力分数 (B, N, N)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, N, N)
        d_k = queries.size(-1)
        attention_scores /= torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # 缩放
        # scores = torch.mean(attention_scores, dim=1)
        # scores = self.fc(scores)
        # scores = nn.Sigmoid()(scores)
        # # Step 3: 归一化注意力分数


        # 将 (16, 16) 划分为 (4, 4) 的分块矩阵
        block_size = 4
        blocks = attention_scores.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
        # blocks 的形状为 (batchsize, 4, 4, 4, 4)

        # 提取对角线上的四个矩阵
        diagonal_blocks = torch.stack([blocks[:, i, i] for i in range(4)], dim=1)

        attention_weights = nn.Softmax(dim=-1)(diagonal_blocks)  # (B, N, N)
        weights = torch.mean(attention_weights, dim=2)

        return weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class DecisionTransformer(nn.Module):
    def __init__(self, seq_features, pred_features, model_dim, num_heads, num_layers):
        super(DecisionTransformer, self).__init__()
        self.seq_projection = nn.Linear(seq_features, model_dim)  # 将时序特征映射到模型维度
        self.pred_projection = nn.Linear(pred_features, model_dim)  # 将预测特征映射到模型维度
        self.positional_encoding = PositionalEncoding(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_classification = nn.Linear(model_dim, 1)  # 输出 4 个预测结果的分类分数

    def forward(self, seq_input, pred_input):
        """
        Args:
            seq_input: (batch_size, seq_len, seq_features)
            pred_input: (batch_size, pred_len, pred_features)

        Returns:
            logits: (batch_size, pred_len) 分类分数
        """
        # 对输入进行线性变换
        # 时序输入添加位置编码
        seq_emb = self.seq_projection(seq_input)  # (batch_size, seq_len, model_dim)
        seq_emb = self.positional_encoding(seq_emb)  # 添加位置编码

        # 预测输入不添加位置编码
        pred_emb = self.pred_projection(pred_input)  # (batch_size, pred_len, model_dim)

        # 拼接时序和预测输入
        combined_input = torch.cat((seq_emb, pred_emb), dim=1)  # (batch_size, seq_len + pred_len, model_dim)

        # Transformer 编码器
        encoded = self.transformer_encoder(combined_input)  # (batch_size, seq_len + pred_len, model_dim)

        # 分类头，仅取预测部分的编码
        pred_encoded = encoded[:, -pred_input.size(1):, :]  # (batch_size, pred_len, model_dim)
        logits = self.fc_classification(pred_encoded).squeeze(-1) # (batch_size, pred_len)
        logits_aftsig = nn.Softmax()(logits)
        return logits_aftsig  # 每个预测结果的得分

class BanditNet(nn.Module):
    def __init__(self, num_arms, pred_model, batch_size, pred_size):
        super(BanditNet, self).__init__()
        self.batch_size = batch_size
        self.pred_size = pred_size
        self.pred_model = pred_model
        self.fc = nn.Linear(num_arms, num_arms)  # 简单的线性网络
        self.attention = MultiTensorAttention(batch_size)
        # self.attention = nn.MultiheadAttention(embed_dim=231, num_heads=3, dropout=0.1, batch_first=True)
        # self.decisionT = DecisionTransformer(seq_features=231, 
        #                                      pred_features=231, 
        #                                      model_dim=231*8, 
        #                                      num_heads=8, 
        #                                      num_layers=2)
        

    def forward(self, data, cache, timesnet_cache):

        intermediate_output = None  # 用于存储中间结果
        def hook_fn(module, input, output):
            nonlocal intermediate_output
            intermediate_output = output

        # 假设我们需要 layer2 的输出
        hook = self.pred_model.timesnet.projection.register_forward_hook(hook_fn)

        # 模型预测
        bike_start, bike_end, taxi_start, taxi_end = self.pred_model(data)
        bike_start, bike_end, taxi_start, taxi_end = bike_start.unsqueeze(-1), bike_end.unsqueeze(-1), \
                                                        taxi_start.unsqueeze(-1), taxi_end.unsqueeze(-1)
        new_predict = torch.cat((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
        # cache处理
        # updated_cache = cache.clone()
        updated_cache = torch.cat((new_predict.unsqueeze(1), cache[:, -3: :, : ,:]), dim=1)
        results = updated_cache[:, :, 0, :, :]
        # results = updated_cache[:, :, 0, :, :2]
        # interoutput 处理
        # print(intermediate_output)
        new_timesnet_cache = torch.cat((intermediate_output[:, -4:, :].unsqueeze(1), timesnet_cache[:, -3:, :, :]), dim=1)
        # 移除 hook
        hook.remove()
        # qvalu
        # attention_scores = self.attention(tensors)
        q1 = data[0][:, -4:, :, :]
        q2 = data[2][:, -4:, :, :]
        q = torch.cat((q1, q2), dim=-1)
        q = q.permute(0,1,3,2).reshape(self.batch_size, 16, 231)
        # q = intermediate_output[:, -8:-4:,:]
        k = results.permute(0,1,3,2).reshape(self.batch_size, 16, 231)
        # k = new_timesnet_cache[:, :, 0, :].reshape(self.batch_size, 4, -1)
        v = q
        # _, attn_weights = self.attention(q, k, v)
        attn_weights = self.attention(q, k, v)
        # attn_weights = self.attention(q, k, v)
        # logits = self.decisionT(q, k)
        # attn_weights = torch.mean(attn_weights, dim=1)

        return attn_weights, updated_cache, new_predict, new_timesnet_cache
    
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
    pred_model.load_state_dict(torch.load(args.save))
    for param in pred_model.parameters():
        param.requires_grad = False
    # BanditNet
    bandit_Net = BanditNet(
        num_arms=4, 
        pred_model=pred_model, 
        batch_size=args.batch_size,
        pred_size=args.pred_size)
    bandit_Net.to(device)
    bandit_Net.train()
    optimizer = optim.Adam(bandit_Net.parameters(), lr=0.00001)
    
    # env
    env = MultiArmedBanditEnv(num_arms=4, batch_size=args.batch_size, pred_size=args.pred_size)

    num_episodes = 400
    epsilon_start = 1.0  # 初始探索率
    epsilon_end = 0.01   # 最低探索率
    epsilon_decay = 0.995  # 衰减因子
    for episode in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        data, cache, timesnet_cache = env.reset()
        total_loss = 0
        t = 0
        while data != None:
            # 做决策
            q_values, cache, new_predict, timesnet_cache= bandit_Net(data, cache ,timesnet_cache)  # Q值估计
            # 自由探索 (ε-greedy 策略)
            if random.random() < epsilon:  # 以 ε 的概率进行随机探索
                action = torch.randint(0, q_values.size(1), (q_values.size(0),))  # 随机选择动作
            else:
                action = torch.argmax(q_values, dim=1)  # 选择最大Q值对应的动作

            # 在环境中执行动作
            q_values = q_values.permute(0,2,1)
            final_pred = cache[:,:,0,:,:] * q_values.unsqueeze(-2)
            final_pred = torch.sum(final_pred,dim=1)
            reward, data, cache, timesnet_cache = env.step(action=action, newcache=cache, new_timesnet_cache=timesnet_cache)
            bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics = env.get_acc_train(final_pred)
            if t < 4:
                t += 1
                continue
            t += 1

            # 计算当前Q值并生成目标值
            target = q_values.clone()  # 克隆当前Q值作为目标值
            target = target.detach()  # 确保目标值不会参与梯度计算
            # target[torch.arange(args.batch_size), action] = reward[torch.arange(args.batch_size), action]  # 仅更新当前动作的目标值
            # target[torch.arange(args.batch_size), action] = reward  # 仅更新当前动作的目标值
            target = reward
            # epsilon = 1e-9
            # target = torch.clamp(target, min=epsilon)
            # q_values = torch.clamp(q_values, min=epsilon)

            # 前向传播和优化
            # optimizer.zero_grad()
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
            loss = 0
            for i in range(4):
                loss += nn.CrossEntropyLoss()(q_values[:,i,:], target[:,i])
            # print(f"Loss1: {loss:.4f}")
            loss += (bike_start_metrics + bike_end_metrics + taxi_start_metrics + taxi_end_metrics) * 100
            print(f"Loss1: {bike_start_metrics:.4f}")
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss
            print("e;t", episode, t)
            print("q_values", q_values)
            print("target", target)
        print(f"Loss: {total_loss:.4f}")
        # 打印训练进度
        print(f"Episode {episode + 1}/{num_episodes}")
        save_model(bandit_Net, args.rlsave, episode)