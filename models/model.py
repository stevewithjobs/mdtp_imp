import torch
from torch import nn
from models.smoe import GatedSpatialMoE2d, GatedSpatialMoE2d_s

from models.expert_s import Generator, ReasoningNet, ReasoningNetImp, NodeSampler
from models.expert_t import LstmAttention
from models.TimesNet import TimesBlock, Model, Model_moe, Model_withoutmoe, Model_moeconv, Model_moenew, Model_onetimenet, Model_3D
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
from collections import deque

import functools
import time



class Net(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.generator = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )
        self.rnn1 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2,
        )
        self.rnn2 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2
        )

        # # concat net
        # self.ffn2 = nn.Sequential(
        #     nn.Linear(node_num * 2, node_num * 4),
        #     nn.ReLU()
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(node_num * 2, node_num),
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(node_num * 2, node_num),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(node_num * 2, node_num),
        #     nn.ReLU()
        # )
        # self.fc4 = nn.Sequential(
        #     nn.Linear(node_num * 2, node_num),
        #     nn.ReLU()
        # )
        # sum net
        self.ffn1 = nn.Sequential(
            nn.Linear(node_num, node_num * 4),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        bike_node, bike_adj, taxi_node, taxi_adj = x[0], x[1], x[2], x[3],
        gcn_output1, gcn_output2 = self.generator(bike_node, bike_adj, taxi_node, taxi_adj)
        lstm_output1 = self.rnn1(gcn_output1)
        lstm_output2 = self.rnn2(gcn_output2)
        # sum
        sum_output = self.ffn1(lstm_output1 + lstm_output2).reshape(self.batch_size, self.node_num, -1)
        bike_start = self.fc1(lstm_output1 + sum_output[:, :, 0])
        bike_end = self.fc2(lstm_output1 + sum_output[:, :, 1])
        taxi_start = self.fc3(lstm_output2 + sum_output[:, :, 2])
        taxi_end = self.fc4(lstm_output2 + sum_output[:, :, 3])
        # cancat
        # concat_output = self.ffn2(torch.cat((lstm_output1, lstm_output2), -1)).reshape(self.batch_size, self.node_num, -1)
        # bike_start = self.fc1(torch.cat((lstm_output1, concat_output[:, :, 0]), -1))
        # bike_end = self.fc2(torch.cat((lstm_output1, concat_output[:, :, 1]), -1))
        # taxi_start = self.fc3(torch.cat((lstm_output2, concat_output[:, :, 2]), -1))
        # taxi_end = self.fc4(torch.cat((lstm_output2, concat_output[:, :, 3]), -1))

        return bike_start, bike_end, taxi_start, taxi_end

class NetNothing(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features):
        super(NetNothing, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.generator = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )
        self.rnn1 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2,
        )
        self.rnn2 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2
        )
        # sum net
        self.ffn1 = nn.Sequential(
            nn.Linear(node_num, node_num * 4),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, input):
        bike_node, bike_adj, taxi_node, taxi_adj = input[0], input[1], input[2], input[3],
        gcn_output1, gcn_output2 = self.generator(bike_node, bike_adj, taxi_node, taxi_adj)
        lstm_output1 = self.rnn1(gcn_output1)
        lstm_output2 = self.rnn2(gcn_output2)
        # nobridge
        bike_start = self.fc1(lstm_output1)
        bike_end = self.fc2(lstm_output1)
        taxi_start = self.fc3(lstm_output2)
        taxi_end = self.fc4(lstm_output2)

        return bike_start, bike_end, taxi_start, taxi_end

class NetImp(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features):
        super(NetImp, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.generator = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )
        self.rnn1 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2,
        )
        self.rnn2 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2
        )
        self.rnn3 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2
        )
        self.rn = ReasoningNet(
            window_size=window_size,
            out_features=out_features,
            node_num=node_num
        )
        # self.rn = Reasoning_net_imp(
        #     window_size=window_size,
        #     out_features=out_features,
        #     node_num=node_num
        # )
        self.ffn1 = nn.Sequential(
            nn.Linear(node_num, node_num * 2),
            nn.ReLU()
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(node_num, node_num * 2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        bike_node, bike_adj, taxi_node, taxi_adj = x[0], x[1], x[2], x[3],
        gcn_output1, gcn_output2 = self.generator(bike_node, bike_adj, taxi_node, taxi_adj)
        rn_output, weight1, weight2 = self.rn(gcn_output1, gcn_output2)

        lstm_output1 = self.rnn1(gcn_output1)
        lstm_output2 = self.rnn2(gcn_output2)
        lstm_output3 = self.rnn3(rn_output)

        bike_output = self.ffn1(lstm_output1 + lstm_output3).reshape(self.batch_size, self.node_num, -1)
        taxi_output = self.ffn2(lstm_output2 + lstm_output3).reshape(self.batch_size, self.node_num, -1)

        bike_start = self.fc1(lstm_output1 + bike_output[:, :, 0])
        bike_end = self.fc2(lstm_output1 + bike_output[:, :, 1])
        taxi_start = self.fc3(lstm_output2 + taxi_output[:, :, 0])
        taxi_end = self.fc4(lstm_output2 + taxi_output[:, :, 1])

        return bike_start, bike_end, taxi_start, taxi_end, weight1, weight2

class NetMoe(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config):
        super(NetMoe, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size
        self.generator1 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )
        self.generator2 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )
        self.rnn1 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2,
        )
        self.rnn2 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2
        )
        self.rnn3 = LstmAttention(
            node_size=node_num,
            input_size=node_num * out_features,
            hidden_dim=lstm_features,
            n_layers=2
        )
        self.rn = ReasoningNetImp(
            window_size=window_size,
            out_features=out_features,
            node_num=node_num
        )
        self.smoe1 = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
        self.smoe2 = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(node_num, node_num * 2),
            nn.ReLU()
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(node_num, node_num * 2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        ''' expert'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]

        a = 0.75
        bike_node_mix = bike_node_ori * a + taxi_node_ori * (1 - a)
        taxi_node_mix = bike_node_ori * (1 - a) + taxi_node_ori * a
        bike_adj_mix = bike_adj_ori * a + taxi_adj_ori * (1 - a)
        taxi_adj_mix = bike_adj_ori * (1 - a) + taxi_adj_ori * a

        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)
        gcn_output3, gcn_output4 = self.generator2(bike_node_mix, bike_adj_mix, taxi_node_mix, taxi_adj_mix)
        rn_output, weight1, weight2 = self.rn(gcn_output1, gcn_output2)

        '''smoe:'''
        expert1 = gcn_output1.view(self.batch_size, self.window_size, self.node_num, -1)
        expert1 = expert1.unsqueeze(2)
        expert2 = gcn_output2.view(self.batch_size, self.window_size, self.node_num, -1)
        expert2 = expert2.unsqueeze(2)
        expert3 = gcn_output3.view(self.batch_size, self.window_size, self.node_num, -1)
        expert3 = expert3.unsqueeze(2)
        expert4 = gcn_output4.view(self.batch_size, self.window_size, self.node_num, -1)
        expert4 = expert4.unsqueeze(2)
        expert5 = rn_output.view(self.batch_size, self.window_size, self.node_num, -1)
        expert5 = expert5.unsqueeze(2)

        experts = torch.cat((expert1, expert2, expert3, expert4, expert5), dim=2)

        smoe1 = self.smoe1(x=x[0], experts=experts)
        smoe2 = self.smoe2(x=x[2], experts=experts)

        smoe1 = smoe1.sum(dim=2, keepdim=True)
        smoe1 = smoe1.squeeze(2)
        smoe1 = smoe1.view(self.batch_size, self.window_size, -1)
        smoe2 = smoe2.sum(dim=2, keepdim=True)
        smoe2 = smoe2.squeeze(2)
        smoe2 = smoe2.view(self.batch_size, self.window_size, -1)

        lstm_output1 = self.rnn1(smoe1)
        lstm_output2 = self.rnn2(smoe2)

        bike_output = self.ffn1(lstm_output1 ).reshape(self.batch_size, self.node_num, -1)
        taxi_output = self.ffn2(lstm_output2).reshape(self.batch_size, self.node_num, -1)

        bike_start = self.fc1(bike_output[:, :, 0])
        bike_end = self.fc2(bike_output[:, :, 1])
        taxi_start = self.fc3(taxi_output[:, :, 0])
        taxi_end = self.fc4(taxi_output[:, :, 1])
        
        weight_loss = 1 - weight1 + 1 - weight2
        weight_loss = torch.mean(weight_loss) 
        # weight_loss = 0
        return bike_start, bike_end, taxi_start, taxi_end, weight_loss

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

class Net_timesnet(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config):
        super(Net_timesnet, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size
        self.generator1 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )

        self.smoe1 = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
        self.smoe2 = GatedSpatialMoE2d(
            smoe_config=smoe_config
        )
     
        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=1, 
            top_k=2, 
            d_model=node_num * out_features, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 2,
        )

        self.timesnet = Model_withoutmoe(
             timesnetconfig,
        )
        # self.timesnet = Model_withoutmoe(
        #      timesnetconfig,
        #      smoe_config
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        ''' spatio'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)

        '''temporal'''   
        timesnetout1, timesnetout2 = self.timesnet(x[0], x[2], gcn_output1, gcn_output2)

        bike_output = timesnetout1.reshape(self.batch_size, self.node_num, -1)
        taxi_output = timesnetout2.reshape(self.batch_size, self.node_num, -1)

        bike_start = self.fc1(bike_output[:, :, 0])
        bike_end = self.fc2(bike_output[:, :, 1])
        taxi_start = self.fc3(taxi_output[:, :, 0])
        taxi_end = self.fc4(taxi_output[:, :, 1])
    
        return bike_start, bike_end, taxi_start, taxi_end
    


class Net_timesnet_moe(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config):
        super(Net_timesnet_moe, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size

        self.generator1 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )
        self.generator1 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )


        smoe_config_s = SpatialMoEConfig(
            in_planes=2,
            out_planes=32,
            num_experts=64,
            gate_block=functools.partial(SpatialLatentTensorGate2d,
                                node_num = 231),
            save_error_signal=True,
            dampen_expert_error=False,
            # dampen_expert_error=True,
            unweighted=False,
            block_gate_grad=True,
            routing_error_quantile=0.7,
            pred_size=0
        )
        self.smoe1 = GatedSpatialMoE2d(
            smoe_config=smoe_config_s
        )
        self.smoe2 = GatedSpatialMoE2d(
            smoe_config=smoe_config_s
        )

        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=1, 
            top_k=3, 
            d_model=node_num * out_features, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 2,
            batch_size=batch_size,
        )
    
        self.timesnet = Model_moe(
             timesnetconfig,
             smoe_config,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        '''online判断'''
        
        ''' spatio'''
        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)
        
        '''temporal'''
        timesnetout1, timesnetout2 = self.timesnet(x[0], x[2], gcn_output1, gcn_output2)

        bike_output = timesnetout1.reshape(self.batch_size, self.node_num, -1)
        taxi_output = timesnetout2.reshape(self.batch_size, self.node_num, -1)

        bike_start = self.fc1(bike_output[:, :, 0])
        bike_end = self.fc2(bike_output[:, :, 1])
        taxi_start = self.fc3(taxi_output[:, :, 0])
        taxi_end = self.fc4(taxi_output[:, :, 1])
        
        return bike_start, bike_end, taxi_start, taxi_end
    
class Net_timesnet_moe_nstp(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
        super(Net_timesnet_moe_nstp, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size

        self.generator = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )

        self.pred_size = pred_size
        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=self.pred_size, 
            top_k=1, 
            d_model=node_num * out_features, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 2,
            batch_size=batch_size
        )
    
        self.timesnet = Model_moe(
             timesnetconfig,
             smoe_config,
        )
        # self.timesnet = Model_moenew(
        #      timesnetconfig,
        #      smoe_config,
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        ''' spatio'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        gcn_output1, gcn_output2 = self.generator(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)

        # _, _, weight2 = self.rn(gcn_output1, gcn_output2)
        '''temporal'''
        # gcn_output1 = bike_node_ori.reshape(self.batch_size, self.window_size, -1)
        # gcn_output2 = taxi_node_ori.reshape(self.batch_size, self.window_size, -1)

        timesnetout1, timesnetout2 = self.timesnet(x[0], x[2], gcn_output1, gcn_output2)
        timesnetout1, timesnetout2 = self.timesnet(x[0], x[2], gcn_output1, gcn_output2)


        bike_output = timesnetout1.reshape(self.batch_size, self.pred_size, self.node_num, -1)
        taxi_output = timesnetout2.reshape(self.batch_size, self.pred_size, self.node_num, -1)

        bike_start = self.fc1(bike_output[:, :, :, 0])
        bike_end = self.fc2(bike_output[:, :, :, 1])
        taxi_start = self.fc3(taxi_output[:, :, :, 0])
        taxi_end = self.fc4(taxi_output[:, :, :, 1])
        
        return bike_start, bike_end, taxi_start, taxi_end

class Net_timesnet_nstp(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config):
        super(Net_timesnet_nstp, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size
        self.generator1 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=in_features,
            out_features=out_features
        )
        # self.generator2 = Generator(
        #     window_size=window_size,
        #     node_num=node_num,
        #     in_features=in_features,
        #     out_features=out_features
        # )
        # timesnetconfig = Config(
        #     seq_len=window_size, 
        #     pred_len=1, 
        #     top_k=3, 
        #     d_model=node_num*out_features, 
        #     d_ff=node_num*out_features, 
        #     num_kernels=3, 
        #     e_layers=1, 
        #     c_out=node_num*out_features,
        # )
        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=4, 
            top_k=3, 
            d_model=node_num * out_features, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 2,
        )

        self.timesnet = Model_withoutmoe(
             timesnetconfig,
        )
        # self.timesnet2 = Model_withoutmoe(
        #      timesnetconfig,
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        ''' spatio'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)
        # gcn_output3, gcn_output4 = self.generator2(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)


        '''temporal'''   
        timesnetout1, timesnetout2 = self.timesnet(x[0], x[2], gcn_output1, gcn_output2)
        # timesnetout3, timesnetout4 = self.timesnet2(x[0], x[2], gcn_output3, gcn_output4)


        # bike_output = self.ffn1(timesnetout1).reshape(self.batch_size, self.node_num, -1)
        # taxi_output = self.ffn2(timesnetout2).reshape(self.batch_size, self.node_num, -1)

        # bike_output += timesnetout4.reshape(self.batch_size, self.node_num, -1)
        # taxi_output += timesnetout3.reshape(self.batch_size, self.node_num, -1)

        bike_output = timesnetout1.reshape(self.batch_size, 4, self.node_num, -1)
        taxi_output = timesnetout2.reshape(self.batch_size, 4, self.node_num, -1)

        bike_start = self.fc1(bike_output[:, :, :, 0])
        bike_end = self.fc2(bike_output[:, :, :, 1])
        taxi_start = self.fc3(taxi_output[:, :, :, 0])
        taxi_end = self.fc4(taxi_output[:, :, :, 1])
    
        return bike_start, bike_end, taxi_start, taxi_end

class Net_timesnet_smoe_timesnet(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
        super(Net_timesnet_smoe_timesnet, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size

        self.generator1 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=2,
            out_features=16
        )
        self.conv1 = nn.Conv1d(
            in_channels=node_num * 16, 
            out_channels=node_num * 2, 
            kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=node_num * 16, 
            out_channels=node_num * 2, 
            kernel_size=1)

        # self.generator2 = Generator(
        #     window_size=window_size,
        #     node_num=node_num,
        #     in_features=2,
        #     out_features=16
        # )

        # self.smoe1 = GatedSpatialMoE2d_s(
        #     smoe_config=smoe_config
        # )
        # self.smoe2 = GatedSpatialMoE2d_s(
        #     smoe_config=smoe_config
        # )

        self.pred_size = pred_size
        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=self.pred_size, 
            top_k=1, 
            d_model=node_num * 16, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 2,
            batch_size=batch_size
        )
    
        self.timesnet = Model_withoutmoe(
             timesnetconfig,
            #  smoe_config,
        )
        # self.timesnet = Model_moenew(
        #      timesnetconfig,
        #      smoe_config,
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        ''' spatio'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]

        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)
        #  压缩
        # gcn_output1, gcn_output2 = gcn_output1.permute(0, 2, 1), gcn_output2.permute(0, 2, 1)
        # gcn_output1 = self.conv1(gcn_output1)
        # gcn_output2 = self.conv2(gcn_output2)
        # gcn_output1, gcn_output2 = gcn_output1.permute(0, 2, 1), gcn_output2.permute(0, 2, 1)
        '''temporal'''
        timesnetout1, timesnetout2 = self.timesnet(x[0], x[2], gcn_output1, gcn_output2)

        bike_output = timesnetout1.view(self.batch_size, self.pred_size, self.node_num, -1)
        taxi_output = timesnetout2.view(self.batch_size, self.pred_size, self.node_num, -1)

        bike_start = self.fc1(bike_output[:, :, :, 0])
        bike_end = self.fc2(bike_output[:, :, :, 1])
        taxi_start = self.fc3(taxi_output[:, :, :, 0])
        taxi_end = self.fc4(taxi_output[:, :, :, 1])

        return bike_start, bike_end, taxi_start, taxi_end
    

class Net_timesnet_onetimesnet(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
        super(Net_timesnet_onetimesnet, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size

        self.generator1 = Generator(
            window_size=window_size,
            node_num=node_num,
            in_features=2,
            out_features=8
        )

        # self.smoe1 = GatedSpatialMoE2d_s(
        #     smoe_config=smoe_config
        # )
        # self.smoe2 = GatedSpatialMoE2d_s(
        #     smoe_config=smoe_config
        # )

        self.pred_size = pred_size
        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=self.pred_size, 
            top_k=1, 
            d_model=node_num * 8, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 2,
            batch_size=batch_size
        )
    
        self.timesnet1 = Model_onetimenet(
             timesnetconfig,
            #  smoe_config,
        )

        self.timesnet2 = Model_onetimenet(
             timesnetconfig,
            #  smoe_config,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )

    def forward(self, x):
        t1 = time.time()

        ''' spatio'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]

        '''sample'''
        # bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = self.nodesample(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)
    
        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori)
        

        '''temporal'''
        # timesnetout1= self.timesnet1(bike_node_ori, gcn_output1)
        # timesnetout2= self.timesnet2(taxi_node_ori, gcn_output2)     
        

        gcn_output = torch.cat((gcn_output1, gcn_output2), dim=0)
        # gcn_output = gcn_output1
        timesnetout, wanttime = self.timesnet1(bike_node_ori, gcn_output)

        timesnetout = timesnetout.view(self.batch_size * 2, self.pred_size, self.node_num, -1)
        bike_output, taxi_output = torch.chunk(timesnetout, 2, dim=0)


        # timesnetout = timesnetout.view(self.batch_size, self.pred_size, self.node_num, -1)
        # bike_output = timesnetout
        # taxi_output = timesnetout

        bike_start = self.fc1(bike_output[:, :, :, 0])
        bike_end = self.fc2(bike_output[:, :, :, 1])

        taxi_start = self.fc3(taxi_output[:, :, :, 0])
        taxi_end = self.fc4(taxi_output[:, :, :, 1])
        t2 = time.time()
        

        return bike_start, bike_end, taxi_start, taxi_end, (t2-t1)
    

class Net_timesnet_sample_onetimesnet(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
        super(Net_timesnet_sample_onetimesnet, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size

        sampling_ratio = 2 /3
        sampled_node_num = int(node_num * sampling_ratio)
        self.nodesample = NodeSampler(
            node_num = node_num, 
            sample_node_num=sampled_node_num, 
            window_size=window_size
        )

        self.generator1 = Generator(
            batch_size=batch_size,
            window_size=window_size,
            node_num=node_num,
            in_features=2,
            out_features=8
        )

        # self.generator2 = Generator(
        #     window_size=window_size,
        #     node_num=node_num,
        #     in_features=4,
        #     out_features=8
        # )

        # self.mlp = nn.Sequential(
        #     nn.Linear(21 * 8, 21 * 16),
        #     nn.ReLU(),
        #     nn.Linear(21 * 16, 14 * 8)
        # )

        self.pred_size = pred_size
        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=self.pred_size, 
            top_k=1, 
            d_model=node_num * 16, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 4,
            batch_size=batch_size
        )
    
        self.timesnet = Model_onetimenet(
             timesnetconfig,
            #  smoe_config,
        )


        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.gcntime = 0
        self.tstime = 0
        self.t = 0

    
    def forward(self, x, window_size):
        
        ''' spatio'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        '''sample'''
        
        connect, mask1, mask2 = self.nodesample()
        
        # window_size = 24
        # bike_node_ori = bike_node_ori[:, -window_size, :, :]
        # bike_adj_ori = bike_adj_ori[:, -window_size, :, :]
        # taxi_node_ori = taxi_node_ori[:, -window_size, :, :]
        # taxi_adj_ori = taxi_adj_ori[:, -window_size, :, :]

        
        # t11 = time.time()
        
        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori, 0)
        
        
        # t22 = time.time()
        # if self.t > 2 :
        #     self.gcntime += t22 - t11
        # print("gcn_time_total", self.gcntime)
        # self.gcn_output1 = gcn_output1
        # self.gcn_output2 = gcn_output2
        # self.gcn_output1.retain_grad()
        # self.gcn_output2.retain_grad()
        

        # mask1 = mask1.view(1, 1, self.node_num, 1)
        # mask2 = mask2.view(1, 1, self.node_num, 1)
        # gcn_output1 = gcn_output1 * mask1
        # gcn_output2 = gcn_output2 * mask2

        # bike_node_mask = gcn_output1
        # bike_node_mask[:, :, mask1, :] = 0
        # bike_adj_mask = bike_adj_ori
        # bike_adj_mask[:, :, mask1, :] = 0
        # bike_adj_mask[:, :, :, mask1] = 0

        # taxi_node_mask = gcn_output2
        # taxi_node_mask[:, :, mask2, :] = 0
        # taxi_adj_mask = taxi_adj_ori
        # taxi_adj_mask[:, :, mask2, :] = 0
        # taxi_adj_mask[:, :, :, mask2] = 0
        
        # gcn_output1, gcn_output2 = self.generator2(gcn_output1, bike_adj_ori, gcn_output2, taxi_adj_ori, 0)

        # gcn_output1, gcn_output2 = self.generator2(bike_node_mask, bike_adj_mask, taxi_node_mask, taxi_adj_mask, 0)

        # mask1 = mask1.view(1, self.window_size, self.node_num, 1)
        # mask2 = mask2.view(1, self.window_size, self.node_num, 1)
        
        gcn_output1[:, :, mask1, :] = 0
        gcn_output2[:, :, mask2, :] = 0

        # gcn_conncet = (gcn_output1 + gcn_output2)/2

        # gcn_output1[:, :, connect, :] = gcn_conncet[:, :, connect, :]
        # gcn_output2[:, :, connect, :] = gcn_conncet[:, :, connect, :]

        gcn_output = torch.cat((gcn_output1, gcn_output2), dim=-1)
        
        '''temporal'''     
        
        # gcn_output = torch.cat((gcn_output1, gcn_output2), dim=0).view(self.batch_size * 2, self.window_size, -1)
        gcn_output = gcn_output.view(self.batch_size, self.window_size, -1)

        # import pandas as pd
        # # 将 Tensor 转换为 NumPy 数组
        # tensor_np = torch.mean(gcn_output, dim=[0,-1])
        # tensor_np = tensor_np.cpu().numpy()
        # # 转换为 Pandas DataFrame
        # df = pd.DataFrame({'Values': tensor_np})

        # # 保存为 Excel 文件
        # excel_file = "tensor_data.xlsx"
        # df.to_excel(excel_file, index=False)
        t1 = time.time()
        # window_size = window_size.item()
        gcn_output = gcn_output[:, -window_size:, :]
        t2 = time.time()
        if self.t > 2 :
            self.gcntime += t2 - t1
        print("slice", self.gcntime)

        t1 = time.time()

        timesnetout, _ = self.timesnet(bike_node_ori, gcn_output)
        
        
        # timesnetout2, _ = self.timesnet2(bike_node_ori, gcn_output)
        timesnetout = timesnetout.view(self.batch_size, self.pred_size, self.node_num, -1)

        bike_start = self.fc1(timesnetout[:, :, :, 0])
        bike_end = self.fc2(timesnetout[:, :, :, 1])

        taxi_start = self.fc3(timesnetout[:, :, :, 2])
        taxi_end = self.fc4(timesnetout[:, :, :, 3])

        t2 = time.time()
        if self.t > 1 :
            self.tstime += t2 - t1
        # print("tstime", self.tstime)
        self.t += 1
        return bike_start, bike_end, taxi_start, taxi_end


# class Net_timesnet_sample_onetimesnet(nn.Module):
#     def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
#         super(Net_timesnet_sample_onetimesnet, self).__init__()
#         self.batch_size = batch_size
#         self.node_num = node_num
#         self.window_size = window_size

#         sampling_ratio = 2 /3
#         sampled_node_num = int(node_num * sampling_ratio)
#         self.nodesample = NodeSampler(
#             node_num = node_num, 
#             sample_node_num=sampled_node_num, 
#             window_size=window_size
#         )

#         self.generator1 = Generator(
#             batch_size=batch_size,
#             window_size=window_size,
#             node_num=node_num,
#             in_features=2,
#             out_features=8
#         )

#         # self.generator2 = Generator(
#         #     window_size=window_size,
#         #     node_num=node_num,
#         #     in_features=4,
#         #     out_features=8
#         # )

#         # self.mlp = nn.Sequential(
#         #     nn.Linear(21 * 8, 21 * 16),
#         #     nn.ReLU(),
#         #     nn.Linear(21 * 16, 14 * 8)
#         # )

#         self.pred_size = pred_size
#         timesnetconfig = Config(
#             seq_len=window_size, 
#             pred_len=self.pred_size, 
#             top_k=1, 
#             d_model=16, 
#             d_ff=32,  
#             num_kernels=2, 
#             e_layers=1, 
#             c_out=node_num * 4,
#             batch_size=batch_size
#         )
    
#         self.timesnet = Model_3D(
#              timesnetconfig,
#             #  smoe_config,
#         )


#         self.fc1 = nn.Sequential(
#             nn.Linear(node_num, node_num),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(node_num, node_num),
#             nn.ReLU()
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(node_num, node_num),
#             nn.ReLU()
#         )
#         self.fc4 = nn.Sequential(
#             nn.Linear(node_num, node_num),
#             nn.ReLU()
#         )
#         self.gcntime = 0
#         self.tstime = 0
#         self.t = 0

    
#     def forward(self, x):
        
#         t1 = time.time()
#         ''' spatio'''
#         bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
#         '''sample'''
        
#         connect, mask1, mask2 = self.nodesample()
        
#         # window_size = 24
#         # bike_node_ori = bike_node_ori[:, -window_size, :, :]
#         # bike_adj_ori = bike_adj_ori[:, -window_size, :, :]
#         # taxi_node_ori = taxi_node_ori[:, -window_size, :, :]
#         # taxi_adj_ori = taxi_adj_ori[:, -window_size, :, :]

        
#         # t11 = time.time()
        
#         gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori, 0)
#         t2 = time.time()
#         if self.t > 2 :
#             self.gcntime += t2 - t1
#         print("gcntime", self.gcntime)
        
#         # t22 = time.time()
#         # if self.t > 2 :
#         #     self.gcntime += t22 - t11
#         # print("gcn_time_total", self.gcntime)
#         # self.gcn_output1 = gcn_output1
#         # self.gcn_output2 = gcn_output2
#         # self.gcn_output1.retain_grad()
#         # self.gcn_output2.retain_grad()
        

#         # mask1 = mask1.view(1, 1, self.node_num, 1)
#         # mask2 = mask2.view(1, 1, self.node_num, 1)
#         # gcn_output1 = gcn_output1 * mask1
#         # gcn_output2 = gcn_output2 * mask2

#         # bike_node_mask = gcn_output1
#         # bike_node_mask[:, :, mask1, :] = 0
#         # bike_adj_mask = bike_adj_ori
#         # bike_adj_mask[:, :, mask1, :] = 0
#         # bike_adj_mask[:, :, :, mask1] = 0

#         # taxi_node_mask = gcn_output2
#         # taxi_node_mask[:, :, mask2, :] = 0
#         # taxi_adj_mask = taxi_adj_ori
#         # taxi_adj_mask[:, :, mask2, :] = 0
#         # taxi_adj_mask[:, :, :, mask2] = 0
        
#         # gcn_output1, gcn_output2 = self.generator2(gcn_output1, bike_adj_ori, gcn_output2, taxi_adj_ori, 0)

#         # gcn_output1, gcn_output2 = self.generator2(bike_node_mask, bike_adj_mask, taxi_node_mask, taxi_adj_mask, 0)

#         # mask1 = mask1.view(1, self.window_size, self.node_num, 1)
#         # mask2 = mask2.view(1, self.window_size, self.node_num, 1)
        
#         gcn_output1[:, :, mask1, :] = 0
#         gcn_output2[:, :, mask2, :] = 0

#         # gcn_conncet = (gcn_output1 + gcn_output2)/2

#         # gcn_output1[:, :, connect, :] = gcn_conncet[:, :, connect, :]
#         # gcn_output2[:, :, connect, :] = gcn_conncet[:, :, connect, :]

#         gcn_output = torch.cat((gcn_output1, gcn_output2), dim=-1)
        
#         '''temporal'''     
        
#         # gcn_output = torch.cat((gcn_output1, gcn_output2), dim=0).view(self.batch_size * 2, self.window_size, -1)
#         gcn_output = gcn_output.view(self.batch_size, self.window_size, -1)

#         # import pandas as pd
#         # # 将 Tensor 转换为 NumPy 数组
#         # tensor_np = torch.mean(gcn_output, dim=[0,-1])
#         # tensor_np = tensor_np.cpu().numpy()
#         # # 转换为 Pandas DataFrame
#         # df = pd.DataFrame({'Values': tensor_np})

#         # # 保存为 Excel 文件
#         # excel_file = "tensor_data.xlsx"
#         # df.to_excel(excel_file, index=False)
#         gcn_output = gcn_output[:, -24:, :]
#         t1 = time.time()

#         timesnetout, _ = self.timesnet(bike_node_ori, gcn_output)
        
#         t2 = time.time()
#         if self.t > 2 :
#             self.tstime += t2 - t1
#         print("tstime", self.tstime)
#         # timesnetout2, _ = self.timesnet2(bike_node_ori, gcn_output)
#         timesnetout = timesnetout.view(self.batch_size, self.pred_size, self.node_num, -1)

#         bike_start = self.fc1(timesnetout[:, :, :, 0])
#         bike_end = self.fc2(timesnetout[:, :, :, 1])

#         taxi_start = self.fc3(timesnetout[:, :, :, 2])
#         taxi_end = self.fc4(timesnetout[:, :, :, 3])

    
#         self.t += 1
#         return bike_start, bike_end, taxi_start, taxi_end
    