import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
from collections import deque
from utils.mdtp import MyDataset_rl, set_seed
from torch.utils.data.dataloader import DataLoader
from models.model import Net_timesnet_onetimesnet, Net_timesnet_sample_onetimesnet
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
import functools
from utils import mdtp
# from models.expert_s import Generator_rl
from train_rl_timesnet import TrafficEnv, DQNAgent, custom_collate_fn
import time
import yaml



parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:4', help='GPU setting')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
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
parser.add_argument('--rlsave', type=str, default='./mmsm_model/mmsm_30.pth', help='save path')
parser.add_argument('--smoe_start_epoch', type=int, default=99, help='smoe start epoch')
parser.add_argument('--gpus', type=str, default='4', help='gpu')
parser.add_argument('--log', type=str, default='0.log', help='log name')


args = parser.parse_args()
config = yaml.safe_load(open('config.yml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_dqn(agent, env):

    

    state = env.reset()
    modeltime = 0
    done = False
    t = 0

    bike_start_loss = []
    bike_end_loss = []
    bike_start_rmse = []
    bike_end_rmse = []
    bike_start_mape = []
    bike_end_mape = []
    taxi_start_loss = []
    taxi_end_loss = []
    taxi_start_rmse = []
    taxi_end_rmse = []
    taxi_start_mape = []
    taxi_end_mape = []

    infer_time = 0
    action_time = 0
    tmp = 0
    action_list = []
    while not done:
        # t1 = time.time()
        # action = agent.act(state, epsilon=0.0)  # Always choose the best action during testing
        # t2 = time.time()
        if t > 0:
            t1 = time.time()
            action = agent.act(state, epsilon=0.0)  # Always choose the best action during testing
            t2 = time.time()
        else:
            action = torch.zeros((1), device=device)
            t1 = t2 = 0

        # if t % 3 == 0:
        #     action = torch.zeros((1), device=device)
        # else:
        #     action = torch.ones((1), device=device)
        tmp += action
        action_list.append(action)

        next_state, pred_final, done, model_time= env.steptest(t, action)  # Use test_mode=True
        # next_state, pred_final, done, model_time= env.step_best(t, action)  # Use test_mode=True

        if t > 2:
            infer_time += model_time
            action_time += t2 -t1
        if done is True: 
            break
        state = next_state
        bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics = env.get_acc(t, pred_final)
        
       
        bike_start_loss.append(bike_start_metrics[0])
        bike_end_loss.append(bike_end_metrics[0])
        bike_start_rmse.append(bike_start_metrics[1])
        bike_end_rmse.append(bike_end_metrics[1])
        bike_start_mape.append(bike_start_metrics[2])
        bike_end_mape.append(bike_end_metrics[2])

        taxi_start_loss.append(taxi_start_metrics[0])
        taxi_end_loss.append(taxi_end_metrics[0])
        taxi_start_rmse.append(taxi_start_metrics[1])
        taxi_end_rmse.append(taxi_end_metrics[1])
        taxi_start_mape.append(taxi_start_metrics[2])
        taxi_end_mape.append(taxi_end_metrics[2])
        log = 'Iter: {:02d}\nTest Bike Start MAE: {:.4f}, Test Bike End MAE: {:.4f}, ' \
              'Test Taxi Start MAE: {:.4f}, Test Taxi End MAE: {:.4f}, \n' \
              'Test Bike Start RMSE: {:.4f}, Test Bike End RMSE: {:.4f}, ' \
              'Test Taxi Start RMSE: {:.4f}, Test Taxi End RMSE: {:.4f}, \n' \
              'Test Bike Start MAPE: {:.4f}, Test Bike End MAPE: {:.4f}, ' \
              'Test Taxi Start MAPE: {:.4f}, Test Taxi End MAPE: {:.4f}'
        print(
            log.format(t, bike_start_metrics[0]*config['bike_volume_max'], bike_end_metrics[0]*config['bike_volume_max'], taxi_start_metrics[0]*config['taxi_volume_max'], taxi_end_metrics[0]*config['taxi_volume_max'],
                       bike_start_metrics[1]*config['bike_volume_max'], bike_end_metrics[1]*config['bike_volume_max'], taxi_start_metrics[1]*config['taxi_volume_max'], taxi_end_metrics[1]*config['taxi_volume_max'],
                       bike_start_metrics[2]*100, bike_end_metrics[2]*100, taxi_start_metrics[2]*100, taxi_end_metrics[2]*100, ))
        t += 1
    print("-----------------------------------------------------------------")
    # log1 = 'Average test bike start MAE: {:.4f}, Average test bike end MAE: {:.4f}, ' \
    #        'Average test taxi start MAE: {:.4f}, Average test taxi end MAE: {:.4f}, \n' \
    #        'Average test bike start RMSE: {:.4f}, Average test bike end RMSE: {:.4f}, ' \
    #        'Average test taxi start RMSE: {:.4f}, Average test taxi end RMSE: {:.4f}, \n' \
    #        'Average test bike start MAPE: {:.4f}, Average test bike end MAPE: {:.4f},' \
    #        'Average test taxi start MAPE: {:.4f}, Average test taxi end MAPE: {:.4f}'
    log1 = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'

    print(log1.format((np.mean(bike_start_loss))*config['bike_volume_max'], (np.mean(bike_end_loss))*config['bike_volume_max'],
                      (np.mean(taxi_start_loss))*config['taxi_volume_max'], (np.mean(taxi_end_loss))*config['taxi_volume_max'],
                      (np.mean(bike_start_rmse))*config['bike_volume_max'], (np.mean(bike_end_rmse))*config['bike_volume_max'],
                      (np.mean(taxi_start_rmse))*config['taxi_volume_max'], (np.mean(taxi_end_rmse))*config['taxi_volume_max'],
                      (np.mean(bike_start_mape))*100, (np.mean(bike_end_mape))*100,
                      (np.mean(taxi_start_mape))*100, (np.mean(taxi_end_mape))*100))
    

    # 计算并输出总耗时
    print(f"infer_time: {infer_time} seconds")
    print(f"action_time: {action_time} seconds")
    print(f"total_time: {infer_time+action_time} seconds")
    print("action:", 265-tmp)
    # 使用 enumerate 获取索引和元素
    for index, action in enumerate(action_list):
        if action == 0:
            continue
        print(index, action)  # 输出序号和元素
    return 


if __name__ == '__main__':

    # load data
    bikevolume_test_save_path = os.path.join(args.bike_base_path, 'BV_test.npy')
    bikeflow_test_save_path = os.path.join(args.bike_base_path, 'BF_test.npy')
    taxivolume_test_save_path = os.path.join(args.taxi_base_path, 'TV_test.npy')
    taxiflow_test_save_path = os.path.join(args.taxi_base_path, 'TF_test.npy')

    bike_train_data = MyDataset_rl(bikevolume_test_save_path, args.window_size, args.batch_size)
    taxi_train_data = MyDataset_rl(taxivolume_test_save_path, args.window_size, args.batch_size)
    bike_adj_data = MyDataset_rl(bikeflow_test_save_path, args.window_size, args.batch_size)
    taxi_adj_data = MyDataset_rl(taxiflow_test_save_path, args.window_size, args.batch_size)

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
    model.load_state_dict(torch.load(args.save),strict=False)
    model.eval() 

    # Assuming the training is already done
    for iter, (bike_adj, bike_node, taxi_adj, taxi_node) in enumerate(zip(bike_adj_loader, bike_train_loader,
                                                                          taxi_adj_loader, taxi_train_loader)):
        env = TrafficEnv(bike_adj, bike_node, taxi_adj, taxi_node, model, device, window_size=24, pred_size=args.pred_size)

        agent = DQNAgent(device=device, state_size=462, batch_size=args.batch_size, action_size=2)
        agent.qnetwork.load_state_dict(torch.load(args.rlsave))  # Load pre-trained model weights
        agent.qnetwork.eval()
        # Test the agent
        test_dqn(agent, env)