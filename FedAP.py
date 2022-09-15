import torch
from torch import nn
import numpy as np
import json
from Config import Config
import random
import copy
import os
from nets.student_nets import StuNet
from nets.metor_nets import MentorNet
from nets.channel_nets import channel_net
from nets.base_nets import base_net
from data_utils import get_data_loaders
from KDHT import KDHT,Train_for_weak_clients
# torch.cuda.set_device(0)
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def FedAP(stu_model, mentor_model, clients_loader,train_dataloader, test_dataloader, cfg):
    strong_client_ids = random.sample(range(cfg.n_clients), cfg.n_strong_clients)
    print("strong_client_ids:", strong_client_ids)
    SNR_for_clients = random.sample(range(cfg.SNR_MIN,cfg.SNR_MAX),cfg.n_clients)
    print("SNR_for_clients", SNR_for_clients)
    for com_round in range(cfg.communication_rounds):
        weights_for_clients = []
        print(f"communication_round {com_round} start...")
        for client_id in range(cfg.n_clients):
            # Train strong clients with KDHT
            if client_id in strong_client_ids:
                stu_weights = KDHT(stu_model,mentor_model,clients_loader[client_id],test_dataloader,cfg, SNR_for_clients[client_id],client_id,com_round)
            else:
                stu_weights = Train_for_weak_clients(stu_model,clients_loader[client_id],test_dataloader,cfg,SNR_for_clients[client_id],client_id,com_round)
            # Pruning for weights
            stu_weights = Adaptive_Pruning(stu_weights,SNR_for_clients[client_id],cfg.SNR_MAX,cfg.SNR_MIN)
            weights_for_clients.append(stu_weights)

        # server do:
        print("merge weights")
        w_avg = copy.deepcopy(weights_for_clients[0])
        for k in w_avg.keys():
            if "num_batches_tracked" in k:
                continue
            w_avg[k] = w_avg[k] * clients_loader[0].dataset.__len__()
            for i in range(1,cfg.n_clients):
                w_avg[k] += w_avg[k] * clients_loader[i].dataset.__len__()
            w_avg[k] = torch.div(w_avg[k],train_dataloader.dataset.__len__())

        print("broadcast weights to clients")
        for i in range(cfg.n_clients):
            checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{i}")
            os.makedirs(checkpoint_path, exist_ok=True)
            channel_name = "rali" if cfg.use_Rali else "awgn"
            SNR = SNR_for_clients[i]
            stu_weights_path = os.path.join(checkpoint_path, f"student_{SNR}_{channel_name}.pth")
            torch.save(w_avg, stu_weights_path)

def Adaptive_Pruning(weights, SNR_client, SNR_MAX, SNR_MIN):
    prune_ratio = (SNR_client-SNR_MIN)/(SNR_MAX-SNR_MIN)
    print("prune ratio:",prune_ratio)
    if prune_ratio == 0:
        return weights
    temp_weigths = copy.deepcopy(weights)
    for k in weights.keys():
        if "num_batches_tracked" in k:
            continue
        shapes = weights[k].shape
        temp = weights[k].view(-1)
        k = int(prune_ratio * temp.shape[0])
        vals, p_indexes = torch.topk(temp.abs(), k=k, dim=-1, largest=False)
        p_weights = torch.zeros_like(temp)
        for i in range(p_weights.shape[0]):
            if i not in p_indexes:
                p_weights[i] = temp[i]
            else:
                p_weights[i] = 0 # 1e-5
        p_weights = p_weights.view(shapes)
        print(p_weights)
        temp_weigths[k] = p_weights
    print("pruning weights",temp_weigths)
    return temp_weigths

if __name__ == '__main__':
    # hyparametes set
    same_seeds(2048)
    cfg = Config()
    # prepare data
    clients_loader, train_loader, test_loader, stats = get_data_loaders(cfg, True)
    print(stats)
    # prepare model
    Stu_net = StuNet(cfg.Stu_model_name)
    Stu_model = base_net(Stu_net, channel_net(snr=cfg.SNR, rali=cfg.use_Rali, if_RTN=cfg.use_RTN))
    Stu_model.to(cfg.device)
    Mentor_net = MentorNet()
    Mentor_model = base_net(Mentor_net, channel_net(snr=cfg.SNR, rali=cfg.use_Rali, if_RTN=cfg.use_RTN))

    FedAP(Stu_model, Mentor_model, clients_loader, train_loader, test_loader, cfg)

    # Adaptive_Pruning(Stu_model.state_dict(),cfg.SNR,cfg.SNR_MAX,cfg.SNR_MIN)




