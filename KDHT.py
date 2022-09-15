############### test KDHT ##############
from nets.student_nets import StuNet
from nets.metor_nets import MentorNet
from nets.channel_nets import channel_net
from nets.base_nets import base_net
import torch
import random
from torch import nn
import torchvision
import numpy as np
import os
from torch.nn import functional as F
from data_utils import get_data_loaders
from Config import Config
import json

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

# show image and save
def save_images(y, x_rec, save_pth):
    os.makedirs(save_pth,exist_ok=True)
    imgs_sample = (y.data + 1) / 2.0
    filename = os.path.join(save_pth,"raw.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    os.makedirs(os.path.join(save_pth, "all_imgs_raw"), exist_ok=True)
    for i in range(len(imgs_sample)):
        torchvision.utils.save_image(imgs_sample[i], os.path.join(save_pth, f"all_imgs_raw/{i}.jpg"))
    # Show 32 of the images.
    # grid_img = torchvision.utils.make_grid(imgs_sample[:100].cpu(), nrow=10)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # print("origin images")
    # plt.show()

    imgs_sample = (x_rec.data + 1) / 2.0
    filename = os.path.join(save_pth, "rec.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=1)
    os.makedirs(os.path.join(save_pth, "all_imgs_rec"), exist_ok=True)
    for i in range(len(imgs_sample)):
        torchvision.utils.save_image(imgs_sample[i], os.path.join(save_pth, f"all_imgs_rec/{i}.jpg"))
    # Show 32 of the images.
    # grid_img = torchvision.utils.make_grid(imgs_sample[:100].cpu(), nrow=10)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.title("reconstruct images")
    # plt.show()

# training based on KDHT
def KDHT(stu_model, mentor_model, train_dataloader, test_dataloader, cfg, client_snr = None, client_id=0,com_round=0):
    checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{client_id}")
    os.makedirs(checkpoint_path, exist_ok=True)
    channel_name = "rali" if cfg.use_Rali else "awgn"
    SNR = client_snr if client_snr else cfg.SNR
    # laod weights
    mentor_weights_path = os.path.join(checkpoint_path,f"mentor_{SNR}_{channel_name}.pth")
    if os.path.exists(mentor_weights_path):
        weights = torch.load(mentor_weights_path,map_location="cpu")
        mentor_model.load_state_dict(weights)
    stu_weights_path = os.path.join(checkpoint_path, f"student_{SNR}_{channel_name}.pth")
    if os.path.exists(stu_weights_path):
        weights = torch.load(stu_weights_path, map_location="cpu")
        stu_model.load_state_dict(weights)

    stu_model.to(cfg.device)
    mentor_model.to(cfg.device)
    # define optimizer
    optimizer_stu_encoder = torch.optim.Adam(stu_model.isc_model.encoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_decoder = torch.optim.Adam(stu_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_channel = torch.optim.Adam(stu_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_mentor_encoder = torch.optim.Adam(mentor_model.isc_model.encoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_mentor_decoder = torch.optim.Adam(mentor_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_mentor_channel = torch.optim.Adam(mentor_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=cfg.weight_delay)

    # define loss function
    mse = nn.MSELoss()
    kl = nn.KLDivLoss()
    # training
    train_mentor_loss = []
    train_stu_loss = []
    for epoch in range(cfg.epochs_for_clients):
        stu_model.train()
        mentor_model.train()
        for x,y in train_dataloader:
            optimizer_stu_encoder.zero_grad()
            optimizer_stu_channel.zero_grad()
            optimizer_stu_decoder.zero_grad()
            optimizer_mentor_encoder.zero_grad()
            optimizer_mentor_channel.zero_grad()
            optimizer_mentor_decoder.zero_grad()
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            mentor_s_encoding, mentor_c_decoding, mentor_x_rec = mentor_model(x)
            stu_s_encoding, stu_c_decoding, stu_x_rec = stu_model(x)
            # compute loss
            l_mentor_task = mse(mentor_x_rec,y)
            l_stu_task = mse(stu_x_rec,y)
            mentor_dist_1 = F.log_softmax(mentor_x_rec)
            mentor_dist_2 = F.softmax(mentor_x_rec)
            stu_dist_1 = F.log_softmax(stu_x_rec)
            stu_dist_2 = F.softmax(stu_x_rec)
            l_mentor_dis = kl(mentor_dist_1,stu_dist_2)/(l_mentor_task+l_stu_task)
            l_stu_dis = kl(stu_dist_1,mentor_dist_2)/(l_mentor_task+l_stu_task)
            l_mentor_hid = l_stu_hid = (mse(stu_s_encoding,mentor_s_encoding) + mse(stu_c_decoding,mentor_c_decoding))\
                                       /(l_mentor_task+l_stu_task)
            l_stu = l_stu_task+l_stu_dis+l_stu_hid
            l_mentor = l_mentor_task+l_mentor_dis+l_mentor_hid
            total_loss = l_stu+l_mentor
            total_loss.backward()
            optimizer_stu_encoder.step()
            optimizer_stu_decoder.step()
            optimizer_stu_channel.step()
            optimizer_mentor_decoder.step()
            optimizer_mentor_encoder.step()
            optimizer_mentor_channel.step()

            print(f"client{client_id}-epoch {epoch} | student loss:{l_stu} | task_loss:{l_stu_task} | dis_loss:{l_stu_dis} | hid_loss:{l_stu_hid}")
            print(f"client{client_id}-epoch {epoch} | mentor loss:{l_mentor} | task_loss:{l_mentor_task} | dis_loss:{l_mentor_dis} | hid_loss:{l_mentor_hid}")

            train_stu_loss.append(l_stu.item())
            train_mentor_loss.append(l_mentor.item())
        # save_weights
        torch.save(mentor_model.state_dict(),mentor_weights_path)
        torch.save(stu_model.state_dict(),stu_weights_path)

    # testing
    test_mentor_loss, test_stu_loss = Test_KDHT_ISC(stu_model, mentor_model, test_dataloader, cfg)
    train_mentor_loss = np.mean(train_mentor_loss)
    train_stu_loss = np.mean(train_stu_loss)
    loss_records = {"stu_train_loss":train_stu_loss,"mentor_train_loss":train_mentor_loss,"stu_test_loss": test_stu_loss,"mentor_test_loss": test_mentor_loss}
    with open(os.path.join(cfg.logs_dir, f"{client_id}", "loss", f"round_{com_round}_loss.json"), "w",
              encoding="utf-8")as f:
        f.write(json.dumps(loss_records, ensure_ascii=False, indent=4))
    return stu_model.state_dict()

# test student and mentor models
def Test_KDHT_ISC(stu_model, mentor_model, test_dataloader, cfg, client_id=1):
    stu_model.to(cfg.device)
    mentor_model.to(cfg.device)
    stu_model.eval()
    mentor_model.eval()
    # define loss function
    mse = nn.MSELoss()
    kl = nn.KLDivLoss()
    test_stu_loss = []
    test_mentor_loss = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            mentor_s_encoding, mentor_c_decoding, mentor_x_rec = mentor_model(x)
            stu_s_encoding, stu_c_decoding, stu_x_rec = stu_model(x)
            # compute loss
            l_mentor_task = mse(mentor_x_rec, y)
            l_stu_task = mse(stu_x_rec, y)

            mentor_dist_1 = F.log_softmax(mentor_x_rec)
            mentor_dist_2 = F.softmax(mentor_x_rec)
            stu_dist_1 = F.log_softmax(stu_x_rec)
            stu_dist_2 = F.softmax(stu_x_rec)
            l_mentor_dis = kl(mentor_dist_1, stu_dist_2) / (l_mentor_task + l_stu_task)
            l_stu_dis = kl(stu_dist_1, mentor_dist_2) / (l_mentor_task + l_stu_task)
            l_mentor_hid = l_stu_hid = (mse(stu_s_encoding, mentor_s_encoding) + mse(stu_c_decoding, mentor_c_decoding)) \
                                       / (l_mentor_task + l_stu_task)
            l_stu = l_stu_task + l_stu_dis + l_stu_hid
            l_mentor = l_mentor_task + l_mentor_dis + l_mentor_hid
            print(
                f"client_id{client_id}-test | student loss:{l_stu} | task_loss:{l_stu_task} | dis_loss:{l_stu_dis} | hid_loss:{l_stu_hid}")
            print(
                f"client_id{client_id}-test | mentor loss:{l_mentor} | task_loss:{l_mentor_task} | dis_loss:{l_mentor_dis} | hid_loss:{l_mentor_hid}")

            test_stu_loss.append(l_stu.item())
            test_mentor_loss.append(l_mentor.item())
            save_images(y,mentor_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","mentor_imgs"))
            save_images(y,stu_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","student_imgs"))
    test_stu_loss = np.mean(test_stu_loss)
    test_mentor_loss = np.mean(test_mentor_loss)
    return test_mentor_loss, test_stu_loss

def Train_for_weak_clients(stu_model, train_dataloader, test_dataloader, cfg, client_snr=None, client_id=1,com_round=1):
    checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{client_id}")
    os.makedirs(checkpoint_path, exist_ok=True)
    channel_name = "rali" if cfg.use_Rali else "awgn"
    SNR = client_snr if client_snr else cfg.SNR
    # laod weights
    stu_weights_path = os.path.join(checkpoint_path, f"student_{SNR}_{channel_name}.pth")
    if os.path.exists(stu_weights_path):
        weights = torch.load(stu_weights_path, map_location="cpu")
        stu_model.load_state_dict(weights)

    stu_model.to(cfg.device)
    # define optimizer
    optimizer_stu_encoder = torch.optim.Adam(stu_model.isc_model.encoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_decoder = torch.optim.Adam(stu_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_channel = torch.optim.Adam(stu_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=cfg.weight_delay)

    # define loss function
    mse = nn.MSELoss()
    train_stu_loss = []
    # training
    for epoch in range(cfg.epochs_for_clients):
        train_stu_loss = []
        stu_model.train()
        for x,y in train_dataloader:
            optimizer_stu_encoder.zero_grad()
            optimizer_stu_channel.zero_grad()
            optimizer_stu_decoder.zero_grad()
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            stu_s_encoding,stu_c_decoding,stu_x_rec = stu_model(x)
            # compute loss

            l_stu_task = mse(stu_x_rec,y)
            l_stu_coding = mse(stu_s_encoding,stu_c_decoding)
            l_stu = l_stu_task+l_stu_coding
            l_stu.backward()
            optimizer_stu_encoder.step()
            optimizer_stu_decoder.step()
            optimizer_stu_channel.step()

            print(f"client{client_id}-epoch {epoch} | student loss:{l_stu} | task_loss:{l_stu_task} | code_loss:{l_stu_coding}")

            train_stu_loss.append(l_stu.item())

        # save_weights
        torch.save(stu_model.state_dict(), stu_weights_path)


    # testing
    test_stu_loss = Test_Stu_ISC(stu_model,test_dataloader, cfg)
    train_stu_loss = np.mean(train_stu_loss)
    loss_records = {"stu_train_loss":train_stu_loss,"stu_test_loss":test_stu_loss}
    with open(os.path.join(cfg.logs_dir,f"{client_id}","loss",f"round_{com_round}_loss.json"),"w",encoding="utf-8")as f:
        f.write(json.dumps(loss_records,ensure_ascii=False,indent=4))
    return stu_model.state_dict()

def Test_Stu_ISC(stu_model, test_dataloader, cfg, client_id=1):
    stu_model.to(cfg.device)
    stu_model.eval()
    # define loss function
    mse = nn.MSELoss()
    test_stu_loss = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            stu_s_encoding, stu_c_decoding, stu_x_rec = stu_model(x)
            # compute loss
            l_stu_task = mse(stu_x_rec, y)
            l_stu_coding = mse(stu_s_encoding,stu_c_decoding)

            l_stu = l_stu_task + l_stu_coding

            print(
                f"client_id{client_id}-test | student loss:{l_stu} | task_loss:{l_stu_task} | code_loss:{l_stu_coding}")

            test_stu_loss.append(l_stu.item())
            save_images(y,stu_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","student_imgs"))
    test_stu_loss = np.mean(test_stu_loss)
    return test_stu_loss

if __name__ == '__main__':
    # hyparametes set
    same_seeds(2048)
    config = Config()
    # prepare data
    _, train_loader, test_loader, stats = get_data_loaders(config, True)
    print(stats)

    # prepare model
    Stu_net = StuNet(config.Stu_model_name)
    Stu_model = base_net(Stu_net, channel_net(snr=config.SNR, rali=config.use_Rali, if_RTN=config.use_RTN))
    Mentor_net = MentorNet()
    Mentor_model = base_net(Mentor_net, channel_net(snr=config.SNR, rali=config.use_Rali, if_RTN=config.use_RTN))

    KDHT(Stu_model,Mentor_model,train_loader,test_loader,config)













