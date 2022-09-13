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
# hyparametes set
same_seeds(2048)
config = Config()
# prepare data
_, train_loader, test_loader, stats = get_data_loaders(config.hp,True)
print(stats)

# prepare model
Stu_net = StuNet(config.Stu_model_name)
Stu_model = base_net(Stu_net,channel_net(snr=config.SNR,rali=config.use_Rali,if_RTN=config.use_RTN))
Mentor_net = MentorNet()
Mentor_model = base_net(Mentor_net,channel_net(snr=config.SNR,rali=config.use_Rali,if_RTN=config.use_RTN))

# show image and save
def save_images(y, x_rec, save_pth):
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

criticisen = torch.nn.MSELoss()

# training based on KDHT
def KDHT(stu_model, mentor_model, train_dataloader, test_dataloader, cfg, client_id=1):
    stu_model.to(cfg.device)
    mentor_model.to(cfg.device)
    # define optimizer
    optimizer_stu_encoder = torch.optim.Adam(stu_model.isc_model.encoder.parameters(), lr=cfg.isc_lr/100,
                                             weight_decay=config.weight_delay)
    optimizer_stu_decoder = torch.optim.Adam(stu_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=config.weight_delay)
    optimizer_stu_channel = torch.optim.Adam(stu_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=config.weight_delay)
    optimizer_mentor_encoder = torch.optim.Adam(mentor_model.isc_model.encoder.parameters(), lr=cfg.isc_lr / 100,
                                             weight_decay=config.weight_delay)
    optimizer_mentor_decoder = torch.optim.Adam(mentor_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=config.weight_delay)
    optimizer_mentor_channel = torch.optim.Adam(mentor_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=config.weight_delay)

    # define loss function
    mse = nn.MSELoss()
    kl = nn.KLDivLoss()
    loss_records = []
    # training
    for epoch in range(cfg.local_epochs):
        train_mentor_loss = []
        train_stu_loss = []
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
            mentor_s_encoding,mentor_c_decoding,mentor_x_rec = mentor_model(x)
            stu_s_encoding,stu_c_decoding,stu_x_rec = stu_model(x)
            # compute loss
            l_mentor_task = mse(mentor_x_rec,y)
            l_stu_task = mse(stu_x_rec,y)
            l_mentor_dis = kl(mentor_x_rec,stu_x_rec)/(l_mentor_task+l_stu_task)
            l_stu_dis = kl(stu_x_rec,mentor_x_rec)/(l_mentor_task+l_stu_task)
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
            optimizer_stu_channel.step()

            print(f"client{client_id}-epoch {epoch} | student loss:{l_stu} | task_loss:{l_stu_task} | dis_loss:{l_stu_dis} | hid_loss:{l_stu_hid}")
            print(f"client{client_id}-epoch {epoch} | mentor loss:{l_mentor} | task_loss:{l_mentor_task} | dis_loss:{l_mentor_dis} | hid_loss:{l_mentor_hid}")

            train_stu_loss.append(l_stu.item())
            train_mentor_loss.append(l_mentor.item())
        train_stu_loss = np.mean(train_stu_loss)
        train_mentor_loss = np.mean(train_mentor_loss)

        # testing
        test_mentor_loss, test_stu_loss = test_ISC(stu_model,mentor_model, test_dataloader, cfg)
        loss_records.append([train_mentor_loss,train_stu_loss,test_mentor_loss, test_stu_loss])
        with open(os.path.join(cfg.logs_dir,f"{client_id}","loss",f"epoch_{epoch}_loss.json"),"w",encoding="utf-8")as f:
            f.write(json.dumps(loss_records,ensure_ascii=False,indent=4))

        # save_weights
        checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{client_id}")
        os.makedirs(checkpoint_path,exist_ok=True)
        channel_name = "rali" if cfg.use_Rali else "awgn"
        torch.save(mentor_model.state_dict(),
                   os.path.join(checkpoint_path,f"mentor_{cfg.SNR}_{channel_name}.pth"))
        torch.save(stu_model.state_dict(),
                   os.path.join(checkpoint_path, f"student_{cfg.SNR}_{channel_name}.pth"))
        return stu_model.state_dict()

# test student and mentor models
def test_ISC(stu_model, mentor_model, test_dataloader, cfg, client_id=1):
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
            l_mentor_dis = kl(mentor_x_rec, stu_x_rec) / (l_mentor_task + l_stu_task)
            l_stu_dis = kl(stu_x_rec, mentor_x_rec) / (l_mentor_task + l_stu_task)
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
            save_images(y,mentor_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","imgs"))
            save_images(y,stu_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","imgs"))
    test_stu_loss = np.mean(test_stu_loss)
    test_mentor_loss = np.mean(test_mentor_loss)
    return test_mentor_loss, test_stu_loss


if __name__ == '__main__':
    KDHT(Stu_model,Mentor_model,train_loader,test_loader,config)













