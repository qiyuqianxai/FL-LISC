import json
import os
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# font_size = 16
# plt.rcParams.update({'font.size':font_size})

def getalljson(path, files):
    filelist = os.listdir(path)
    for file in filelist:
        # print(file,filecount)
        if file.lower().endswith('.json'):
            files.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            getalljson(os.path.join(path, file), files)
        else:
            pass

############### draw accuracy figure on MNIST ####################

# CL
with open("logs/CL_MNIST/loss_acc/round_0_10.json","r",encoding="utf-8")as f:
    content = json.load(f)
    stu_test_acc = content['stu_test_acc']
    split_interval = int(len(stu_test_acc)/50)
    CL_ACC = []
    for i in range(0,len(stu_test_acc),split_interval):
        if i+split_interval < len(stu_test_acc):
            CL_ACC.append(np.mean(stu_test_acc[i:i+split_interval]))
CL_ACC = CL_ACC[:50]

# FedAvg
files = []
getalljson("logs/FedAvg_MNIST_2/",files)
FedAvg_ACC = []
for com_round in range(50):
    round_acc = []
    for file in files:
        if file.find(f"round_{com_round}_") > 0:
            with open(file,"r",encoding="utf-8")as f:
                content = json.load(f)
                acc = np.mean(content['stu_test_acc'])
                round_acc.append(acc)
    FedAvg_ACC.append(np.mean(round_acc))

FedAvg_ACC_for_clients_stu = []
for client_id in range(9):
    client_stu_acc = []
    for file in files:
        if file.find(f"/{client_id}\\")>0:
            with open(file,"r",encoding="utf-8")as f:
                content = json.load(f)
                acc = content['stu_test_acc']
                client_stu_acc.append(np.mean(acc))
    client_stu_acc = [val for i, val in enumerate(client_stu_acc) if i % 5 == 0]
    FedAvg_ACC_for_clients_stu.append(client_stu_acc[:10])
with open("logs/FTTQ_MNIST","r",encoding="utf-8")as f:
    contents = f.read()
import re
req_1 = "Global loss .*?,"
req_2 = "Global Acc .*?\n"
FTTQ_loss = re.findall(req_1,contents)
FTTQ_acc = re.findall(req_2,contents)
FTTQ_loss = [float(loss.replace("Global loss ","").replace(",","")) for loss in FTTQ_loss]
FTTQ_acc = [float(acc.replace("Global Acc ","").replace("\n","")) for acc in FTTQ_acc]
FTTQ_acc = sorted(FTTQ_acc)
# print(FTTQ_acc)

# STC
res = np.load("logs/STC_MNIST.npz")
STC_loss = res['loss_test']
STC_acc = res['accuracy_test'][:50]

# ours
files = []
getalljson("logs/KDHT_FedAP_MNIST_3/",files)
KF_ACC = []
for com_round in range(50):
    round_acc = []
    for file in files:
        if file.find(f"round_{com_round}_") > 0:
            with open(file,"r",encoding="utf-8")as f:
                content = json.load(f)
                acc = content['stu_test_acc']
                round_acc.append(acc)
    KF_ACC.append(np.mean(round_acc))

KF_ACC_for_clients_stu = []
# KF_ACC_for_clients_mentor = []
for client_id in range(9):
    client_stu_acc = []
    client_mentor_acc = []
    for file in files:
        if file.find(f"/{client_id}\\")>0:
            with open(file,"r",encoding="utf-8")as f:
                content = json.load(f)
                acc = content['stu_test_acc']
                client_stu_acc.append(acc)
                # acc = content['mentor_test_acc']
                # client_mentor_acc.append(acc)
    # client_mentor_acc = [val for i,val in enumerate(client_mentor_acc) if i%5==0]
    client_stu_acc = [val for i,val in enumerate(client_stu_acc) if i%5==0]
    # KF_ACC_for_clients_mentor.append(client_mentor_acc[:10])
    KF_ACC_for_clients_stu.append(client_stu_acc[:10])

font_size = 16
plt.figure(figsize=(12,9))
plt.rcParams.update({'font.size':font_size})
plt.plot(np.array(KF_ACC), c='r', label='KDAT+FedAP',marker="v")
plt.plot(np.array(FedAvg_ACC),c='b',label='FedAvg',marker='x')
plt.plot(np.array(FTTQ_acc),c='g',label='FTTQ',marker='o')
plt.plot(np.array(STC_acc), c='y', label='STC',marker='+')
plt.plot(np.array(CL_ACC),c='purple',label='CL',marker='.')
plt.legend(loc='best')
plt.ylabel('Test Accuracy')
plt.xlabel('Communication Rounds')
plt.xticks()
plt.yticks()
# plt.xlim(-0.1,49.1)
plt.grid()
plt.savefig("Global_Accuracy_on_MNIST.png",bbox_inches='tight', pad_inches=0.2)


######## draw local accuracy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
import random
f, ax = plt.subplots(figsize=(12, 9))
acc_records_1 = np.array(KF_ACC_for_clients_stu)
acc_records_2 = np.array(FedAvg_ACC_for_clients_stu)
acc_records = acc_records_1 - acc_records_2
acc_records = pd.DataFrame(acc_records)
# acc_records.index = [f"Client {i}" for i in range(9)]
acc_records.columns = [i+1 for i in range(50) if (i+1)%5==0]
sns.set(font_scale=1.25)
hm = sns.heatmap(acc_records,
                 cbar=True,
                 annot=True,
                 square=True,
                 # fmt=".3f",
                 vmin=0,  # 刻度阈值
                 vmax=1,
                 linewidths=.5,
                 cmap="YlGnBu",  # 刻度颜色
                 annot_kws={"size": 10},
                 xticklabels=True,
                 yticklabels=True)  # seaborn.heatmap相关属性

plt.ylabel("Index of Clients")
plt.xlabel("Communication Rounds")

# plt.title("主要变量之间的相关性强弱", fontsize=20)
plt.savefig("lcoal_acc.png",bbox_inches='tight', pad_inches=0.2)
# plt.show()


############### draw accuracy figure on CIFAR10 ####################

# with open("logs/CL_CIFAR10/loss_acc/round_0_10.json","r",encoding="utf-8")as f:
#     content = json.load(f)
#     stu_test_acc = content['stu_test_acc']
#     split_interval = int(len(stu_test_acc)/20)
#     CL_ACC = []
#     for i in range(0,len(stu_test_acc),split_interval):
#         if i+split_interval < len(stu_test_acc):
#             CL_ACC.append(np.mean(stu_test_acc[i:i+split_interval]))
# print(CL_ACC)
#
# # FedAvg
# files = []
# getalljson("logs/FedAvg_CIFAR10/",files)
# FedAvg_ACC = []
# for com_round in range(20):
#     round_acc = []
#     for file in files:
#         if file.find(f"round_{com_round}_") > 0:
#             with open(file,"r",encoding="utf-8")as f:
#                 content = json.load(f)
#                 acc = np.mean(content['stu_test_acc'])
#                 round_acc.append(acc)
#     FedAvg_ACC.append(np.mean(round_acc))
# print(FedAvg_ACC)
#
# # FTTQ
# with open("logs/FTTQ_CIFAR10","r",encoding="utf-8")as f:
#     contents = f.read()
# import re
# req_1 = "Global loss .*?,"
# req_2 = "Global Acc .*?\n"
# FTTQ_loss = re.findall(req_1,contents)
# FTTQ_acc = re.findall(req_2,contents)
# FTTQ_loss = [float(loss.replace("Global loss ","").replace(",","")) for loss in FTTQ_loss]
# FTTQ_acc = [float(acc.replace("Global Acc ","").replace("\n","")) for acc in FTTQ_acc]
# FTTQ_acc = sorted(FTTQ_acc)
# print(FTTQ_acc)
#
# # STC
# res = np.load("logs/STC_CIFAR10.npz")
# STC_loss = res['loss_test']
# STC_acc = res['accuracy_test']
# # for i in range(9):
# #     print(i,res[f'client{i}_acc'])
# # print(res.files)
#
# # ours
# files = []
# getalljson("logs/KDHT_FedAP_CIFAR10/",files)
# KF_ACC = []
# for com_round in range(20):
#     round_acc = []
#     for file in files:
#         if file.find(f"round_{com_round}_") > 0:
#             with open(file,"r",encoding="utf-8")as f:
#                 content = json.load(f)
#                 acc = content['mentor_test_acc']
#                 round_acc.append(acc)
#     KF_ACC.append(np.mean(round_acc))
# print(KF_ACC)
# plt.figure()
# plt.plot(np.array(KF_ACC), c='r', label='KDHT+FedAP')
# plt.plot(np.array(FTTQ_acc),c='b',label='FTTQ')
# plt.plot(np.array(FedAvg_ACC),c='g',label='FedAvg')
# plt.plot(np.array(STC_acc), c='y', label='STC')
# plt.legend(loc='best')
# plt.ylabel('Test Accuracy')
# plt.xlabel('Communication Rounds')
# plt.xticks()
# plt.yticks()
# plt.grid()
# plt.savefig("Global_Accuracy_on_CIFAR10.png",bbox_inches='tight', pad_inches=0.2)
# ############### draw communication overhead figure ####################
# plt.figure()
# val = [np.sum(np.array(rl_e)),np.sum(np.array(ea_e)),
#        np.sum(np.array(fl_e)),np.sum(np.array(ra_e))]
# print(val)
# plt.bar(["LANCE with\nUEQN","LANCE with\nDEA","FL","LANCE with\nRA"], val, color='rbgy',alpha=0.5)  # or `color=['r', 'g', 'b']`
# plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
# plt.ylabel('Energy consumption (J)',fontsize=font_size+2)
# plt.xlabel('Methods',fontsize=font_size+2)
# plt.xticks(size=font_size-4)
# plt.yticks(size=font_size-4)
# plt.savefig("LANCE_ea_fl_ra_e.png",bbox_inches='tight', pad_inches=0.2)