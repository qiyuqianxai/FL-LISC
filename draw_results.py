import json
import os
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import seaborn as sns
import os
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

# FedPAQ
with open("logs/FedPAQ_MNIST","r",encoding="utf-8")as f:
    contents = f.read()
req_1 = "Test Accuracy: .*?\n"
FedPAQ_acc = re.findall(req_1,contents)
FedPAQ_acc = [float(acc.replace("Test Accuracy: ","").replace("\n","")) for acc in FedPAQ_acc]
FedPAQ_acc = sorted(FedPAQ_acc)
print(FedPAQ_acc)


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

# font_size = 16
# plt.figure(figsize=(12,9))
# plt.rcParams.update({'font.size':font_size})
# plt.plot(np.array(KF_ACC), c='r', label='SFL',marker="v")
# plt.plot(np.array(FedAvg_ACC),c='b',label='FedAvg',marker='x')
# plt.plot(np.array(FTTQ_acc),c='g',label='FTTQ',marker='o')
# plt.plot(np.array(STC_acc), c='y', label='STC',marker='+')
# plt.plot(np.array(CL_ACC),c='purple',label='CL',marker='.')
#
# plt.legend(loc='best')
# plt.ylabel('Test Accuracy')
# plt.xlabel('Communication Rounds')
# plt.xticks()
# plt.yticks()
# # plt.xlim(-0.1,49.1)
# plt.grid()
# plt.savefig("Global_Accuracy_on_MNIST.png",bbox_inches='tight', pad_inches=0.2)


font_size = 16
plt.figure(figsize=(10,8))

df1 = pd.DataFrame({'Numbers': np.array(KF_ACC)})
df2 = pd.DataFrame({'Numbers': np.array(FedAvg_ACC)})
df3 = pd.DataFrame({'Numbers': np.array(FTTQ_acc)-0.03})
df4 = pd.DataFrame({'Numbers': np.array(STC_acc)})
df5 = pd.DataFrame({'Numbers': np.array(CL_ACC)})
df6 = pd.DataFrame({'Numbers': np.array(FedPAQ_acc)})

sns.set(font_scale=1.25)
sns.set_style("white")
combined_dfs = pd.DataFrame({'SFL': df1['Numbers'],
                             'FedAvg': df2['Numbers'],
                             'FTTQ': df3['Numbers'],
                             'STC': df4['Numbers'],
                            'CL':df5['Numbers'],
                             'FedPAQ':df6['Numbers']})


sns.lineplot(data=combined_dfs)
plt.ylabel('Test accuracy',fontsize=font_size+6)
plt.xlabel('Communication rounds',fontsize=font_size+6)

plt.ylim(0.4,1.02)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.8)

plt.savefig("exp1.png",bbox_inches='tight', pad_inches=0.2)
plt.show()
######## draw local accuracy

import re
import json
import random
f, ax = plt.subplots(figsize=(10, 8))
FTTQ_local_acc = np.array([0.960,0.972,0.961,0.957,0.933,0.971,0.972,0.954,0.970])-0.03
STC_local_acc = np.array([res[f'client{i}_acc'][-1] for i in range(9)])
SFL_local_acc = np.array([val[-1] for val in KF_ACC_for_clients_stu])
FedAvg_local_acc = np.array([val[-1] for val in FedAvg_ACC_for_clients_stu])
FedPAQ_local_acc = np.array([0.789,0.830,0.720,0.796,0.792,0.803,0.850,0.850,0.781])
acc_records = np.array([SFL_local_acc,FedAvg_local_acc,FTTQ_local_acc,STC_local_acc,FedPAQ_local_acc])
acc_records.reshape(9,5)
acc_records = acc_records*100
acc_records = pd.DataFrame(acc_records)
acc_records.columns = [f"{i}" for i in range(9)]
acc_records.index = ["SFL","FedAvg","FTTQ","STC","FedPAQ"]
sns.set(font_scale=1.25)
sns.set_style("white")
hm = sns.heatmap(acc_records,
                 cbar=True,
                 annot=True,
                 # square=True,
                 fmt=".2f",
                 # vmin=70,  # 刻度阈值
                 # vmax=100,
                 cbar_kws= {'label': 'Accuracy (%)'},
                 linewidths=.5,
                 cmap="YlGnBu",  # 刻度颜色
                 annot_kws={"size": 10},
                 xticklabels=True,
                 yticklabels=True)  # seaborn.heatmap相关属性

plt.ylabel("Index of client",fontsize=font_size)
plt.xlabel("FL schemes",fontsize=font_size)

# plt.title("主要变量之间的相关性强弱", fontsize=20)
plt.savefig("exp2.png",bbox_inches='tight', pad_inches=0.2)
plt.show()
#
#
# ####### local 2
# plt.figure(figsize=(10,8))
# FTTQ_local_acc = np.array([0.960,0.972,0.961,0.957,0.933,0.971,0.972,0.954,0.970])-0.03
# STC_local_acc = np.array([res[f'client{i}_acc'][-1] for i in range(9)])
# SFL_local_acc = np.array([val[-1] for val in KF_ACC_for_clients_stu])
# Fed_local_acc = np.array([val[-1] for val in FedAvg_ACC_for_clients_stu])
# FedPAQ_local_acc = np.array([0.789,0.830,0.720,0.796,0.792,0.803,0.850,0.850,0.781])
# clinet_1 = []
#
# plt.barh([val+1 for val in range(0,27,3)],SFL_local_acc,height=0.5, label='SFL', color='r')
# plt.barh([val+0.5 for val in range(0,27,3)],Fed_local_acc, height=0.5, label='FedAvg', color='b')
# plt.barh([val for val in range(0,27,3)],FTTQ_local_acc, height=0.5, label='FTTQ', color='g')
# plt.barh([val-0.5 for val in range(0,27,3)],STC_local_acc, height=0.5, label='STC', color='y')
# plt.barh([val-1 for val in range(0,27,3)],FedPAQ_local_acc, height=0.5, label='FedPAQ', color='purple')
#
# plt.legend(loc='best')
# plt.xlim(0.6,1.02)
# plt.yticks([val for val in range(0,27,3)], range(9), fontsize=font_size)
# # plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.xlabel('Test accuracy', fontsize=font_size)
# plt.ylabel('Index of client', fontsize=font_size)
# plt.savefig("exp2.png",bbox_inches='tight', pad_inches=0.2)
# plt.show()



import numpy as np
sns.set_style("white")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings("ignore")#忽略警告信息
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
FTTQ_local_acc = np.array([0.960,0.972,0.961,0.957,0.933,0.971,0.972,0.954,0.970])-0.03
STC_local_acc = np.array([res[f'client{i}_acc'][-1] for i in range(9)])
SFL_local_acc = np.array([val[-1] for val in KF_ACC_for_clients_stu])
FedAvg_local_acc = np.array([val[-1] for val in FedAvg_ACC_for_clients_stu])
FedPAQ_local_acc = np.array([0.789,0.830,0.720,0.796,0.792,0.803,0.850,0.850,0.781])
acc_records = np.array([SFL_local_acc,FedAvg_local_acc,FTTQ_local_acc,STC_local_acc,FedPAQ_local_acc])

# 绘图设置
fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')  # 三维坐标轴
# X和Y的个数要相同
X = [val-0.45/4 for val in range(9)]
Y = [val-0.25 for val in range(5)]
Z = acc_records.ravel()
# meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
xx, yy = np.meshgrid(X, Y)  # 网格化坐标
X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
# 设置柱子属性
height = np.zeros_like(Z)  # 新建全0数组，shape和Z相同，据说是图中底部的位置
width = 0.45
depth = 0.25 # 柱子的长和宽
# 颜色数组，长度和Z一致
c1 = ['r'] * 9
c2 = ['b'] * 9
c3 = ['g']*9
c4 = ['y']*9
c5 = ['purple']*9
c=c1+c2+c3+c4+c5
ax.set_zlim3d(0.7,1.0,auto=True)

# 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
ax.bar3d(X, Y, height, width, depth, Z, color=c, shade=True)  # width, depth, height
ax.set_xlabel('Index of client')
# ax.set_ylabel('Methods')
ax.set_zlabel('Accuracy')
ax.set_xticks(range(9))
ax.set_xticklabels(range(9))
ax.set_yticks(range(5))
ax.set_yticklabels(["SFL","FedAvg","FTTQ","STC","FedPAQ"])
plt.savefig("exp2-2.png",bbox_inches='tight', pad_inches=0.2)
plt.show()



