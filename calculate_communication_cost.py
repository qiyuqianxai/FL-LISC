import random
import json
import os
from matplotlib import pyplot as plt
import numpy as np
communication_rounds = 10
param_size = 35.17
ratio = 1
clients = 9
max_snr = 50
min_snr = 0
B_for_clients = 10
p_kt = 0.5
test_count = 30
inter_val = 2
font_size = 16
plt.rcParams.update({'font.size':font_size})
plt.figure(figsize=(12,9))
# 1
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1+com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients*np.log2(1+c_snr)
            t = param_size/v
            e = p_kt*t
            ce.append(e)
        mean_std.append(mean_SNR)
    mean_std = np.std(mean_std)
    ce = sum(ce)
    res.append([mean_std,ce])
res = sorted(res,key=lambda x:x[0])
res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
print(np.std(res_y))
plt.plot(res_x,res_y, c='b', label='FedAvg',marker="x")

# 2
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1+com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients*np.log2(1+c_snr)
            prune_param = param_size*((c_snr-min_snr)/(max_snr-min_snr))
            t = prune_param/v
            e = p_kt*t
            ce.append(e)
        mean_std.append(mean_SNR)
    mean_std = np.std(mean_std)
    ce = sum(ce)
    res.append([mean_std,ce])
res = sorted(res,key=lambda x:x[0])
res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
print(np.std(res_y))
plt.plot(res_x,res_y, c='r', label='CEFL',marker="v")

# 3
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1+com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        mean_std.append(mean_SNR)
        SNR_for_each_clinets = random.sample(SNR_for_each_clinets,int(len(SNR_for_each_clinets)*0.6))
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients*np.log2(1+c_snr)
            prune_param = param_size
            t = prune_param/v
            e = p_kt*t
            ce.append(e)
    mean_std = np.std(mean_std)
    ce = sum(ce)
    res.append([mean_std,ce])
res = sorted(res,key=lambda x:x[0])
# res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
print(np.std(res_y))
plt.plot(res_x,res_y, c='g', label='FTTQ',marker="o")

# 4
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1+com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        mean_std.append(mean_SNR)
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients*np.log2(1+c_snr)
            prune_param = param_size*0.6
            t = prune_param/v
            e = p_kt*t
            ce.append(e)
    mean_std = np.std(mean_std)
    ce = sum(ce)

    res.append([mean_std,ce])
res = sorted(res,key=lambda x:x[0])
res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
print(np.std(res_y))
plt.plot(res_x,res_y, c='y', label='STC',marker="+")

plt.legend(loc='best')
plt.ylabel('Communication Overhead (J)')
plt.xlabel('Standard Deviation of the mean SNR of each communication round')
# plt.xticks(range(10),[i+1 for i in range(50) if (i+1)%5==0])
plt.yticks()
# plt.xlim(-0.1,49.1)
plt.grid()
plt.savefig("CE_vs_SNR.png",bbox_inches='tight', pad_inches=0.2)

###################################################################################################
# 1
plt.figure(figsize=(12,9))
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1 + com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients * np.log2(1 + c_snr)
            t = param_size / v
            e = p_kt * t
            ce.append(e)
        mean_std.append(mean_SNR)
    mean_std = np.std(mean_std)
    ce = np.std(ce)
    res.append([mean_std, ce])
res = sorted(res, key=lambda x: x[0])
res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
ax = plt.subplot(813)
ax.plot(res_x, res_y, c='b', label='FedAvg', marker="x")
# ax.set_title("FedAvg")
ax.legend(loc='best')
# ax.set_ylabel('Standard Deviation of communication overhead')
ax.set_xlabel('Standard Deviation of SNR')
ax.set_ylim(0.01,0.25)
# 2
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1 + com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients * np.log2(1 + c_snr)
            prune_param = param_size * ((c_snr - min_snr) / (max_snr - min_snr))
            t = prune_param / v
            e = p_kt * t
            ce.append(e)
        mean_std.append(mean_SNR)
    mean_std = np.std(mean_std)
    ce = np.std(ce)
    res.append([mean_std, ce])
res = sorted(res, key=lambda x: x[0])
res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
print(np.std(res_y))
ax = plt.subplot(811)
ax.plot(res_x, res_y, c='r', label='CEFL', marker="v")
# ax.set_title("CEFL")
ax.legend(loc='best')
# ax.set_ylabel('Standard Deviation of communication overhead')
ax.set_xlabel('Standard Deviation of SNR')
ax.set_ylim(0.01,0.25)
# 3
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1 + com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        mean_std.append(mean_SNR)
        SNR_for_each_clinets = random.sample(SNR_for_each_clinets, int(len(SNR_for_each_clinets) * 0.6))
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients * np.log2(1 + c_snr)
            prune_param = param_size
            t = prune_param / v
            e = p_kt * t
            ce.append(e)
    mean_std = np.std(mean_std)
    ce = np.std(ce)
    res.append([mean_std, ce])
res = sorted(res, key=lambda x: x[0])
# res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
print(np.std(res_y))
ax = plt.subplot(815)
ax.plot(res_x, res_y, c='g', label='FTTQ', marker="o")
# ax.set_title("FTTQ")
ax.legend(loc='best')
# ax.set_ylabel('Standard Deviation of communication overhead')
ax.set_xlabel('Standard Deviation of SNR')
ax.set_ylim(0.01,0.25)

# 4
random.seed(2048)
np.random.seed(2048)
res = []
for i in range(test_count):
    ce = []
    mean_std = []
    for com in range(communication_rounds):
        SNR_for_each_clinets = random.sample(range(1 + com, max_snr, inter_val), clients)
        mean_SNR = np.mean(SNR_for_each_clinets)
        mean_std.append(mean_SNR)
        for c_snr in SNR_for_each_clinets:
            v = B_for_clients * np.log2(1 + c_snr)
            prune_param = param_size * 0.6
            t = prune_param / v
            e = p_kt * t
            ce.append(e)
    mean_std = np.std(mean_std)
    ce = np.std(ce)
    res.append([mean_std, ce])
res = sorted(res, key=lambda x: x[0])
res_x = [x[0] for x in res]
res_y = [x[1] for x in res]
print(np.std(res_y))
ax = plt.subplot(817)
ax.plot(res_x, res_y, c='y', marker="+",label="STC")
# ax.set_title("STC")
ax.legend(loc='best')
# ax.set_ylabel('Std of Communication Cost')
ax.set_xlabel('Standard Deviation of SNR')
ax.set_ylim(0.01,0.25)
plt.text(1.2, 1, 'Standard Deviation of communication overhead', va='center', rotation='vertical')
plt.savefig("std_CE_vs_SNR.png", bbox_inches='tight', pad_inches=0.2)