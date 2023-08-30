import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
from ALGE import ALGE_C
from other_algorithms import ci,k_shell,h_index,lid,dcl,ncvr,rcnn,glstm
import numpy as np
import math
import influence_evaluation.Model
from Utils import load_csv_net_gcc
import pickle


def calculate(G,data_memory):
    k_c,r_c,node_c,node_rank_CI = ci(G,data_memory)
    k_k,r_k,node_k,node_rank_K  = k_shell(G, data_memory)
    k_h,r_h,node_h,node_rank_H  = h_index(G, data_memory)
    k_l,r_l,node_lid,node_rank_L  = lid(G, data_memory)
    k_d,r_d,node_dcl,node_rank_D  = dcl(G, data_memory)
    k_n,r_n,node_ncvr,node_rank_N  = ncvr(G, data_memory)
    k_r,r_r,node_rcnn,node_rank_R = rcnn(G, data_memory)
    k_gl,r_gl,node_glstm,node_rank_G  = glstm(G, data_memory)
    k_algec,r_algec,node_algec,node_rank_algec,_  = ALGE_C(G, data_memory)
    ken_list = [k_c,k_k,k_h,k_l,k_d,k_n,k_r,k_gl,k_algec]
    rank_list = [r_c,r_k,r_h,r_l,r_d,r_n,r_r,r_gl,r_algec]
    node_sort = [node_c,node_k,node_h,node_lid,node_dcl,node_ncvr,node_rcnn,node_glstm,node_algec]
    node_rank = [node_rank_CI,node_rank_K,node_rank_H,node_rank_L,node_rank_D ,node_rank_N ,node_rank_R,node_rank_G,node_rank_algec]
    return ken_list,rank_list,node_sort,node_rank
def frequency_rank(rank_list,name,n=0):
    if n==0:n = len(rank_list[0])
    rank_list = [l[0:n] for l in rank_list]
    m= ['o','+','^','x','s','D','o','^','x','s']
    edgecolors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
    colors = ['none','orange','none','red','none','none','none','none','olive','none']
    methods = ['CI','kshell','H-index','LID','DCL','NCVR','RCNN','GLSTM','GEE','GEE-F']
    plt.title(name)
    plt.figure(figsize=[8, 5])  # 设置画布
    plt.subplots_adjust(right=0.7,bottom=0.1)
    plt.xlabel('Rank')
    plt.ylabel('Frequency(ln)')
    for i in range(len(rank_list)):
        r = rank_list[i]
        x = list(set(rank_list[i]))
        y = [math.log(r.count(j)) for j in x]
        plt.scatter(x,y,marker=m[i],label =methods[i],s=20,alpha=0.8,c=colors[i],edgecolors=edgecolors[i])  # edgecolor
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=True, fontsize=10)
    plt.savefig('%s rank frequency_%s.png' % (name,n),dpi=300)
if __name__=='__main__':
    # 影响力label路径，网络数据路径，所有网络名称
    inpath  = '..\\dataset\\real\\'
    influence_path= '..\\dataset\\real_influence\\'
    datanames = set(os.listdir(inpath))
    net_names = []
    for i in datanames: net_names.append(i)
    net_names.sort()
    ken_table = []
    rank_record = []
    sort_record = []
    node_rank_record = []
    for x in range(0,34):
        name = net_names[x]  # 网络名
        print("net %s %s" % (x + 1, name))
        data = pd.read_csv(influence_path + '%s_gcc_Influence_P=1.5betac_Run1000.csv' % (name))  # 仿真得到的真实值
        data_memory = [list(data.loc[i]) for i in range(len(data))]
        G, pos = load_csv_net_gcc(name)
        G = nx.convert_node_labels_to_integers(G)
        ken_list,rank_list,node_sort,node_rank=calculate(G,data_memory)
        for x in data_memory: x[0] = int(x[0])
        simu_I = data_memory.copy()
        simu_I.sort(key=lambda x: x[1], reverse=True)
        simu_sort = [x[0] for x in simu_I]
        node_rank=[simu_sort]+node_rank
        # frequency_rank(rank_list,name,n)
        ken_table.append(ken_list)
        rank_record.append(rank_list)
        sort_record.append((node_sort))
        node_rank_record.append(node_rank)
    # with open('ken_table.pkl', 'wb') as f:
    #     pickle.dump(ken_table, f)
    # with open('rank_record.pkl', 'wb') as f:
    #     pickle.dump(rank_record, f)
    # with open('sort_record.pkl', 'wb') as f:
    #     pickle.dump(sort_record, f)
    # with open('node_rank_record.pkl', 'wb') as f:
    #     pickle.dump(node_rank_record, f)

    # with open('data.pkl', 'rb') as f:
    #     loaded_a = pickle.load(f)

