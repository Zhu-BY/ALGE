from Utils import get_neigbors,matrix_,H_index,embedding_,LSTMModel
import math
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import torch
import torch.nn as nn

def h_index(G,data_memory,train_nodes=[]):
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]

    H = H_index(G)
    h = [[x, H[x]] for x in H]
    h.sort(key=lambda x: x[1], reverse=True)
    h_sort = [x[0] for x in h if x[0] in simu_sort]

    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        h_sort.remove(node)
    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_H = [h_sort.index(x) if x in h_sort else len(h_sort) for x in
                   simu_sort]
    ken_H = kendalltau(node_rank_simu, node_rank_H)
    H = list(H.values())
    value_ = sorted(H, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]<value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    # print(ken_H[0])
    return ken_H[0],rank,h_sort,node_rank_H

def k_shell(G, data_memory,train_nodes=[]):
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]


    K = nx.core_number(G)
    k = [[x, K[x]] for x in K]
    k.sort(key=lambda x: x[1], reverse=True)
    k_sort = [x[0] for x in k if x[0] in simu_sort]

    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        k_sort.remove(node)

    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_K = [k_sort.index(x) if x in k_sort else len(k_sort) for x in
                   simu_sort]
    ken_K = kendalltau(node_rank_simu, node_rank_K)
    K = list(K.values())
    # #计算肯德尔系数
    value_ = sorted(K, reverse=True)
    rank = [1]
    for i in range(1, len(value_)):
        if value_[i] < value_[i - 1]:
            rank.append(i + 1)
        elif value_[i] == value_[i - 1]:
            rank.append(rank[-1])
    # print(ken_K[0])
    return ken_K[0], rank,k_sort,node_rank_K


def ci(G,data_memory,train_nodes=[]):
    l=3
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]
    CI = dict()
    for i in G.nodes():
        ball = get_neigbors(G,i,l)
        CI[i] = (G.degree()[i]-1)*sum([G.degree()[j]-1 for j in ball[l]])
    ci = [[x, CI[x]] for x in CI]
    ci.sort(key=lambda x: x[1], reverse=True)
    ci_sort = [x[0] for x in ci if x[0] in simu_sort]

    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        ci_sort.remove(node)
    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_CI = [ci_sort.index(x) if x in ci_sort else len(ci_sort) for x in
                   simu_sort]
    ken_CI = kendalltau(node_rank_simu, node_rank_CI)
    CI = list(CI.values())
    # #计算肯德尔系数
    value_ = sorted(CI, reverse=True)
    rank = [1]
    for i in range(1, len(value_)):
        if value_[i] < value_[i - 1]:
            rank.append(i + 1)
        elif value_[i] == value_[i - 1]:
            rank.append(rank[-1])
    # print(ken_CI[0])
    return ken_CI[0],rank,ci_sort,node_rank_CI

def lid(G,data_memory,train_nodes=[]):
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]

    d_m = np.zeros((len(G), len(G)))  # distance matrix
    d_m = np.zeros((len(G), len(G)))  # distance matrix
    node_list = list(G.nodes())
    for i in range(len(G)):
        v = node_list[i]
        for u in node_list:
            d_m[v][u] = nx.shortest_path_length(G, v, u)
    DI = []
    DI2 = []
    N = len(G)
    for node in node_list:
        rmax = math.ceil(0.5 * max(d_m[node]))
        ls = list(range(1, rmax + 1))
        ni_l = []
        Ni_l = []
        for l in ls:
            nil = len([x for x in d_m[node] if x <= l])
            ni_l.append(nil)
            Nil = len([x for x in d_m[node] if x == l])
            Ni_l.append(Nil)
        Ii_l = [-(nil / N) / math.log(nil / N) for nil in ni_l]
        y = [yi for yi in Ii_l]
        x = [math.log(xi) for xi in ls]
        # DIi = -least_square(x, y)
        # DI.append(DIi)
        DI2i = [ls[i] / (1 + math.log(ni_l[i] / N)) * Ni_l[i] / N for i in range(len(ls))]
        DI2.append(DI2i)
    s = sorted(range(len(DI2)), key=lambda k: DI2[k][0], reverse=False)
    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        s.remove(node)

    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_s = [s.index(x) if x in s else len(s) for x in
                   simu_sort]
    ken_LID = kendalltau(node_rank_simu, node_rank_s)
    # print(ken_LID[0])
    LID = [DI2[k][0] for k in range(len(DI2))]
    value_ = sorted(LID, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]<value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    return ken_LID[0],rank,s,node_rank_s

def dcl(G,data_memory,train_nodes=[]):
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]

    DCL = []
    for node in list(G.nodes()):
        ki = G.degree(node)
        CCi = nx.clustering(G, node)
        neighbors = list(nx.neighbors(G, node))
        n_k = sum([G.degree(x) for x in neighbors])
        egonet = G.subgraph(neighbors + [node])
        Ei = len(egonet.edges()) - ki
        DCLi = ki * 1 / (CCi + 1 / ki) + n_k / (Ei + 1)
        DCL.append(DCLi)
    s = sorted(range(len(DCL)), key=lambda k: DCL[k], reverse=True)
    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        s.remove(node)

    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_s = [s.index(x) if x in s else len(s) for x in
                   simu_sort]
    ken_DCL = kendalltau(node_rank_simu, node_rank_s)
    # print(ken_DCL[0])
    value_ = sorted(DCL, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]<value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    return ken_DCL[0],rank,s,node_rank_s

def ncvr(G,data_memory,train_nodes=[]):
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]  # 从大到小排序的节点

    seeds = []
    S_=[]
    S = dict()
    Va = dict()
    NC = dict()
    coreness = nx.core_number(G)
    for node in list(G.nodes()):
        NC[node] = (coreness[node] - min(coreness.values())) / (max(coreness.values()) - min(coreness.values()))
    for node in list(G.nodes()):
        S[node] = 0
        Va[node] = 1
    k = sum(dict(G.degree()).values()) / len(G)
    theta = 0.5
    j=0
    while j<len(G):
        for node in list(G.nodes()):
            neighbor = list(nx.neighbors(G, node))
            S[node] = sum([Va[i] * NC[i] * (1 - theta) + Va[i] * theta for i in neighbor])
        # 投票分数最大的节点：
        for node in seeds: S[node] = -999
        seed = max(zip(S.values(), S.keys()))
        for x in list(S.keys()):
            if S[x]==seed[0]:
                S_.append(S[x])
                seeds = seeds + [x]
                j+=1
                Va[x] = 0
                neighbor2 = get_neigbors(G, x, 2)
                for d in [1, 2]:
                    for node in neighbor2[d]:
                        delta = 1 / (k * d)
                        Va[node] = max(Va[node] - delta, 0)
    s = seeds
    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        s.remove(node)
    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_s = [s.index(x) if x in s else len(s) for x in
                   simu_sort]
    ken_NC = kendalltau(node_rank_simu, node_rank_s)
    value_ = sorted(S_, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]!=value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    # print(ken_NC[0])
    return ken_NC[0],rank,s,node_rank_s

def rcnn(G,data_memory,train_nodes=[]):
    L=28
    model = torch.load('..\\influence_evaluation\\RCNN.pth')  # 载入迁移学习模型
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]  # 从大到小排序的节点

    matrix = matrix_(G,L=L)
    model.eval()
    value = model(matrix)
    nodes_list = list(G.nodes())
    prediction_I = value.detach().numpy()
    prediction_I_with_node = [[nodes_list[i], prediction_I[i][0]] for i in range(len(prediction_I))]
    prediction_I_with_node.sort(key=lambda x: x[1], reverse=True)
    pre_sort = [x[0] for x in prediction_I_with_node if x[0] in simu_sort]  # 从大到小排序的节点

    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        pre_sort.remove(node)

    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_RCNN = [pre_sort.index(x) if x in pre_sort else len(pre_sort) for x in simu_sort]  # 按仿真排序的节点，在预测节点中的对应rank
    ken_RCNN = kendalltau(node_rank_simu, node_rank_RCNN)
    # print(ken_RCNN[0])
    value_ = value.flatten().detach().numpy()
    value_ = sorted(value_, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]<value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    return ken_RCNN[0],rank,pre_sort,node_rank_RCNN

def glstm(G,data_memory,train_nodes=[]):
    model = torch.load('..\\influence_evaluation\\GLSTM.pth')  # 载入迁移学习模型
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]  # 从大到小排序的节点
    # node_rank_simu = list(range(1, len(simu_sort) + 1))
    embedding = embedding_(G)
    model.eval()
    value = model(embedding)
    nodes_list = list(G.nodes())
    prediction_I = value.detach().numpy()
    prediction_I_with_node = [[nodes_list[i], prediction_I[i][0]] for i in range(len(prediction_I))]
    prediction_I_with_node.sort(key=lambda x: x[1], reverse=True)
    pre_sort = [x[0] for x in prediction_I_with_node if x[0] in simu_sort]  # 从大到小排序的节点

    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        pre_sort.remove(node)

    node_rank_simu = list(range(1, len(simu_sort) + 1))

    node_rank_GLSTM = [pre_sort.index(x) if x in pre_sort else len(pre_sort) for x in
                       simu_sort]
    ken_GLSTM = kendalltau(node_rank_simu, node_rank_GLSTM)
    # print(ken_GLSTM[0])
    value_ = value.flatten().detach().numpy()
    value_ = sorted(value_, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]<value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    return ken_GLSTM[0],rank,pre_sort,node_rank_GLSTM
if __name__=='__main__':
    Edge = pd.read_csv('..\\dataset\\synthetic\\train_1000_4.csv')
    data = pd.read_csv("..\\dataset\\synthetic\\train_1000_4_Influence.csv")
    u = list(Edge['u'])
    v = list(Edge['v'])
    edge_list = [(u[i], v[i]) for i in range(len(v))]
    G = nx.Graph()
    G.add_edges_from(edge_list)
    for i in list(G.nodes()):
        G.nodes[i]['state'] = 0
    G = nx.convert_node_labels_to_integers(G)
    # print(cal_influence(G))
    data_memory = [list(data.loc[i]) for i in range(len(data))]
    for x in data_memory: x[0] = int(x[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I]  # 从大到小排序的节点
    node_rank_simu = list(range(1, len(simu_sort) + 1))
    # RCNN-Train
    L = 28
    matrix = matrix_(G, L)
    # 2conv(5*5,1,16,1,2;5*5,16,32,1,2),2pool(2*2,max),1full(32*L/4*L/4*1),ReLU,MSE
    net = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * int(L / 4) * int(L / 4), 1))

    loss = nn.MSELoss()
    model = net
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器
    train_ls = []
    for epoch in range(300):
        data_train = data_memory
        nodes = [x[0] for x in data_train]
        labels = [x[1] for x in data_train]
        value = model(matrix)
        y = torch.cat([value[node].unsqueeze(1) for node in nodes], 0)
        train_labels = torch.tensor(labels).reshape(-1, 1)
        l = loss(y, train_labels)
        train_ls.append(l.detach().numpy())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print('epoch:%s, ' % epoch, 'train_ls:%s, ' % train_ls[-1])

    torch.save(model, 'RCNN.pth')

    # GLSTM-Train
    embedding=embedding_(G)
    # LSTM结构 LSTM 128,DENSE 32,REGRESSOR 1
    loss = nn.L1Loss()
    model=LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # 优化器
    train_ls = []
    for epoch in range(1000):
        data_train=data_memory
        nodes = [x[0] for x in data_train]
        # features = torch.cat([node_features[i] for i in sample_index],dim=1)
        labels = [x[1] for x in data_train]
        value = model(embedding)
        # y = torch.cat([value[node] for node in nodes]).reshape(-1,1)
        # y = torch.cat([(torch.clamp(value[node], 1, float('inf'))).unsqueeze(1) for node in nodes], 0)
        y = torch.cat([value[node].unsqueeze(1) for node in nodes], 0)
        train_labels = torch.tensor(labels).reshape(-1, 1)
        l = loss(y, train_labels)
        train_ls.append(l.detach().numpy())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print('epoch:%s, ' % epoch, 'train_ls:%s, ' % train_ls[-1])
    torch.save(model,'GLSTM.pth')