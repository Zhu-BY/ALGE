'''Find Influential Nodes Based on Graph Entropy and Embedding(GEE)
微调GNN、微调linear层的结果最好
'''
import networkx as nx
import random as rd
import copy
import pandas as pd
import os
import dgl
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings
from influence_evaluation.Model import GATv3Net
from Utils import get_dgl_g_input
from influence_evaluation.Active_learning import al_model
warnings.filterwarnings('ignore')

def ALGE_C(G,data_memory):
    model = torch.load('..\\influence_evaluation\\ALGE_B.pth')
    gatv3_ALGE_sample_sets = al_model(G)
    g = dgl.from_networkx(G)
    node_features_ = get_dgl_g_input(G)
    node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)
    for d in data_memory: d[0] = int(d[0])
    simu_I = data_memory.copy()
    simu_I.sort(key=lambda x: x[1], reverse=True)
    model.train()
    train_nodes = gatv3_ALGE_sample_sets
    ken_ALGE_C, all_ls_F, test_ls_F,test_ls,rank_C,model,pre_sort2,node_rank_C,pre_I_with_node = train(train_nodes, model, data_memory, g, node_features,G)
    return ken_ALGE_C,rank_C,pre_sort2,node_rank_C,pre_I_with_node,train_nodes

def calculate_ken(G,pre_gat,data_memory,train_nodes=[],n=0):
    nodes_list = list(G.nodes())
    prediction_I = pre_gat.detach().numpy()
    prediction_I_with_node = [[nodes_list[i], prediction_I[i][0]] for i in range(len(prediction_I))]
    prediction_I_with_node.sort(key=lambda x: x[1], reverse=True)
    simu_I_with_node = data_memory.copy()
    simu_I_with_node.sort(key=lambda x: x[1], reverse=True)
    simu_sort = [x[0] for x in simu_I_with_node]  # 从大到小排序的节点
    pre_sort = [x[0] for x in prediction_I_with_node if x[0] in simu_sort]  # 从大到小排序的节点
    for node in train_nodes: # test set Ken
        simu_sort.remove(node)
        pre_sort.remove(node)
    # draw_rank(simu_sort, pre_sort, 'Model%s_Prediction%s'%(m,i))
    node_rank_simu = list(range(1, len(simu_sort) + 1))
    node_rank_pre = [pre_sort.index(x) if x in pre_sort else len(pre_sort) for x in simu_sort]  # 按仿真排序的节点，在预测节点中的对应rank
    if n==0:n = len(node_rank_simu)
    ken_pre = kendalltau(node_rank_simu[0:n], node_rank_pre[0:n])
    return ken_pre[0],pre_sort,node_rank_pre
def rmse(value,train_nodes, labels):
    loss = nn.MSELoss()
    labels = torch.tensor(labels).reshape(-1, 1)
    y = torch.cat([value[node] for node in train_nodes]).reshape(-1,1)
    rmse = torch.sqrt(loss(y,labels))
    return rmse.item()
def train(train_nodes, model, data_memory, g, node_features,G):
    data_train = [x for x in data_memory if x[0] in train_nodes]  # 训练集
    data_test = [x for x in data_memory if x[0] not in train_nodes]  # 测试
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    loss = nn.MSELoss()
    train_ls, test_ls, train_logls, test_logls = [], [], [], []  # 训练损失随训练次数曲线
    # 微调前测试集loss
    model.eval()
    value = model(g, node_features)
    test_loss = rmse(value, [x[0] for x in data_test], [x[1] for x in data_test])
    test_ls.append(test_loss)
    model.train()
    # 开始微调
    for epoch in range(150):
        nodes = [x[0] for x in data_train]
        labels = [x[1] for x in data_train]
        value = model(g, node_features)
        y = torch.cat([value[node].unsqueeze(1) for node in nodes], 0)
        train_labels = torch.tensor(labels).reshape(-1, 1)
        l = loss(torch.log(y), torch.log(train_labels))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        value = model(g, node_features)
        train_ls.append(rmse(value, nodes, labels))
        model.eval()
        value = model(g, node_features)
        test_ls.append(rmse(value, [x[0] for x in data_test], [x[1] for x in data_test]))
        model.train()
    plt.figure()
    plt.title('%s'%len(G))
    plt.plot(list(range(len(test_ls))),test_ls)
    plt.show()
    plt.close()
    pre_gat = model(g, node_features)
    model.eval()
    value = model(g, node_features)
    nodes_list = list(G.nodes())
    prediction_I = value.detach().numpy()
    prediction_I_with_node = [[nodes_list[i], prediction_I[i][0]] for i in range(len(prediction_I))]
    value_ = value.flatten().detach().numpy()
    value_ = sorted(value_, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]<value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    ken_pre2,pre_sort,node_rank_pre = calculate_ken(G, pre_gat, data_memory,train_nodes)
    all_data_loss2 = rmse(pre_gat, [x[0] for x in data_memory], [x[1] for x in data_memory])
    value_ = value.flatten().detach().numpy()
    value_ = sorted(value_, reverse=True)
    rank=[1]
    for i in range(1,len(value_)):
        if value_[i]!=value_[i-1]:
            rank.append(i+1)
        elif value_[i]==value_[i-1]:
            rank.append(rank[-1])
    return ken_pre2, all_data_loss2, test_ls[-1],test_loss,rank,model,pre_sort,node_rank_pre,prediction_I_with_node

def load_train_net(j):
    type = 'Ba'
    Edge = pd.read_csv('..\\dataset\\synthetic\\'+'SyntheticNet%s%s' % (type,j))
    u = list(Edge['u'])
    v = list(Edge['v'])
    edge_list = [(u[i], v[i]) for i in range(len(v))]
    G = nx.Graph()
    G.add_edges_from(edge_list)
    for i in list(G.nodes()):
        G.nodes[i]['state'] = 0
    import os
    # 指定目录和文件前缀
    directory = '..\\dataset\\synthetic'
    prefix = "SyntheticNet%s%sInfluence_p" % (type,j)
    # 遍历目录中的文件
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            filepath = os.path.join(directory, filename)
            break
    data = pd.read_csv(filepath)
    data_memory = [list(data.loc[i]) for i in range(len(data))]
    return G,data_memory

if __name__=='__main__':
    num_heads = 8
    num_layer = 3
    num_out_heads = 1
    heads = ([num_heads] * (num_layer - 1)) + [num_out_heads]
    gat_para = dict([["in_dim", 10], ["out_dim", 32], ["embed_dim", 32], ["heads", heads], ["num_layer", num_layer],
                     ["activation", "elu"], ["bias", True], ["dropout", 0.1]])
    gatnet_para = dict([["out_dim", 1], ["hidden_dim", 32], ["activation", nn.ReLU()]])
    gatv2 = GATv3Net(gatnet_para, gat_para)

    # Edge = pd.read_csv('..\\dataset\\synthetic\\train_1000_4.csv')
    # u = list(Edge['u'])
    # v = list(Edge['v'])
    # edge_list = [(u[i], v[i]) for i in range(len(v))]
    # G = nx.Graph()
    # G.add_edges_from(edge_list)
    # for i in list(G.nodes()):
    #     G.nodes[i]['state'] = 0
    # G = nx.convert_node_labels_to_integers(G)
    # data = pd.read_csv("..\\dataset\\synthetic\\train_1000_4_Influence.csv")
    # data_memory = [list(data.loc[i]) for i in range(len(data))]
    # for x in data_memory: x[0] = int(x[0])
    # g = dgl.from_networkx(G)
    # node_features_ = get_dgl_g_input(G)
    # node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)

    G_list = []
    data_memory_list = []
    g_list = []
    node_features_list = []
    for j in range(50):
        G,data_memory = load_train_net(j)
        G_list.append(G)
        for x in data_memory: x[0] = int(x[0])
        data_memory_list.append(data_memory)
        g = dgl.from_networkx(G)
        g_list.append(g)
        node_features_ = get_dgl_g_input(G)
        node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)
        node_features_list.append(node_features)
    num_epochs = 500
    lr = 0.001
    model = gatv2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    train_logls = []
    train_ls = []
    for epoch in range(num_epochs):
        j = rd.randint(0,49)
        G = G_list[j];data_memory=data_memory_list[j];g=g_list[j];node_features=node_features_list[j]
        nodes_list = list(G.nodes())
        nodes_list = [x[0] for x in data_memory]
        train_nodes = nodes_list
        data_train = [x for x in data_memory if x[0] in train_nodes]
        nodes = [x[0] for x in data_train]
        labels = [x[1] for x in data_train]
        value = model(g,node_features)
        y = torch.cat([value[node].unsqueeze(1) for node in nodes], 0)
        train_labels = torch.tensor(labels).reshape(-1,1)
        l=loss(torch.log(y),torch.log(train_labels))
        if l!=l:
            model.fc1.layer[0].reset_parameters()
            model.fc1.layer[1].reset_parameters()
            model.fc1.layer[2].reset_parameters()
            model.fc1.linear_layer[0].reset_parameters()
            model.fc1.linear_layer[1].reset_parameters()
            model.fc1.linear_layer[2].reset_parameters()
            model.fc2.reset_parameters()
            model.fc3.reset_parameters()
            continue
        train_ls.append(l.detach().numpy())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print('epoch:%s, ' % epoch, 'train_ls:%s, ' % train_ls[-1])
    plt.plot(list(range(len(train_ls))),train_ls)
    plt.show()
    torch.save(model,'ALGE_B.pth')