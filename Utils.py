import matplotlib.pyplot as plt
import networkx as nx
import copy
import torch
import torch.nn as nn
import random as rd
import numpy as np
import pandas as pd


def get_neigbors(g, node, depth):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output


def get_dgl_g_input(G0):
    G = copy.deepcopy(G0)
    input = torch.ones(len(G), 11)
    for i in G.nodes():
        input[i, 0] = G.degree()[i]
        input[i, 1] = sum([G.degree()[j] for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)
        input[i, 2] = sum([nx.clustering(G, j) for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)
        egonet = G.subgraph(list(G.neighbors(i)) + [i])
        input[i, 3] = len(egonet.edges())
        input[i, 4] = sum([G.degree()[j] for j in egonet.nodes()]) - 2 * input[i, 3]
    for l in [1, 2, 3]:
        for i in G.nodes():
            ball = get_neigbors(G, i, l)
            input[i, 5 + l - 1] = (G.degree()[i] - 1) * sum([G.degree()[j] - 1 for j in ball[l]])
    v = nx.voterank(G)
    votescore = dict()
    for i in list(G.nodes()): votescore[i] = 0
    for i in range(len(v)):
        votescore[v[i]] = len(G) - i
    e = nx.eigenvector_centrality(G, max_iter=1000)
    k = nx.core_number(G)
    for i in G.nodes():
        input[i, 8] = votescore[i]
        input[i, 9] = e[i]
        input[i, 10] = k[i]
    for i in range(len(input[0])):
        if max(input[:, i]) != 0:
            input[:, i] = input[:, i] / max(input[:, i])
    return input


def IC_simulation(p, g, set):
    g = copy.deepcopy(g)
    # pos = nx.spring_layout(g)
    if set == []:
        for i in list(g.nodes()):
            g.nodes[i]['state'] = 1 if rd.random() < .01 else 0
            if g.nodes[i]['state'] == 1:
                set.append(i)
    if set != []:
        for i in list(g.nodes()):
            if i in set:
                g.nodes[i]['state'] = 1
    for j in list(g.edges()):
        g.edges[j]['p'] = p  # rd.uniform(0,1)
    nextg = g.copy()
    terminal = 0
    # 仿真开始
    while (terminal == 0):
        for i in list(g.nodes()):
            if g.nodes[i]['state'] == 1:
                for j in g.neighbors(i):
                    if g.nodes[j]['state'] == 0:
                        nextg.nodes[j]['state'] = 1 if rd.random() < nextg.edges[i, j]['p'] else 0
                nextg.nodes[i]['state'] = 2
        g = nextg
        nextg = g
        terminal = 1
        for i in list(g.nodes()):
            if g.nodes[i]['state'] == 1:
                terminal = 0
    count = -len(set)
    for i in list(g.nodes()):
        if g.nodes[i]['state'] == 2:
            count += 1
    return count


def matrix_(G, L):
    # A=[]
    B = []
    labels = []
    Gu = []
    for node in list(G.nodes()):
        L_1_neigbors = []
        for depth in range(1, L):  # 1-7
            neighbors = get_neigbors(G, node, depth)
            neighbors_depth = neighbors[depth]
            if len(neighbors_depth) <= L - 1 - len(L_1_neigbors):
                L_1_neigbors = L_1_neigbors + neighbors_depth
            else:
                degree = [G.degree(x) for x in neighbors_depth]
                rank = sorted(range(len(degree)), key=lambda k: degree[k], reverse=True)
                neighbors_depth1 = [neighbors_depth[rank[i]] for i in range(len(rank))]
                L_1_neigbors = L_1_neigbors + neighbors_depth1[0:L - 1 - len(L_1_neigbors)]
            if len(L_1_neigbors) == L - 1:
                Gu.append([node] + L_1_neigbors)
                break
    for m in range(len(Gu)):
        u0 = list(G.nodes)[m]
        u = Gu[m]
        egonet = G.subgraph(Gu[m])
        Au = nx.adjacency_matrix(egonet)
        # A.append(Au)
        Bu = copy.deepcopy(Au)
        for i in range(len(u)):
            for j in range(len(u)):
                if i == 0 and j > 0:
                    Bu[i, j] = Au[i, j] * egonet.degree(u[j])
                if i > 0 and j == 0:
                    Bu[i, j] = Au[i, j] * egonet.degree(u[i])
                if i == j:
                    Bu[i, j] = egonet.degree(u[i])
                else:
                    Bu[i, j] = Au[i, j]
        Bu = Bu.toarray()
        Bu = torch.FloatTensor(Bu).reshape(1, 1, L, L)
        B.append(Bu)
    matrix = torch.concat(B)  # len(G)*1*len(egonet)*len(egonet)
    return matrix


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim=3, hidden_dim=128, dense_dim=32, target_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, dense_dim)
        self.l2 = nn.Linear(dense_dim, target_size)

    def forward(self, embedding):
        lstm_out, _ = self.lstm(embedding)
        l1_out = self.l1(lstm_out)
        l2_out = self.l2(l1_out)
        return l2_out


def embedding_(G):
    p = nx.degree_centrality(G)# degree centrality
    q = H_index(G)
    r = nx.core_number(G)
    p = list(p.values())
    q = list(q.values())
    r = list(r.values())
    p = [x/max(p) for x in p]
    q = [x/max(q) for x in q]
    r = [x/max(r) for x in r]
    fmat = [torch.Tensor([p[i],q[i],r[i]]).reshape(1,3) for i in range(len(G))]
    embedding = torch.concat(fmat)
    return embedding


def H_index(g, node_set=-1):
    if node_set == -1:
        nodes = list(g.nodes())
        d = dict()
        H = dict()  # H-index
        for node in nodes:
            d[node] = g.degree(node)
        for node in nodes:
            neighbors = list(g.neighbors(node))
            neighbors_d = [d[x] for x in neighbors]
            for y in range(len(neighbors_d)):
                if y > len([x for x in neighbors_d if x >= y]):
                    break
            H[node] = y - 1

    if node_set in list(g.nodes()):  # 计算节点node的H-index
        neighbors = list(g.neighbors(node))
        neighbors_d = [d[x] for x in neighbors]
        for y in range(len(neighbors_d)):
            if y > len([x for x in neighbors_d if x >= y]):
                break
        H = y - 1
    return H


def load_csv_net_gcc(name):
    path = "..\\dataset\\real\\%s\\" % name
    nodes_pos = pd.read_csv(path + "nodes_gcc.csv")
    edges = pd.read_csv(path + "edges_gcc.csv")
    pos = dict()
    for i in range(len(nodes_pos)):
        x = nodes_pos[' _pos'][i]
        x = x.strip('[]')
        x = x.split(",")
        x = list(map(float, x))
        x = np.array(x)
        # nodes_pos[' _pos'][i] = x
        pos[nodes_pos["# index"][i]] = x
    G = nx.Graph()
    G.add_nodes_from(list(nodes_pos['# index']))
    edge_list = [(edges['# source'][i], edges[' target'][i]) for i in range(len(edges))]
    G.add_edges_from(edge_list)
    G.remove_edges_from(nx.selfloop_edges(G))
    for i in list(G.nodes()):
        G.nodes[i]['state'] = 0
    return G, pos


def cal_betac(G):
    # 计算渗流阈值近似值
    d = [G.degree()[node] for node in list(G.nodes())]
    d2 = [x ** 2 for x in d]
    d_ = sum(d) / len(d)
    d2_ = sum(d2) / len(d2)
    betac = d_ / (d2_ - d_)
    return betac
    return G,pos
