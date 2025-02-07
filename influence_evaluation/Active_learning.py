import networkx as nx
from math import log
import numpy as np
from scipy.optimize import leastsq
import copy
def box_covering(G):
    print("diameter ",nx.diameter(G))
    lbmax = min(5,nx.diameter(G)+1)
    lb_list = [x for x in range(1,lbmax+1)]
    c = []
    for n in range(len(G)): c.append([-1]*lbmax)
    for lb in lb_list:
        used_c = set()
        c[0][lb - 1] = 0
        used_c.add(c[0][lb - 1])
        for i in range(1,len(G)):
            ban_c = set()
            for j in range(0,i):
                lij = nx.shortest_path_length(G,i,j)
                if lij>=lb:
                    ban_c.add(c[j][lb-1])
            avai_c = used_c-ban_c
            if avai_c==set():
                c[i][lb-1] = max(ban_c)+1
            else:
                c[i][lb - 1] = min(avai_c)
            used_c.add(c[i][lb - 1])
    return [x+1 for x in c[-1]]
def y_pre(p,x):
    f=np.poly1d(p)
    return f(x)
def error(p,x,y):
    return y-y_pre(p,x)
def least_square(x,y):
    p=np.random.rand(2)
    res = leastsq(error,p,args=(x,y))
    w1,w2 = res[0]
    # x_ = np.linspace(1, 10, 100)
    # y_p = w1 * x_ + w2
    # plt.scatter(x, y)
    # plt.plot(x_, y_p)
    # plt.show()
    # plt.close()
    return w1
def find_threshold_2_div(S,len_G):
    S_ = np.array(S)
    for i in range(len_G): S_[i][i] = 0
    S_f = S_.flatten()
    S_f =sorted(S_f,reverse=True)
    S_f = [x for x in S_f if x>0]
    m = len(S_f)
    while(True):
        InforG = nx.Graph()
        # k = max(int(0.5*(m))-1,0)
        k = max(int(0.5*(m)),0)
        for i in range(len_G):
            for j in range(i+1,len_G):
                if S[i][j]>=S_f[k]:
                    InforG.add_edge(i,j,weight = S[i][j])
        if len(InforG)<len_G:
            S_f = S_f[k+1:m]
            m=len(S_f)
        if len(InforG)==len_G:
            S_f = S_f[0:k+1]
            m=len(S_f)
            if len(S_f)==1:
                break
    return InforG

def al_model(G):
    node_list = list(G.nodes())
    len_G = len(node_list)
    Ns = box_covering(G)
    s = list(range(1,len(Ns)+1))
    # 最小二乘法求斜率即分形维度
    y = [log(yi) for yi in Ns]
    x = [log(xi) for xi in s]
    df = -least_square(x,y)
    # 2 计算本地维度dli和概率集pi
    dl = []
    for node in node_list:
        dij = list(nx.shortest_path_length(G,node).values())
        r_list = list(range(1,max(dij)+1))
        Nir=[]
        for r in r_list:
            Nr = len([x for x in dij if x<=r])
            Nir.append(Nr)
        y = [log(yi) for yi in Nir]
        x = [log(xi) for xi in r_list]
        dli = -least_square(x, y)
        dl.append(dli)
    P=[]
    for i in range(len(G)):
        node = node_list[i]
        pi=[]
        neighbor = list(nx.neighbors(G,node))
        Gi_node = neighbor+[node]
        dlGi = [dl[x] for x in Gi_node]
        degree_list = [G.degree(x) for x in Gi_node]
        m = max(degree_list)+1
        for k in range(m):
            di = G.degree(node)
            if k+1<=di+1:
                pik = dlGi[k]/sum(dlGi)
            else:
                pik = 0
            pi.append(pik)
        P.append(pi)

    PRE=[]
    for i in range(len(G)):PRE.append([-1]*len(G))
    for i in range(len(G)):
        for j in range(len(G)):
            m_ = min(G.degree(i),G.degree(j))+1
            P_i = sorted(P[i],reverse=True)
            P_j = sorted(P[j],reverse=True)
            PRE[i][j]=0
            for k in range(m_):
                PRE[i][j] += ( (P_i[k]/P_j[k])**df-(P_i[k]/P_j[k])   )/(1-df)
    R = []
    for i in range(len(G)): R.append([-1] * len(G))
    for i in range(len(G)):
        for j in range(len(G)):
            R[i][j] = PRE[i][j]+PRE[j][i]
    S = []
    for i in range(len(G)):
        S.append([-1] * len(G))
    Rmin = np.min(R)
    for i in range(len(G)):
        for j in range(len(G)):
            S[i][j] =1-R[i][j]/ Rmin
    InforG = find_threshold_2_div(S,len_G)
    Representative=[]
    InforG_=copy.deepcopy(InforG)
    for i in range(10):
        degree_dict = dict(InforG_.degree(weight="weight"))
        node = max(degree_dict,key=degree_dict.get) #当前度最大的节点
        Representative.append(node)
        neighbor = list(nx.neighbors(InforG_,node))
        InforG_.remove_nodes_from([node]+neighbor)
        if len(InforG_)==0:
            break
    return Representative
