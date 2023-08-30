import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from functools import partial
from multiprocessing import Pool
import pandas as pd
import os
from Utils import cal_betac,IC_simulation
def load_csv_net(name):
    path = ".\\dataset\\real\\%s\\"%name
    nodes_pos = pd.read_csv(path+"nodes.csv")
    edges = pd.read_csv(path + "edges.csv")
    pos = dict()
    for i in range(len(nodes_pos)):
        x = nodes_pos[' _pos'][i]
        x = x.strip('array([])')
        x = x.split(",")#根据‘，’来将字符串分割成单个元素
        x = list(map(float, x))#分离出来的单个元素也是字符串类型的，将其转成浮点
        x = np.array(x)
        # nodes_pos[' _pos'][i] = x
        pos[nodes_pos["# index"][i]] = x
    G= nx.Graph()
    edge_list = [(edges['# source'][i],edges[' target'][i]) for i in range(len(edges))]
    G.add_edges_from(edge_list)
    for i in list(G.nodes()):
        G.nodes[i]['state'] = 0
    return G,pos
if __name__=="__main__":
    nets_path = '.\\dataset\\real'
    influence_path = '.\\dataset\\real_influence\\'
    datanames = set(os.listdir(nets_path))
    net_names = []
    for i in datanames:
        net_names.append(i)
    net_names.sort()
    pool = Pool(24)
    for x in range(0,len(net_names)):
        name = net_names[x]
        print("net %s %s"%(x+1,name))
        G = load_csv_net(name)
        # G = load_ns_data()
        G= nx.convert_node_labels_to_integers(G)
        # g = dgl.from_networkx(G)
        p = cal_betac(G) * 1.5
        print(p)
        # node_features = get_dgl_g_input(G)
        data_size = len(G)
        run = 1000
        data_memory = [None] * data_size
        nodes_list = list(G.nodes())
        for iter in range(data_size):
            node = nodes_list[iter]
            simu_influences = []
            simu_influences = pool.map(partial(IC_simulation, g=G, set=[node]), [p] * run)
            influence = sum(simu_influences)/run
            data_memory[iter] = [node,influence]
            if iter%100==0:
                print(iter)
        data = pd.DataFrame()
        data['node'] = [x[0] for x in data_memory]
        data['node'].astype(pd.Int64Dtype())
        data['labels'] = [x[1] for x in data_memory]
        data.to_csv(influence_path+'%s_Influence_P=1.5betac_Run%s.csv'%(name,run),index=False)

    # GCC
    datanames = set(os.listdir(nets_path))
    net_names = []
    for i in datanames:
        net_names.append(i)
    net_names.sort()
    for x in range(0, len(net_names)):
        name = net_names[x]
        print("net %s %s" % (x + 1, name))
        G0,pos = load_csv_net(name)
        nx.set_node_attributes(G0, pos,'pos')
        G = nx.convert_node_labels_to_integers(G0)
        pos2 = dict()
        for node in list(G.nodes()):
            pos2[node]=G.nodes[node]['pos']
        nx.draw(G,pos2,node_size = 1,width = 0.05,node_color = 'r')
        plt.show();plt.close()
        data = pd.read_csv(influence_path + '%s_Influence_P=1.5betac_Run1000.csv' % (name))  # 仿真得到的真实值
        data_memory = [list(data.loc[i]) for i in range(len(data))]
        labels=dict()
        for node in list(G.nodes()):
            labels[node] = data['labels'][node]
        nx.set_node_attributes(G, labels, 'labels')
        largest = max(nx.connected_components(G),key=len)
        lcc= G.subgraph(largest)
        new_G = nx.convert_node_labels_to_integers(lcc)
        pos3 = dict()
        for node in list(new_G.nodes()):
            pos3[node]=new_G.nodes[node]['pos']
        nx.draw(new_G,pos2,node_size = 1,width = 0.05,node_color = 'r')
        plt.show();plt.close()
        edges = list(new_G.edges())
        nodes_pos =pd.DataFrame(columns=['# index',' _pos'])
        edges_csv =pd.DataFrame(columns=['# source',' target'])
        influence = pd.DataFrame(columns=['node','labels'])
        node_list = list(new_G.nodes())
        source = [edges[x][0] for x in range(len(edges))]
        target = [edges[x][1] for x in range(len(edges))]
        pos = [list(new_G.nodes[node]['pos']) for node in node_list]
        labels = [new_G.nodes[node]['labels'] for node in node_list]
        nodes_pos['# index']= node_list
        nodes_pos[" _pos"] = pos
        edges_csv['# source'] = source
        edges_csv[' target'] = target
        influence['node'] = node_list
        influence['labels']=labels
        net_path = nets_path+"\\%s\\"%name
        nodes_pos.to_csv(net_path+"nodes_gcc.csv",index=False)
        edges_csv.to_csv(net_path+"edges_gcc.csv",index=False)
        influence.to_csv(influence_path+'%s_gcc_Influence_P=1.5betac_Run1000.csv' % (name),index=False)

