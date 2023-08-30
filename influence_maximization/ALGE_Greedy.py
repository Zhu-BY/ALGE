from Utils import get_neigbors
import random as rd
import math
def influenced_nodes(G,node,I):
    I = math.ceil(I)
    influenced_nodes=[]
    o = get_neigbors(G,node,len(G))
    counto=[]
    for j in range(len(G)):
        if len(o[j+1])==0:
            break
        counto.append(sum([len(o[i+1]) for i in range(j+1)]))
    for i in range(0,len(counto)):
        if counto[i]<I:
            influenced_nodes += o[i+1]
        elif counto[i] == I:
           influenced_nodes += o[i + 1]
           break
        elif counto[i]>I and i==0:
            influenced_nodes = rd.sample(o[i+1],I)
            break
        elif counto[i]>I and i>0:
            influenced_nodes += rd.sample(o[i + 1], I-counto[i-1])
            break
    influenced_nodes.append(node)
    return influenced_nodes

def ALGE_Greedy(G,pre_I,k=10):
    nodes_list = list(G.nodes())
    node_influence_nodes = [influenced_nodes(G, pre_I[i][0], pre_I[i][1]) for i in range(len(pre_I))]
    seedset = []
    now_final_s = []
    for i in range(k):
        nodes_add_influece = []
        for vertex in nodes_list:
            if vertex not in seedset:
                vertex_inf_nodes = node_influence_nodes[vertex]
                temp_final_s = list(set(now_final_s+vertex_inf_nodes))
                add_inf = len(temp_final_s)-len(now_final_s)
                nodes_add_influece.append(add_inf)
            else:
                nodes_add_influece.append(-1)
        now_node = nodes_list[nodes_add_influece.index(max(nodes_add_influece))]
        seedset.append(now_node)
        now_final_s = list(set(now_final_s+node_influence_nodes[now_node]))
    return seedset,now_final_s

