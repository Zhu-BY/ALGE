import numpy as np
import pandas as pd
import os
from Utils import cal_betac,load_csv_net_gcc,IC_simulation
from ALGE_Greedy import ALGE_Greedy
from influence_evaluation.ALGE import ALGE_C
from influence_evaluation.other_algorithms import ci,k_shell,h_index,lid,dcl,ncvr,rcnn,glstm
from functools import partial
from multiprocessing import Pool

if __name__=="__main__":
    pool = Pool(16)
    inpath = '..\\dataset\\real'
    influence_path = '..\\dataset\\real_influence\\'
    datanames = set(os.listdir(inpath))
    net_names = []
    for i in datanames: net_names.append(i)
    net_names.sort()
    methods_name = ['CI', 'kshell', 'H-index', 'LID', 'DCL', 'NCVR', 'RCNN', 'GLSTM','ALGE-Greedy']
    methods = [ci,k_shell,h_index,lid,dcl,ncvr,rcnn,glstm]
    n = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for x in range(0,34):
        name = net_names[x]  # 网络名
        print("net %s %s" % (x + 1, name))
        G, pos = load_csv_net_gcc(name)
        data = pd.read_csv(influence_path + '%s_gcc_Influence_P=1.5betac_Run1000.csv' % (name))  # 仿真得到的真实值
        data_memory = [list(data.loc[i]) for i in range(len(data))]
        p = cal_betac(G) * 1.5
        count_m = []
        for j in range(9): # 9 methods
            count_n=[]
            for k in n:
                if j!=8:
                    _,_,sort_m,_ = methods[j](G,data_memory)
                    top_n_nodes = sort_m[0:k]
                if j==8: # ALGE_Greedy
                    _,_,_,_,pre_I=ALGE_C(G,data_memory)
                    top_n_nodes,_ = ALGE_Greedy(G,pre_I,k)  # GEE_IM_Greedy算法的结果
                top_n_seeds = top_n_nodes
                run = 1000
                count_t_run=[]
                count_t_run = pool.map(partial(IC_simulation, g=G,set=[top_n_seeds]), [p] * run)
                count_n.append(count_t_run)
            count_m.append(count_n)
        np.save('%s IMP simulation1000.npy'%name,count_m)
        print('Finish')

