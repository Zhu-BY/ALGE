import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class GATv3(nn.Module):
    def __init__(self, GATv3_P):
        super(GATv3, self).__init__()
        self.K = GATv3_P["num_layer"]
        if GATv3_P["activation"] == "relu":  self.AF = nn.ReLU()
        if GATv3_P["activation"] == "tanh":   self.AF = nn.Tanh()
        if GATv3_P["activation"] == "sigmod":  self.AF = nn.Sigmoid()
        if GATv3_P["activation"] == "elu":  self.AF = nn.ELU()
        self.in_dim,self.out_dim ,self.embed_dim, self.bias,self.dropout,self.heads = GATv3_P["in_dim"], GATv3_P["out_dim"], GATv3_P["embed_dim"], GATv3_P["bias"],GATv3_P["dropout"], GATv3_P['heads']
        self.layer = nn.ModuleList()
        self.layer.append(GATv2Conv(self.in_dim,32,num_heads = self.heads[0],bias = self.bias,activation = None,negative_slope=0.2,attn_drop=0.0))
        for i in range(1,self.K-1):
            self.layer.append(GATv2Conv(32*self.heads[i-1],32,num_heads = self.heads[i],bias = self.bias,activation = None,attn_drop= 0.0))
        self.layer.append(GATv2Conv(32*self.heads[-2],32, num_heads = self.heads[-1],bias = self.bias,activation = None,attn_drop= 0.0))
        self.dropout = nn.Dropout(self.dropout)

        self.linear_layer = nn.ModuleList()
        for i in range(0,self.K):
            self.linear_layer.append(nn.Linear(self.layer[i].fc_src.in_features,self.layer[i].fc_src.out_features))

    def forward(self, g, input_features):
        gs=[g.to(device)]*self.K
        h = input_features.to(device)
        for i,(layer,g) in enumerate(zip(self.layer,gs)):
            if i != self.K - 1:
                h = F.elu(layer(g,h).flatten(1)+self.linear_layer[i](h))
            else:
                h = layer(g,h).mean(1)+self.linear_layer[i](h)
        return h


class GATv3Net(nn.Module):
    def __init__(self,Net_P,GATv3_P):
        super(GATv3Net, self).__init__()
        self.fc1 = GATv3(GATv3_P).to(device)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.fc3.weight  = nn.init.normal_(self.fc3.weight,0.1,0.01)
        self.activation = Net_P["activation"]
    def forward(self, g,input_features):
        value = self.fc1(g,input_features)
        value = self.activation(value)
        value = self.fc2(value)
        value = self.activation(value)
        value = self.fc3(value)
        return value
    def reset_parameters(self):
        self.fc1.layer[0].reset_parameters()
        self.fc1.layer[1].reset_parameters()
        self.fc1.layer[2].reset_parameters()
        self.fc1.linear_layer[0].reset_parameters()
        self.fc1.linear_layer[1].reset_parameters()
        self.fc1.linear_layer[2].reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.weight  = nn.init.normal_(self.fc3.weight,0.1,0.01)
