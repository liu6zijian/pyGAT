import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(self, in_c, out_c, hid_c):
        super(GCN, self).__init__()
        # self.W1 = nn.Parameter(torch.randn([in_c, hid_c]) )
        # torch.nn.init.xavier_uniform_(self.W1,gain=1.414)
        # self.W2 = nn.Parameter(torch.randn([hid_c, out_c]) )
        # torch.nn.init.xavier_uniform_(self.W2,gain=1.414)
        self.fc1 = nn.Linear(in_c, hid_c)
        self.fc2 = nn.Linear(hid_c, out_c)
    def forward(self, x, adj):
        # adj = GCN.adj_preprocess(adj)
        x = adj.mm(x)
        # x = F.relu(x.mm(self.W1))
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = adj.mm(x)
        # x = x.mm(self.W2)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    # @staticmethod
    # def adj_preprocess(adj):
    #     N = adj.size(0)
    #     matrix_I = torch.eye(N,dtype=adj.dtype,device=adj.device)
    #     D = adj.sum(dim=0).pow(-1)
    #     D[D == float("inf")] = 0.
    #     D = torch.diag(D)
    #     adj += matrix_I
    #     return D.mm(adj)



