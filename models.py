import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class initial_H(nn.Module):
    def __init__(self, Hd, in_dim):
        super().__init__()

        self.d_dim = Hd['drug_F'].shape[1]
        self.t_dim = Hd['target_F'].shape[1]
        self.fc1 = nn.Linear(self.d_dim, in_dim, bias=False)
        self.fc2 = nn.Linear(self.t_dim, in_dim, bias=False)

    def forward(self, Hd):
        HD = Hd['drug_F'].cuda()
        HT = Hd['target_F'].cuda()
        HD = F.normalize(HD, dim=1)
        HT = F.normalize(HT, dim=1)
        new_HD = self.fc1(HD)
        new_HT = self.fc2(HT)
        H = torch.cat([new_HD, new_HT], dim=0)

        return H


class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feature))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output + self


class Decoder(nn.Module):
    def __init__(self, train_W):
        super().__init__()
        self.train_W = nn.Parameter(train_W)

    def forward(self, H, drug_num, target_num):
        HR = H[0:drug_num]
        HD = H[drug_num:(drug_num + target_num)]
        supp1 = torch.mm(HR, self.train_W)
        decoder = torch.mm(supp1, HD.transpose(0, 1))
        return decoder


class GCN_decoder(nn.Module):
    def __init__(self, Hd, in_dim, hgcn_dim, train_W, dropout):
        super().__init__()

        self.I_H = initial_H(Hd, in_dim)
        self.gc1 = GraphConvolution(in_dim, hgcn_dim)
        self.gc3 = GraphConvolution(hgcn_dim, hgcn_dim)
        self.decoder = Decoder(train_W)
        self.dropout = dropout

    def forward(self, Hd, G, drug_num, target_num):
        ini_H = self.I_H(Hd)  # get initial features
        H = self.gc1(ini_H, G)  # GCN encoder
        H = F.relu(H)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc3(H, G)
        H = F.relu(H)
        decoder = self.decoder(H, drug_num, target_num)  # Bilinear decoder

        return decoder, H
