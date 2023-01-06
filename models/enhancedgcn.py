import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.linear_1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        # output = torch.sparse.mm(adj, input)
        x = torch.spmm(adj, input)
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class EnhancedGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, n_layers=5, dropout=0.5, input_dropout=0.0, norm=True,
                 input_layer=True, output_layer=True):

        super(EnhancedGCN, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.norm = norm
        if input_layer:
            self.input_linear = nn.Linear(in_features=nfeat, out_features=nhid)
            self.input_drop = nn.Dropout(input_dropout)
        if output_layer:
            self.output_linear = nn.Linear(in_features=nhid, out_features=nclass)
            self.output_normalization = nn.LayerNorm(nhid)
        self.convs = nn.ModuleList()
        if norm:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = nfeat
            else:
                in_hidden = nhid
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = nclass
            else:
                out_hidden = nhid

            self.convs.append(GraphConvolution(in_hidden, out_hidden, dropout=dropout))
            if norm:
                self.norms.append(nn.LayerNorm(in_hidden))

    def forward(self, x, adj):
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = F.gelu(x)

        for i, layer in enumerate(self.convs):
            if self.norm:
                x_res = self.norms[i](x)
                x_res = layer(x_res, adj)
                x = x + x_res
            else:
                x = layer(x,adj)
            if i == self.n_layers - 1:
                mid = x

        if self.output_layer:
            x = self.output_normalization(x)
            x = self.output_linear(x).squeeze(1)
        return mid, x