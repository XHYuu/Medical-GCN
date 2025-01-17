import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

warnings.filterwarnings("ignore")


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        batch_size = input_feature.size(0)
        support = torch.bmm(input_feature, self.weight.unsqueeze(0).expand(batch_size, -1, -1))
        output = torch.bmm(adjacency, support)
        if self.use_bias:
            output += self.bias.unsqueeze(0).unsqueeze(0).expand_as(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    def __init__(self, n_input):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(n_input, 16)
        self.gcn2 = GraphConvolution(16, 2)

    def forward(self, adjacency, feature):
        h = F.leaky_relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)  # [batch_size, feature_number, num_classes]
        logits = logits.mean(dim=1)  # 对节点特征求平均，得到 [batch_size, num_classes]
        return logits
