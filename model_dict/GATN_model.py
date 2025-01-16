import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # 权重矩阵 W
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # 注意力权重向量 a
        # 均匀初始化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # 输入特征乘以权重矩阵 W，形状为 (N, out_features)
        bsz, N, _ = Wh.size()  # 节点数量

        # 计算注意力分数（e_ij）
        # Wh.repeat(1, N).view(N * N, -1) 每行在列方向重复 N 次，并展平成 N*N 行
        # Wh.repeat(N, 1)] 每行在行方向重复 N 次 获得 N*N 行
        # 实现每行与所有行之间的匹配矩阵
        a_input = torch.cat([Wh.repeat(1, 1, N).view(bsz, N * N, -1),
                             Wh.repeat(1, N, 1)], dim=2)  # 拼接 Wh_i || Wh_j
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))  # 形状为 (N * N,)

        # 将注意力分数映射回邻接矩阵的形状
        e = e.view(bsz, N, N)
        zero_vec = -9e15 * torch.ones_like(e)  # 生成一个与 e 形状相同的张量且值无穷小的 zero_vec
        attention = torch.where(adj > 0, e, zero_vec)  # 有连接的点保留注意力，其他的不保留
        attention = F.softmax(attention, dim=2)  # 对每个节点的邻居进行 Softmax 归一化
        attention = F.dropout(attention, self.dropout, training=self.training)  # Dropout

        # 聚合邻居特征
        h_prime = torch.matmul(attention, Wh)

        # 使用多头注意力时候需要输出激活后的特征
        if self.concat:
            return F.elu(h_prime)  # elu指数线性单元激活函数
        else:
            return h_prime

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GAT(nn.Module):
    def __init__(self, n_input, nclass, nheads, dropout=0.5, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义多头注意力机制
        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(n_input, n_input // 2, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer((n_input // 2) * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)  # Dropout 输入特征
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 多头注意力拼接
        x = F.dropout(x, self.dropout, training=self.training)  # Dropout 输出特征
        x = self.out_att(x, adj)  # 输出层
        x = x.mean(dim=1)
        return x


if __name__ == '__main__':
    N = 58  # 节点数量
    F_in = 20  # 输入特征维度
    F_out = 2  # 输出类别数
    n_heads = 3  # 注意力头数量
    bsz = 19

    # 输入特征和邻接矩阵
    x = torch.rand(bsz, N, F_in)
    adj = torch.randint(0, 2, (bsz, N, N))

    gat = GAT(n_input=F_in, nclass=F_out, dropout=0.5, alpha=0.2, nheads=n_heads)

    output = gat(adj, x)
    print(output.shape)
    print(output)
    print(torch.max(output, dim=1)[1])
