import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()
        self.conv_layer = nn.Conv2d(1, out_channels, kernel_size=(1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)
    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(self.non_linearity (self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output
class SpecialSpmmFunctionFinal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]
        return b.to_dense()
    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            dge_sources = edge_sources.cpu()
            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None

class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)

class SpGraphAttentionLayer(nn.Module):

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim
        self.w_1 = nn.Parameter(torch.zeros(size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.w_1.data, gain=1.414)
        self.w_2 = nn.Parameter(torch.zeros(size=(  1, out_features)))
        nn.init.xavier_normal_(self.w_2.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()
        #前向传播的过程
    def forward(self, input, entity_list, edge_embed, entity_list_nhop, edge_embed_nhop):
        entity_list_tail = entity_list_nhop[0,:]
        entity_list_head = entity_list_nhop[1,:]
        N = input.size()[0]
        entity_list = torch.cat((entity_list[:, :], entity_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat((edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)
        edge_h = torch.cat((input[entity_list[0, :], :], input[entity_list[1, :], :], edge_embed[:, :]), dim=1).t()
        edge_m = self.w_1.mm(edge_h)
        powers = -self.leakyrelu(self.w_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        e_rowsum = self.special_spmm_final(entity_list, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12
        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)
        edge_e = self.dropout(edge_e)
        # edge_e: E
        edge_w = (edge_e * edge_m).t() # aijk * Cijk
        # edge_w: E * D
        h_prime = self.special_spmm_final(entity_list, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

