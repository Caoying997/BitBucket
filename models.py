import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SpGraphAttentionLayer, ConvKB


class SpGAT(nn.Module):
    def __init__(self, num_nodes, entity_input_dim, entity_output_dim, relation_input_dim, dropout, alpha, nheads):
        super(SpGAT, self).__init__()

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes,
                                                 entity_input_dim,
                                                 entity_output_dim,
                                                 relation_input_dim,
                                                 dropout = dropout, alpha = alpha, concat = True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.W = nn.Parameter(torch.zeros(size=(relation_input_dim, nheads * entity_output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.out_att = SpGraphAttentionLayer(num_nodes,
                                             entity_output_dim * nheads,
                                             nheads * entity_output_dim,
                                             nheads * entity_output_dim,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False  )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embeddings, entity_list, relation_type, entity_list_nhop, relation_type_nhop):
        relation_embed_nhop = relation_embeddings[relation_type_nhop[:, 0]] + relation_embeddings[relation_type_nhop[:, 1]] +\
                              entity_embeddings[entity_list_nhop[0,:]] + entity_embeddings[entity_list_nhop[1,:]]

        edge_embed = relation_embeddings[relation_type]  +  entity_embeddings[entity_list[0,:]] + entity_embeddings[entity_list[1, :]]
        out_entity_1 = torch.cat([att(entity_embeddings, entity_list, edge_embed, entity_list_nhop, relation_embed_nhop)
                       for att in self.attentions], dim=1)
        out_entity_1 = self.dropout_layer(out_entity_1)
        out_relation_1 = relation_embeddings.mm(self.W)
        edge_embed = out_relation_1[relation_type]
        relation_embed_nhop = out_relation_1[relation_type_nhop[:, 0]] + out_relation_1[relation_type_nhop[:, 1]] +\
                              out_entity_1[entity_list_nhop[0,:]]+out_entity_1[entity_list_nhop[1,:]]
        out_entity_1 = F.elu(self.out_att(out_entity_1, entity_list, edge_embed, entity_list_nhop, relation_embed_nhop))
        return out_entity_1, out_relation_1
class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,drop_GAT, alpha, nheads_GAT):
        super().__init__()
        self.num_nodes     = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.entity_out_dim_2 = entity_out_dim[1]

        self.num_relation = initial_relation_emb.shape[0]
        self.relation_in_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]
        self.relation_out_dim_2 = relation_out_dim[1]

        self.drop_GAT = drop_GAT
        self.nheads_GAT_2 = nheads_GAT[1]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))
        #可以将num_relation 换成 num_nodes
        self.final_relation_embeddings = nn.Parameter(torch.randn(self.num_relation, self.relation_out_dim_1 * self.nheads_GAT_1))
        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)
        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_in_dim, self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.entity_out_dim_2 )))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):

        edge_list = adj[0]
        edge_type = adj[1]
        edge_list_tail= edge_list[0]
        edge_list_head =  edge_list[1]
        edge_list_nhop = torch.cat((train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat([train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        self.entity_embeddings.data = F.normalize( self.entity_embeddings.data, p=2, dim=1).detach()
        self.relation_embeddings.data = F.normalize(self.relation_embeddings.data, p=2, dim=1)
        out_entity_1, out_relation_1 = self.sparse_gat_1(Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings, edge_list, edge_type,  edge_list_nhop, edge_type_nhop)
        mask_indices = torch.unique(batch_inputs[:, 2]).cpu()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cpu()
        mask[mask_indices] = 1.0
        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1
        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        mask = torch.zeros(self.relation_embeddings.shape[0]).cpu()
        relation_upgraded = self.relation_embeddings.mm(self.W_entities)
        out_relation_1 = relation_upgraded + mask.unsqueeze(-1).expand_as( out_relation_1) * out_relation_1
        out_relation_1 = F.normalize(out_relation_1, p=2, dim=1)
        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data
        return out_entity_1, out_relation_1

class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        super().__init__()
        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.drop_GAT = drop_GAT
        self.nheads_GAT_1 = nheads_GAT[0]
        self.nheads_GAT_2 = nheads_GAT[1]
        # 数据集中关系的信息
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]
        self.drop_conv = drop_conv
        self.alpha = alpha
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels
        self.final_entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))
        self.final_relation_embeddings = nn.Parameter(torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))
        self.ConvKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
                                self.final_relation_embeddings[ batch_inputs[:, 1]].unsqueeze(1),
                                self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.ConvKB(conv_input)
        return out_conv
    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
                                self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
                                self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.ConvKB(conv_input)
        return out_conv

