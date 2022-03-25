import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models import SpKBGATModified, SpKBGATConvOnly
from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
import random
import argparse
import os
import time
import pickle
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-g', '--gamma', default=12.0, type=float)
    args.add_argument("-data", "--data",default="./data/kinship/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,default=200, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int, default=260, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,default=0.00001, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,default=0.000001, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=True)
    args.add_argument("-outfolder", "--output_folder",default="./checkpoints/kinship/out/", help="Folder name to save the models.")
    args.add_argument("-b_gat", "--batch_size_gat", type=int, default=21272115, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int, default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,  default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-entity_out_dim", "--entity_out_dim", type=int, nargs='+',default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-relation_out_dim", "--relation_out_dim", type=int, nargs='+', default=[100, 200],  help="relation output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float, default=1, help="Margin used in hinge loss")
    args.add_argument("-b_conv", "--batch_size_conv", type=int, default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float, default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,default=0.3, help="Dropout probability for convolution layer")
    args = args.parse_args()
    return args

args = parse_args()

def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(args.data, is_unweigted=False, directed=True)
    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'), os.path.join(args.data, 'relation2vec.txt'))
        print("Initializes the embedding of entities and relationships")
    else:
        entity_embeddings = np.random.randn(len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)
    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)

Corpus_, entity_embeddings, relation_embeddings = load_data(args)
if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop, handle, protocol=pickle.HIGHEST_PROTOCOL)
if(args.use_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)
print("Initial entity dimensions {} , relation dimensions {}".format(entity_embeddings.size(), relation_embeddings.size()))
def save_model(model, name, epoch, folder_name):
    torch.save(model.state_dict(),(folder_name + "trained_{}.pth").format(epoch))
def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))
    positive_triples = train_indices[:len_pos_triples]
    negative_triples = train_indices[len_pos_triples:]
    positive_triples = positive_triples.repeat(int(args.valid_invalid_ratio_gat),1)
    head_embedding = entity_embed[positive_triples[:, 0]]  # 头实体的嵌入
    relation_embedding = relation_embed[positive_triples[:, 1]]  # 关系的嵌入
    tail_embedding = entity_embed[positive_triples[:, 2]]  # 尾实体的嵌入
    re_head_embedding, im_head_embedding = torch.chunk(head_embedding, 2, dim=1)
    re_relation_embedding, im_relation_embedding = torch.chunk(relation_embedding, 2, dim=1)
    re_tail_embedding, im_tail_embedding = torch.chunk(tail_embedding, 2, dim=1)
    head_score = re_head_embedding * re_relation_embedding + im_head_embedding * im_relation_embedding
    relation_score = re_relation_embedding + im_relation_embedding
    tail_score = re_tail_embedding * re_relation_embedding + im_tail_embedding * im_relation_embedding
    positive_score = head_score + relation_score - tail_score
    positive_score = torch.norm(positive_score, p=1, dim=1)
    head_embedding = entity_embed[negative_triples[:, 0]]  # 头实体的嵌入
    relation_embedding = relation_embed[negative_triples[:, 1]]  # 关系的嵌入
    tail_embedding = entity_embed[negative_triples[:, 2]]  # 尾实体的嵌入
    re_head_embedding, im_head_embedding = torch.chunk(head_embedding, 2, dim=1)
    re_relation_embedding, im_relation_embedding = torch.chunk(relation_embedding, 2, dim=1)
    re_tail_embedding, im_tail_embedding = torch.chunk(tail_embedding, 2, dim=1)
    head_score = re_head_embedding * re_relation_embedding + im_head_embedding * im_relation_embedding
    relation_score = re_relation_embedding + im_relation_embedding
    tail_score = re_tail_embedding * re_relation_embedding + im_tail_embedding * im_relation_embedding
    negative_score = head_score + relation_score - tail_score
    negative_score = torch.norm(negative_score, p=1, dim=1)
    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cpu()
    loss = gat_loss_func(positive_score, negative_score, y)
    return loss

def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.relation_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv, args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load('{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)),strict=False)
    model_conv.cpu()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities)
def train_model(args):
    print("Training encoder model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.relation_out_dim, args.drop_GAT, args.alpha, args.nheads_GAT)
    model_gat.cpu()
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5,last_epoch=-1)
    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)
    current_batch_2hop_indices = torch.tensor([])
    if (args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, Corpus_.unique_entities_train, node_neighbors_2hop)
    current_batch_2hop_indices = Variable(torch.LongTensor(current_batch_2hop_indices))
    epoch_losses = []
    print("Number of GAT-epochs {}".format(args.epochs_gat))
    for epoch in range(args.epochs_gat):
        print("\nGAT-epoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(list(Corpus_.train_triples)).astype(np.int32)
        model_gat.train()
        start_time = time.time()
        epoch_loss = []
        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (len(Corpus_.train_indices) // args.batch_size_gat) + 1
        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)
            train_indices = Variable(torch.LongTensor(train_indices))
            # train_values = Variable(torch.FloatTensor(train_values))

            entity_embed, relation_embed = model_gat(Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)
            optimizer.zero_grad()
            loss = batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
            end_time_iter = time.time()
            print("GAT-Iteration-> {0},GAT-Iteration_time-> {1:.4f} , GAT-Iteration_loss {2:.4f}".format(iters,  end_time_iter - start_time_iter, loss.data.item()))
        scheduler.step()
        print("GAT-Iteration {} , GAT-average loss {} , GAT-time {}".format(epoch, sum(epoch_loss) / len(epoch_loss),  time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
        save_model(model_gat, args.data, epoch, args.output_folder)
    print("Training dncoder model")
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.relation_out_dim,args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv, args.nheads_GAT, args.out_channels)
    model_gat.cpu()
    model_conv.cpu()
    model_gat.load_state_dict(torch.load('{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)),  strict=False)
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings
    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)
    optimizer = torch.optim.Adam(model_conv.parameters(), lr=args.lr, weight_decay = args.weight_decay_conv)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)
    margin_loss = torch.nn.SoftMarginLoss()
    epoch_losses = []
    print("Number of Conv-epochs {}".format(args.epochs_conv))
    for Conv_epoch in range(args.epochs_conv):
        print("\nConv-epoch-> ", Conv_epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(list(Corpus_.train_triples)).astype(np.int32)
        model_conv.train()
        start_time = time.time()
        epoch_loss = []
        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (len(Corpus_.train_indices) // args.batch_size_conv) + 1
        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)
            train_indices = Variable(torch.LongTensor(train_indices))
            train_values = Variable(torch.FloatTensor(train_values))
            preds = model_conv(Corpus_, Corpus_.train_adj_matrix, train_indices)
            optimizer.zero_grad()
            loss = margin_loss(preds.view(-1), train_values.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
            end_time_iter = time.time()
            print("Conv-Iteration-> {0}  , Conv-Iteration-time-> {1:.4f} , Conv-Iteration-loss {2:.4f}".format(iters,  end_time_iter - start_time_iter,  loss.data.item()))
        scheduler.step()
        print("Conv-Iteration {} , Conv-average loss {} , Conv-time {}".format(epoch, sum(epoch_loss) / len(epoch_loss),  time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
        save_model(model_conv, args.data, Conv_epoch, args.output_folder + "conv/")

def main(args):
    train_model(args)
    evaluate_conv(args, Corpus_.unique_entities_train)
if __name__ == '__main__':
    main(parse_args())


