# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from layers import MLP, EraseAddGate, MLPEncoder, MLPDecoder, SELayer
from utils import gumbel_softmax

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class GKT(nn.Module):

    def __init__(self, concept_num, hidden_dim, embedding_dim, edge_type_num, qt_one_hot_matrix, graph_model=None, dropout=0.5, bias=True, binary=False, has_cuda=False):
        super(GKT, self).__init__()
        self.concept_num = concept_num
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.edge_type_num = edge_type_num
        self.res_len = 2 if binary else 12
        self.has_cuda = has_cuda
        self.qt_kc_one_hot = qt_one_hot_matrix.values
        if self.has_cuda:
            self.qt_kc_one_hot = torch.from_numpy(self.qt_kc_one_hot).cuda()
        else:
            self.qt_kc_one_hot = torch.from_numpy(self.qt_kc_one_hot)
        zero_padding = torch.zeros(1, self.concept_num, device=self.qt_kc_one_hot.device)
        self.qt_kc_one_hot = torch.cat((self.qt_kc_one_hot, zero_padding), dim=0)
        self.graph_model = graph_model

        # one-hot feature and question
        one_hot_feat = torch.eye(self.concept_num)
        self.one_hot_feat = one_hot_feat.cuda() if self.has_cuda else one_hot_feat
        self.se_layer = SELayer(channel=self.concept_num, reduction=4)
        # self.one_hot_q = torch.eye(self.concept_num, device=self.one_hot_feat.device)
        # zero_padding = torch.zeros(1, self.concept_num, device=self.one_hot_feat.device)
        # self.one_hot_q = torch.cat((self.one_hot_q, zero_padding), dim=0)
        # concept and concept & response embeddings
        self.emb_x = nn.Embedding(self.res_len * concept_num, embedding_dim)
        # last embedding is used for padding, so dim + 1
        self.emb_c = nn.Embedding(concept_num + 1, embedding_dim, padding_idx=-1)

        # f_self function and f_neighbor functions
        mlp_input_dim = hidden_dim + embedding_dim
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.f_neighbor_list = nn.ModuleList()

        for i in range(edge_type_num):
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, concept_num)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # prediction layer
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, concept_num, hidden_dim]
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        qt = torch.where(qt != -1, qt, qt.shape[0] * torch.ones_like(qt, device=qt.device))

        # 拼接qc_one-hot矩阵
        # 补齐批次
        x_idx_mat = torch.arange(self.concept_num, device=xt.device)
        x_embedding = self.emb_x(x_idx_mat)  # [res_len * concept_num, embedding_dim]
        x_embedding = self.se_layer(x_embedding)
        #
        masked_feat = F.embedding(qt[qt_mask], self.qt_kc_one_hot.long())  # [mask_num, res_len * concept_num]

        # 是否需要
        xt = xt[qt_mask]
        xt_ = xt.reshape(-1, 1)
        temp = masked_feat * xt_

        res_embedding = temp.mm(x_embedding)  # [mask_num, embedding_dim]
        # [4,600] * [600*32]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = F.embedding(qt, self.qt_kc_one_hot.long())
        # concept_idx_mat[qt_mask, :] = torch.arange(self.concept_num, device=xt.device)
        qc_vector = self.emb_c(concept_idx_mat) #[51, 59, 32] # [batch_size, concept_num, embedding_dim]
        index_tuple = (torch.arange(mask_num, device=xt.device), qt[qt_mask].long())
        qc_vector[qt_mask] = qc_vector[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, qc_vector), dim=-1)  # [batch_size, concept_num, hidden_dim + embedding_dim]
        return tmp_ht


    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            qt: [batch_size]
            m_next: [batch_size, concept_num, hidden_dim]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + embedding_dim]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.concept_num, 1)  #[mask_num, concept_num, hidden_dim + embedding_dim]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)  #[mask_num, concept_num, 2 * (hidden_dim + embedding_dim)]
        concept_embedding, rec_embedding, z_prob = None, None, None

        concept_index = torch.arange(self.concept_num, device=qt.device)
        concept_embedding = self.emb_c(concept_index)  # [concept_num, embedding_dim]
        sp_send, sp_rec, sp_send_t, sp_rec_t = self._get_edges(masked_qt)
        graphs, rec_embedding, z_prob = self.graph_model(concept_embedding, sp_send, sp_rec, sp_send_t, sp_rec_t)
        neigh_features = 0
        for k in range(self.edge_type_num):

            # 拿到masked_qt中每个qt对应的kc_id
            mask_qt_kc = self.qt_kc_one_hot[masked_qt]
            kc_id = torch.arange(self.concept_num).cuda() * mask_qt_kc
            kc_id_non_zero = [row[row.nonzero(as_tuple=True)].long() for row in kc_id]
            # 取出对应kc_id的加和平均
            sublist = kc_id_non_zero[0]
            adj = torch.mean(graphs[k][sublist, :].unsqueeze(dim=-1), dim=0).unsqueeze(dim=0)
            for sublist in kc_id_non_zero[1:]:
                adj_vec = torch.mean(graphs[k][sublist, :].unsqueeze(dim=-1), dim=0).unsqueeze(dim=0)
                adj = torch.cat((adj_vec, adj), dim=0)

            if k == 0:
                neigh_features = adj * self.f_neighbor_list[k](neigh_ht)
            else:
                neigh_features = neigh_features + adj * self.f_neighbor_list[k](neigh_ht)

        # neigh_features: [mask_num, concept_num, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next, concept_embedding, rec_embedding, z_prob

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            ht: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        # mask_num = qt_mask.nonzero().shape[0]
        # 消除警告
        mask_indices = torch.nonzero(qt_mask, as_tuple=True)
        mask_num = mask_indices[0].shape[0]

        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt)  # [batch_size, concept_num, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device), )
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim))
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y

    def _get_next_pred(self, yt, next_qt):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = torch.where(next_qt != -1, next_qt, 50 * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.qt_kc_one_hot.long())
        # dot product between yt and one_hot_qt
        # pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
        pred = torch.topk((yt * one_hot_qt), k=1, dim=1).values.squeeze()
        return pred

    # Get edges for edge inference in VAE
    def _get_edges(self, masked_qt):
        r"""
        Parameters:
            masked_qt: qt index with -1 padding values removed
        Shape:
            masked_qt: [mask_num, ]
            rel_send: [edge_num, concept_num]
            rel_rec: [edge_num, concept_num]
        Return:
            rel_send: from nodes in edges which send messages to other nodes
            rel_rec:  to nodes in edges which receive messages from other nodes
        """
        mask_qt_kc = self.qt_kc_one_hot[masked_qt]
        kc_id = torch.arange(self.concept_num).cuda()
        mask_qt_kc_score = mask_qt_kc * kc_id
        masked_qt = mask_qt_kc_score[mask_qt_kc_score != 0]
        mask_num = torch.count_nonzero(masked_qt).item()

        row_arr = masked_qt.cpu().numpy().reshape(-1, 1)  # [mask_num, 1]
        row_arr = np.repeat(row_arr, self.concept_num, axis=1)  # [mask_num, concept_num]
        col_arr = np.arange(self.concept_num).reshape(1, -1)  # [1, concept_num]
        col_arr = np.repeat(col_arr, mask_num, axis=0)  # [mask_num, concept_num]
        # add reversed edges
        new_row = np.vstack((row_arr, col_arr))  # [2 * mask_num, concept_num]
        new_col = np.vstack((col_arr, row_arr))  # [2 * mask_num, concept_num]
        row_arr = new_row.flatten()  # [2 * mask_num * concept_num, ]
        col_arr = new_col.flatten()  # [2 * mask_num * concept_num, ]
        data_arr = np.ones(2 * mask_num * self.concept_num)
        init_graph = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(self.concept_num, self.concept_num))
        init_graph.setdiag(0)  # remove self-loop edges
        row_arr, col_arr, _ = sp.find(init_graph)
        row_tensor = torch.from_numpy(row_arr).long()
        col_tensor = torch.from_numpy(col_arr).long()
        one_hot_table = torch.eye(self.concept_num, self.concept_num)
        rel_send = F.embedding(row_tensor, one_hot_table)  # [edge_num, concept_num]
        rel_rec = F.embedding(col_tensor, one_hot_table)  # [edge_num, concept_num]
        sp_rec, sp_send = rel_rec.to_sparse(), rel_send.to_sparse()
        sp_rec_t, sp_send_t = rel_rec.T.to_sparse(), rel_send.T.to_sparse()
        sp_send = sp_send.to(device=masked_qt.device)
        sp_rec = sp_rec.to(device=masked_qt.device)
        sp_send_t = sp_send_t.to(device=masked_qt.device)
        sp_rec_t = sp_rec_t.to(device=masked_qt.device)
        return sp_send, sp_rec, sp_send_t, sp_rec_t

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=features.device))
        pred_list = []
        ec_list = []  # concept embedding list in VAE
        rec_list = []  # reconstructed embedding list in VAE
        z_prob_list = []  # probability distribution of latent variable z in VAE
        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1
            tmp_ht = self._aggregate(xt, qt, ht)  # [batch_size, concept_num, hidden_dim + embedding_dim]
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt)  # [batch_size, concept_num, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, concept_num]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
            ec_list.append(concept_embedding)
            rec_list.append(rec_embedding)
            z_prob_list.append(z_prob)
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
        return pred_res, ec_list, rec_list, z_prob_list


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, msg_hidden_dim, msg_output_dim, concept_num, edge_type_num=2,
                 tau=0.1, factor=True, dropout=0., bias=True):
        super(VAE, self).__init__()
        self.edge_type_num = edge_type_num
        self.concept_num = concept_num
        self.tau = tau
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim, factor=factor, dropout=dropout, bias=bias)
        self.decoder = MLPDecoder(input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=dropout, bias=bias)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(edge_type_num, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, edges, sp_rec, sp_send):
        r"""
        Parameters:
            edges: sampled latent graph edge weights from the probability distribution of the latent variable z
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send: one-hot encoded send-node index(sparse tensor)
        Shape:
            edges: [edge_num, edge_type_num]
            sp_rec: [edge_num, concept_num]
            sp_send: [edge_num, concept_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
        """
        x_index = sp_send._indices()[1].long()  # send node index: [edge_num, ]
        y_index = sp_rec._indices()[1].long()   # receive node index [edge_num, ]
        graphs = Variable(torch.zeros(self.edge_type_num, self.concept_num, self.concept_num, device=edges.device))
        for k in range(self.edge_type_num):
            index_tuple = (x_index, y_index)
            graphs[k] = graphs[k].index_put(index_tuple, edges[:, k])  # used for calculation
            #############################
            # here, we need to detach edges when storing it into self.graphs in case memory leak!
            self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, edges[:, k].detach())  # used for saving and visualization
            #############################
        return graphs

    def forward(self, data, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            data: input concept embedding matrix
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            data: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
            output: the reconstructed data
            prob: q(z|x) distribution
        """
        logits = self.encoder(data, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [edge_num, output_dim(edge_type_num)]
        edges = gumbel_softmax(logits, tau=self.tau, dim=-1)  # [edge_num, edge_type_num]
        prob = F.softmax(logits, dim=-1)
        output = self.decoder(data, edges, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [concept_num, embedding_dim]
        graphs = self._get_graph(edges, sp_send, sp_rec)
        return graphs, output, prob