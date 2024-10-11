import gc
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random
import config
import utils
from GNN_utils import KNNGraph
import DataUtil
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm, GATv2Conv, GATConv
from torch_geometric.utils import dropout_adj
from torch_geometric.data import DataLoader
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.data import Data


def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
            elif name.startswith('bias_'):
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    """
    initialize the weight and bias(if) of the given linear layer
    :param linear: linear layer
    :return:
    """
    linear.weight.data.normal_(std=config.init_normal_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.init_normal_std)


def init_wt_normal(wt):
    """
    initialize the given weight following the normal distribution
    :param wt: weight to be normal initialized
    :return:
    """
    wt.data.normal_(std=config.init_normal_std)


def init_wt_uniform(wt):
    """
    initialize the given weight following the uniform distribution
    :param wt: weight to be uniform initialized
    :return:
    """
    wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        if config.use_cuda:
            self.cuda(device=config.device)

    def forward(self, x):
        x = x + self.pe[:x.size()[0], :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Encoder for both code and ast
    """

    def __init__(self, vocab_size):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.pos_encoder = PositionalEncoding(self.hidden_size, dropout=0.2)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)
        if config.use_cuda:
            self.cuda(device=config.device)


    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """
        embedded = self.embedding(inputs) * math.sqrt(self.hidden_size)
        outputs = self.pos_encoder(embedded)
        outputs = self.transformer_encoder(outputs)
        return outputs

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)


class SourceEncoder(nn.Module):
    """
    Encoder for both code and ast
    """

    def __init__(self, vocab_size):
        super(SourceEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)
        if config.use_cuda:
            self.cuda(device=config.device)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """

        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        output, hidden = self.gru(packed)
        outputs = self.pos_encoder(embedded)
        outputs = self.transformer_encoder(outputs)
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)


class SSRvNNEncoder(nn.Module):
    """
    Encoder for both code and ast
    """

    def __init__(self, vocab_size, embedding_dim):
        super(SSRvNNEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.embedding_ast = []
        self.graph_set = None
        self.model_input = []

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        self.GATv2 = GATv2Conv(embedding_dim, embedding_dim, edge_dim=1)

        self.dropout = nn.Dropout(0.2)
        init_rnn_wt(self.gru)
        if config.use_cuda:
            self.cuda(device=config.device)

    def forward(self, inputs, sliced_flatten_ast_lens, batch_size):
        self.batch_size = batch_size
        embedded = self.embedding(inputs[0])

        self.GNNData_prepare(embedded, inputs[1], sliced_flatten_ast_lens)

        data = self.model_input
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.GATv2(x, edge_index, edge_weight)

        x = torch.unsqueeze(x, 1)
        x = torch.chunk(x, self.batch_size, dim=0)
        x = torch.cat(x, dim=1)
        x = torch.transpose(x, 0, 1)

        x = torch.split(x, sliced_flatten_ast_lens, dim=1)
        ans = None
        for i, temp_x in enumerate(x):
            C = temp_x.shape[1]
            b = torch.transpose(temp_x, 1, 2)
            d = nn.MaxPool1d(C)(b)
            e = torch.transpose(d, 1, 2)
            if i == 0:
                ans = e
            else:
                ans = torch.cat((ans, e), dim=1)

        x = torch.transpose(ans, 0, 1)
        outputs, hidden = self.gru(x)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden
    def GNNData_prepare(self, embedded, degree, sliced_flatten_ast_lens):

        temp_embedding = torch.chunk(embedded, self.batch_size, dim=1)
        temp_embedding = torch.stack(temp_embedding, dim=0)
        temp_embedding = torch.squeeze(temp_embedding)

        temp_embedding = torch.split(temp_embedding, sliced_flatten_ast_lens, dim=1)
        self.embedding_ast = temp_embedding
        del temp_embedding

        self.graph_set = []
        subtreeIndex = 0
        for a_net_data in self.embedding_ast:
            self.graph_set.append(Gen_graph(a_net_data, degree, subtreeIndex))
            subtreeIndex = subtreeIndex + 1
        graphSet = []
        for i1 in range(0, self.batch_size):
            for i2 in range(0, len(self.graph_set)):
                graphSet.append(self.graph_set[i2][i1])

        dataloader = DataLoader(graphSet, batch_size=self.batch_size*len(self.graph_set),
                                          shuffle=False,
                                          num_workers=0)
        del graphSet

        for data in dataloader:
            self.model_input = data.to(config.device)
def Gen_graph(data, degree, subtreeIndex):
    loal_distance = []
    data_list = []
    pdist = nn.PairwiseDistance(p=2)

    for i in range(len(data)):
        graph_feature = data[i]
        w = []
        if subtreeIndex >= len(degree[i]):
            node_edge = [[0], [1]]
            edge_index = torch.tensor(node_edge, dtype=torch.long).cuda(device=config.device)
            w = np.hstack((w, 0.00001))

            edge_features = torch.tensor(np.array(w), dtype=torch.float).cuda(device=config.device)

            graph = Data(x=graph_feature, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)
        else:
            node_edge = torch.tensor(degree[i][subtreeIndex]).cuda(device=config.device)
            start_node_matrix = torch.index_select(graph_feature, 0, node_edge[0])
            end_node_matrix = torch.index_select(graph_feature, 0, node_edge[1])
            w = pdist(start_node_matrix, end_node_matrix)
            beata = torch.mean(w)
            loal_weigt = torch.exp((-(w) ** 2) / (2 * (beata ** 2)))
            graph = Data(x=graph_feature, edge_index=node_edge, edge_attr=loal_weigt)
            data_list.append(graph)
    return data_list


class ReduceHidden(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(ReduceHidden, self).__init__()
        self.hidden_size = hidden_size

        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        init_linear_wt(self.linear)
        if config.use_cuda:
            self.cuda(device=config.device)

    def forward(self, code_hidden, ast_hidden):
        """

        :param code_hidden: hidden state of code encoder, [1, B, H]
        :param ast_hidden: hidden state of ast encoder, [1, B, H]
        :return: [1, B, H]
        """
        hidden = torch.cat((code_hidden, ast_hidden), dim=2)
        hidden = self.linear(hidden)
        hidden = F.relu(hidden)
        return hidden


class Attention(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True)
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        if config.use_cuda:
            self.cuda(device=config.device)

    def forward(self, hidden, encoder_outputs):
        """
        forward the net
        :param hidden: the last hidden state of encoder, [1, B, H]
        :param encoder_outputs: [T, B, H]
        :return: softmax scores, [B, 1, T]
        """
        time_step, batch_size, _ = encoder_outputs.size()
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        attn_energies = self.score(h, encoder_outputs)
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)

        return attn_weights

    def score(self, hidden, encoder_outputs):
        """
        calculate the attention scores of each word
        :param hidden: [B, T, H]
        :param encoder_outputs: [B, T, H]
        :return: energy: scores of each word in a batch, [B, T]
        """
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=config.hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        self.source_attention = Attention()
        self.code_attention = Attention()
        self.ast_attention = Attention()
        self.gru = nn.GRU(config.embedding_dim + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2 * self.hidden_size, config.nl_vocab_size)

        if config.use_pointer_gen:
            self.p_gen_linear = nn.Linear(2 * self.hidden_size + config.embedding_dim, 1)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)
        init_linear_wt(self.out)
        if config.use_cuda:
            self.cuda(device=config.device)

    def forward(self, inputs, last_hidden, source_outputs, code_outputs, ast_outputs,
                extend_source_batch, extra_zeros):
        """
        forward the net
        :param inputs: word input of current time step, [B]
        :param last_hidden: last decoder hidden state, [1, B, H]
        :param source_outputs: outputs of source encoder, [T, B, H]
        :param code_outputs: outputs of code encoder, [T, B, H]
        :param ast_outputs: outputs of ast encoder, [T, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: output: [B, nl_vocab_size]
                hidden: [1, B, H]
                attn_weights: [B, 1, T]
        """
        embedded = self.embedding(inputs).unsqueeze(0)
        source_attn_weights = self.source_attention(last_hidden, source_outputs)
        source_context = source_attn_weights.bmm(source_outputs.transpose(0, 1))
        source_context = source_context.transpose(0, 1)

        code_attn_weights = self.code_attention(last_hidden, code_outputs)

        code_context = code_attn_weights.bmm(code_outputs.transpose(0, 1))
        code_context = code_context.transpose(0, 1)

        ast_attn_weights = self.ast_attention(last_hidden, ast_outputs)
        ast_context = ast_attn_weights.bmm(ast_outputs.transpose(0, 1))
        ast_context = ast_context.transpose(0, 1)
        context = 0.5 * source_context + 0.5 * code_context + ast_context

        p_gen = None
        if config.use_pointer_gen:

            p_gen_input = torch.cat([context, last_hidden, embedded], dim=2)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)
            p_gen = p_gen.squeeze(0)

        rnn_input = torch.cat([embedded, context], dim=2)
        outputs, hidden = self.gru(rnn_input, last_hidden)

        outputs = outputs.squeeze(0)
        context = context.squeeze(0)
        vocab_dist = self.out(torch.cat([outputs, context], 1))
        vocab_dist = F.softmax(vocab_dist, dim=1)

        if config.use_pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            source_attn_weights_ = source_attn_weights.squeeze(1)
            attn_dist = (1 - p_gen) * source_attn_weights_

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], dim=1)

            final_dist = vocab_dist_.scatter_add(1, extend_source_batch, attn_dist)

        else:
            final_dist = vocab_dist

        final_dist = torch.log(final_dist + config.eps)

        return final_dist, hidden, source_attn_weights, code_attn_weights, ast_attn_weights, p_gen


class Model(nn.Module):

    def __init__(self, source_vocab_size, code_vocab_size, ast_vocab_size, nl_vocab_size, model=None, is_eval=False):
        super(Model, self).__init__()

        self.source_vocab_size = source_vocab_size
        self.code_vocab_size = code_vocab_size
        self.ast_vocab_size = ast_vocab_size
        self.is_eval = is_eval

        self.source_encoder = TransformerEncoder(self.source_vocab_size)
        self.code_encoder = SourceEncoder(self.code_vocab_size)
        self.ast_encoder = SSRvNNEncoder(self.ast_vocab_size, config.embedding_dim)

        self.reduce_hidden = ReduceHidden()
        self.decoder = Decoder(nl_vocab_size)

        if model:
            assert isinstance(model, str) or isinstance(model, dict)
            if isinstance(model, str):
                model = torch.load(model)
            self.load_state_dict(model)

        if config.use_cuda:
            self.cuda(device=config.device)

        if is_eval:
            self.eval()

    def forward(self, batch, batch_size, nl_vocab, is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        source_batch, source_seq_lens, code_batch, code_seq_lens, \
            ast_batch, ast_seq_lens, nl_batch, nl_seq_lens = batch.get_regular_input()

        sliced_flatten_ast, sliced_flatten_ast_lens = batch.get_sliced_flatten_ast()

        source_outputs = self.source_encoder(source_batch, source_seq_lens)
        code_outputs, code_hidden = self.code_encoder(code_batch, code_seq_lens)
        ast_outputs, ast_hidden = self.ast_encoder(sliced_flatten_ast, sliced_flatten_ast_lens, batch_size)


        code_hidden = code_hidden[0] + code_hidden[1]
        code_hidden = code_hidden.unsqueeze(0)
        ast_hidden = ast_hidden[0] + ast_hidden[1]
        ast_hidden = ast_hidden.unsqueeze(0)
        decoder_hidden = self.reduce_hidden(code_hidden, ast_hidden)

        if is_test:
            return source_outputs, code_outputs, ast_outputs, decoder_hidden

        if nl_seq_lens is None:
            max_decode_step = config.max_decode_steps
        else:
            max_decode_step = max(nl_seq_lens)

        decoder_inputs = utils.init_decoder_inputs(batch_size=batch_size, vocab=nl_vocab)

        extend_source_batch = None
        extra_zeros = None
        if config.use_pointer_gen:
            extend_source_batch, _, extra_zeros = batch.get_pointer_gen_input()
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size + batch.max_oov_num),
                                          device=config.device)
        else:
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size), device=config.device)

        for step in range(max_decode_step):
            decoder_output, decoder_hidden, source_attn_weights, code_attn_weights, ast_attn_weights, _ = self.decoder(
                inputs=decoder_inputs,
                last_hidden=decoder_hidden,
                source_outputs=source_outputs,
                code_outputs=code_outputs,
                ast_outputs=ast_outputs,
                extend_source_batch=extend_source_batch,
                extra_zeros=extra_zeros)
            decoder_outputs[step] = decoder_output

            if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:

                decoder_inputs = nl_batch[step]
            else:
                _, indices = decoder_output.topk(1)
                if config.use_pointer_gen:
                    word_indices = indices.squeeze(1).detach().cpu().numpy()
                    decoder_inputs = []
                    for index in word_indices:
                        decoder_inputs.append(utils.tune_up_decoder_input(index, nl_vocab))
                    decoder_inputs = torch.tensor(decoder_inputs, device=config.device)
                else:
                    decoder_inputs = indices.squeeze(1).detach()
                    decoder_inputs = decoder_inputs.to(config.device)

        return decoder_outputs
