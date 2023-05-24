# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from scipy.io import savemat
from utils import *
from graph_unit import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.num_of_timesteps = num_of_timesteps
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.randn(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.randn(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs)) / torch.sqrt(
            torch.tensor(self.num_of_timesteps))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=-1)

        return S_normalized


class Spatial_Attention_layer_2(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer_2, self).__init__()
        self.num_of_timesteps = num_of_timesteps
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.randn(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.randn(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs)) / torch.sqrt(
            torch.tensor(self.num_of_timesteps))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=-1)

        return S_normalized


class Spatial_Attention_layer_Dot_Product(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, num_of_vertices, d_model):
        super(Spatial_Attention_layer_Dot_Product, self).__init__()
        self.d_model = d_model
        self.WK = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.WQ = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))

    #        print(self.WK)
    # nn.init.xavier_uniform_(self.WK.data, gain=1)
    # nn.init.xavier_uniform_(self.WQ.data, gain=1)
    # nn.init.normal_(self.WK, mean=0, std=1)
    # nn.init.normal_(self.WQ, mean=0, std=1)
    # nn.init.xavier_normal_(self.WK.data, gain=1)
    # nn.init.xavier_normal_(self.WQ.data, gain=1)
    def forward(self, x):
        '''
        :param x: (batch_size, N, T)
        :return: (B,N,N)
        '''

        Q = torch.matmul(x, self.WQ)
        K_T = torch.matmul(x, self.WK).permute(0, 2, 1)

        product = torch.matmul(Q, K_T)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.relu(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=-1)

        return S_normalized


class Spatial_Attention_layer_Dot_Product_fea_emb(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, num_of_vertices, d_model):
        super(Spatial_Attention_layer_Dot_Product_fea_emb, self).__init__()
        self.d_model = d_model
        self.embed_layer = nn.Embedding(
            num_embeddings=num_of_vertices, embedding_dim=d_model
        )
        self.WK = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.WQ = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.device = DEVICE

    def forward(self, x):
        '''
        :param x: (batch_size, N, T)
        :return: (B,N,N)
        '''
        B, N, T = x.shape
        feature_embed = self.embed_layer(
            torch.arange(N).to(self.device)
        )
        feature_embed = torch.repeat_interleave(feature_embed.unsqueeze(0), B, dim=0)
        lamb = 0.5
        Q = lamb * torch.matmul(x, self.WQ) + (1 - lamb) * feature_embed
        K_T = lamb * torch.matmul(x, self.WK).permute(0, 2, 1) + (1 - lamb) * feature_embed.permute(0, 2, 1)

        product = torch.matmul(Q, K_T)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.sigmoid(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=-1)

        return S_normalized

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])

        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe
# class Spatial_Attention_layer_Dot_Product_fea_emb(nn.Module):
#    '''
#    compute spatial attention scores
#    '''
#    def __init__(self, DEVICE, num_of_vertices, d_model):
#        super(Spatial_Attention_layer_Dot_Product_fea_emb, self).__init__()
#        self.embed_layer = nn.Embedding(
#            num_embeddings=num_of_vertices, embedding_dim=d_model // 3
#        )
#        d_model = self.d_model = d_model + d_model // 3
#        self.WK = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
#        self.WQ = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
#        self.device = DEVICE
#    def forward(self, x):
#        '''
#        :param x: (batch_size, N, T)
#        :return: (B,N,N)
#        '''
#        B, N, T = x.shape
#        feature_embed = self.embed_layer(
#            torch.arange(N).to(self.device)
#        )
#        feature_embed = torch.repeat_interleave(feature_embed.unsqueeze(0), B, dim=0)
#        x = torch.cat([x, feature_embed], dim=-1)
#        Q = torch.matmul(x, self.WQ)
#        K_T = torch.matmul(x, self.WK).permute(0, 2, 1)
#
#        product = torch.matmul(Q, K_T)  # (b,N,T)(b,T,N) -> (B, N, N)
#
#        S = torch.sigmoid(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)
#
#        S_normalized = F.softmax(S, dim=-1)
#
#        return S_normalized
class ASTGCN_block_GRU_transformer(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_distribution(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                           d_out=1, n_layers=n_layer,
                                                           num_nodes=num_of_vertices, support_len=1,
                                                           kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,src_key_padding_mask=\
        #        (1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)).squeeze(-1).permute(1, 0))
        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
        #            src_key_padding_mask=(1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)))
        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        #        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        #        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        #        ffted_length = (ffted_real ** 2 + ffted_imag ** 2) ** 0.5

        #        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
        #            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        #        lambd = 0
        #        ffted_length = lambd * ffted_length + (1 - lambd) * x
        #        ffted_length = torch.cat([ffted_length, x], dim=-1)
        spatial_At = torch.mean(self.SAt(torch.squeeze(x)), dim=0)

        #        spatial_At = 0.5 * (spatial_At + spatial_At.T)

        #        spatial_At_mask = spatial_At < 1 / 36
        #        spatial_At[spatial_At_mask] = 0
        #        spatial_At[~spatial_At_mask] = spatial_At[~spatial_At_mask] * 3
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_wo_variable_attention(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_wo_variable_attention, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        #        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_distribution(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                           d_out=1, n_layers=n_layer,
                                                           num_nodes=num_of_vertices, support_len=1,
                                                           kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,src_key_padding_mask=\
        #        (1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)).squeeze(-1).permute(1, 0))
        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
        #            src_key_padding_mask=(1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)))
        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        #        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        #        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        #        ffted_length = (ffted_real ** 2 + ffted_imag ** 2) ** 0.5

        #        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
        #            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        #        lambd = 0
        #        ffted_length = lambd * ffted_length + (1 - lambd) * x
        #        ffted_length = torch.cat([ffted_length, x], dim=-1)
        #        spatial_At = torch.mean(self.SAt(torch.squeeze(x)), dim=0)
        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        #        spatial_At = 0.5 * (spatial_At + spatial_At.T)

        #        spatial_At_mask = spatial_At < 1 / 36
        #        spatial_At[spatial_At_mask] = 0
        #        spatial_At[~spatial_At_mask] = spatial_At[~spatial_At_mask] * 3
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer6(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer6, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(2 * graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        lambd = 0.5
        ffted_length = lambd * ffted_length + (1 - lambd) * x

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer5(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer5, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        num_of_observation = torch.sum(mask, dim=-1)
        # num_of_observation
        weights = 1 / num_of_observation
        weights[torch.where(weights == torch.inf)] = 0

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
        x_TAt = x_TAt * weights.unsqueeze(-1)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_wo_temporal_attention(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_wo_temporal_attention, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        # encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
        #                                          dim_feedforward=self.attention_d_model, dropout=0.3)
        # self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        # self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))

        x_TAt = temporal_observation_embedding
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_wo_frequency_variable_attention(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_wo_frequency_variable_attention, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        # ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        # ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        # ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
        #     torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(x)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


# class ASTGCN_block_GRU_transformer_wo_variable_attention(nn.Module):
#
#    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static, n_class, n_layer=1, kernel_size=3):
#        super(ASTGCN_block_GRU_transformer_wo_variable_attention, self).__init__()
#        self.num_of_vertices = num_of_vertices
#        self.attention_d_model = attention_d_model
#        self.graph_node_d_model = graph_node_d_model
#        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
#        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
#                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
#        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
#
#        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
#        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
#
#        self.GCRNN = GCRNN_epsilon_early_stop_wo_variable_attention(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1, n_layers=n_layer,
#                                   num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)
#
#        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
#        self.d_static = d_static
#        if d_static != 0:
#            self.emb = nn.Linear(d_static, num_of_vertices)
#            self.classifier = nn.Sequential(
#                nn.Linear(num_of_vertices * 2, 200),
#                nn.ReLU(),
#                nn.Linear(200, n_class)).to(DEVICE)
#        else:
#            self.classifier = nn.Sequential(
#                nn.Linear(num_of_vertices, 200),
#                nn.ReLU(),
#                nn.Linear(200, n_class)).to(DEVICE)
#        self.DEVICE = DEVICE
#
#        self.to(DEVICE)
#    def forward(self, x, delta_t, static=None, lengths=0):
#        '''
#        :param x: (B, N_nodes, F_in, T_in)
#        :return: (B, N_nodes, T_out)
#        '''
#        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
#        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
#        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2, 1).unsqueeze(1)
#        x = value
#        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
#        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
#        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1)\
#            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
#        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
#        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
#        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
#        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
#        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding, num_of_vertices, dim=1)
#        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
#        temporal_transformer_output = temporal_transformer_output\
#            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
#        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
#        # x_TAt = temporal_transformer_output
#        # return x_TAt.permute(0, 2, 1, 3)
#
#
#        # ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
#        # ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
#        # ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
#        #     torch.float32)  # [batch, variable_num, 1, time]
#        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
#        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
#
#        # spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
#        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
#        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
#
#        # [batches, nodes, hidden_dim, timestamps]
#        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)
#
#        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))
#
#        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
#        if self.d_static != 0:
#            emb = self.emb(static)
#            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
#        else:
#            output = self.classifier(torch.squeeze(out))
#        return output

class ASTGCN_block_GRU_transformer_wo_time_interval_modeling(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_wo_time_interval_modeling, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_wo_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                                 n_layers=n_layer,
                                                 num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer2(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer2, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)

        self.spectral_input_projection = nn.Linear(num_of_timesteps, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, attention_d_model)
        self.SAt = Spatial_Attention_layer(DEVICE, 3, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)
        # print(temporal_transformer_output[0, 0, 0, 0])

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag)  # [batch, variable_num, 1, time]
        ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        ffted_input = self.spectral_input_projection(torch.cat([ffted_length, ffted_phi], dim=2))

        # spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_input)), dim=0)
        spatial_At = torch.mean(self.SAt(torch.cat([ffted_input, mask.permute(0, 2, 1, 3)], dim=2)), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)
        # print(spatial_gcn[0, 0, 0])
        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer3(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer3, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        num = torch.sum(mask, dim=-1).permute(0, 2, 1)
        num[torch.where(num == 0)] = 1
        mean = torch.repeat_interleave((torch.sum(x, dim=-1) / (num)).unsqueeze(-1),
                                       num_of_timesteps, dim=-1)

        x = x * mask.permute(0, 2, 1, 3) + mean * (1 - mask.permute(0, 2, 1, 3))
        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer4(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer4, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        lengths_mask = torch.zeros_like(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[2]):
                if torch.where(mask[i][0][j] == 1)[0].shape[0] != 0:
                    lengths_j = torch.max(torch.where(mask[i][0][j] == 1)[0])
                    lengths_mask[i][0][j][:lengths_j] = 1
        # num = torch.sum(mask, dim=-1).permute(0, 2, 1)
        # num[torch.where(num==0)] = 1
        # mean = torch.repeat_interleave((torch.sum(x, dim=-1) / (num)).unsqueeze(-1),
        #                                num_of_timesteps, dim=-1)
        #
        # x = x * mask.permute(0, 2, 1, 3) + mean * (1 - mask.permute(0, 2, 1, 3))
        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        ffted_length = ffted_length * lengths_mask.permute(0, 2, 1, 3)
        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


# class ASTGCN_block_GRU_transformer_nufft(nn.Module):
#
#    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static, n_class, n_layer=1, kernel_size=3):
#        super(ASTGCN_block_GRU_transformer_nufft, self).__init__()
#        self.num_of_vertices = num_of_vertices
#        self.attention_d_model = attention_d_model
#        self.graph_node_d_model = graph_node_d_model
#        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
#        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
#                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
#        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
#
#        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
#        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
#        self.spectral_input_projection = nn.Conv2d(in_channels=3, out_channels=self.attention_d_model, kernel_size=1)
#
#        # self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
#        self.SAt = Spatial_Attention_layer(DEVICE, self.attention_d_model, num_of_vertices, num_of_timesteps)
#
#        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1, n_layers=n_layer,
#                                   num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)
#
#        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
#        self.d_static = d_static
#        if d_static != 0:
#            self.emb = nn.Linear(d_static, num_of_vertices)
#            self.classifier = nn.Sequential(
#                nn.Linear(num_of_vertices * 2, 200),
#                nn.ReLU(),
#                nn.Linear(200, n_class)).to(DEVICE)
#        else:
#            self.classifier = nn.Sequential(
#                nn.Linear(num_of_vertices, 200),
#                nn.ReLU(),
#                nn.Linear(200, n_class)).to(DEVICE)
#        self.DEVICE = DEVICE
#
#        self.to(DEVICE)
#    def forward(self, x, nufft, delta_t, static=None, lengths=0):
#        '''
#        :param x: (B, N_nodes, F_in, T_in)
#        :return: (B, N_nodes, T_out)
#        '''
#        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
#        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
#        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2, 1).unsqueeze(1)
#        x = value
#        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
#        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
#        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1)\
#            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
#        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
#        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
#        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
#        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
#        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding, num_of_vertices, dim=1)
#        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
#        temporal_transformer_output = temporal_transformer_output\
#            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
#        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
#        # x_TAt = temporal_transformer_output
#        # return x_TAt.permute(0, 2, 1, 3)
#
#        fft_embedding = self.spectral_input_projection(nufft)
#
#        spatial_At = torch.mean(self.SAt(fft_embedding.permute(0, 3, 1, 2)), dim=0)
#        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
#
#        # [batches, nodes, hidden_dim, timestamps]
#        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)
#
#        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))
#
#        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
#        if self.d_static != 0:
#            emb = self.emb(static)
#            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
#        else:
#            output = self.classifier(torch.squeeze(out))
#        return output

class ASTGCN_block_GRU_transformer_nufft(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        # self.spectral_input_projection = nn.Conv2d(in_channels=3, out_channels=self.attention_d_model, kernel_size=1)
        # self.spectral_input_projection = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        # self.spectral_projection = nn.Conv2d(5, 1024, kernel_size=1)
        # self.spectral_projection2 = nn.Conv2d(1024, 1, kernel_size=1)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, 2 * num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.attention_d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, num_of_vertices),
                nn.ReLU(),
                nn.Linear(num_of_vertices, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        # fft_embedding = self.spectral_input_projection(nufft)
        # nufft = self.spectral_projection(nufft)
        # spatial_At = torch.mean(self.SAt(fft_embedding.permute(0, 3, 1, 2)), dim=0)
        # spatial_At = torch.mean(self.SAt(torch.squeeze(nufft[:, 1, :, :]).permute(0, 2, 1)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
        # fuse_embedding = torch.cat([x.permute(0, 2, 1, 3), mask, nufft.permute(0, 1, 3, 2)], dim=1)
        # fuse_embedding = self.spectral_projection(fuse_embedding)
        # fuse_embedding = self.spectral_projection2(fuse_embedding)
        # nufft = self.spectral_projection(nufft[:, 1, :, :].permute(0, 2, 1))
        #        fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(x)], dim=-1)
        step_emb = step * mask / 48
        fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(step_emb)], dim=-1)
        spatial_At = torch.mean(self.SAt(torch.squeeze(fuse_embedding)), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_fre(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_fre, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        # self.frequency_input_projection = nn.Conv2d(in_channels=num_of_timesteps, out_channels=attention_d_model, kernel_size=1)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)
        self.frequency_projection = nn.Linear(num_of_timesteps, self.graph_node_d_model // 4)
        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model + graph_node_d_model // 4, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
        # frequency_info = torch.repeat_interleave(self.frequency_input_projection(
        #     ffted_length.permute(0, 3, 2, 1)), num_of_timesteps, dim=-2).permute(0, 1, 3, 2)
        # x_TAt = self.time_frequency_fuse(torch.cat([x_TAt, frequency_info], dim=1))

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)
        frequency_info = self.frequency_projection(ffted_length)

        fuse_info = torch.cat([spatial_gcn.unsqueeze(-1), frequency_info.permute(0, 3, 1, 2)], dim=1)
        out = torch.squeeze(self.final_conv(fuse_info))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_density(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_density, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_density(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                      d_out=1, n_layers=n_layer,
                                                      num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)
        self.density_weight = nn.Linear(1, graph_node_d_model)
        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, density, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, density, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class Spatial_Attention_layer_Dot_Product_prune(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, num_of_vertices, d_model):
        super(Spatial_Attention_layer_Dot_Product_prune, self).__init__()
        self.d_model = d_model
        self.WK = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.WQ = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        # nn.init.xavier_uniform_(self.WK.data, gain=1)
        # nn.init.xavier_uniform_(self.WQ.data, gain=1)
        # nn.init.normal_(self.WK, mean=0, std=1)
        # nn.init.normal_(self.WQ, mean=0, std=1)
        # nn.init.xavier_normal_(self.WK.data, gain=1)
        # nn.init.xavier_normal_(self.WQ.data, gain=1)

    def forward(self, x, prune_mask):
        '''
        :param x: (batch_size, N, T)
        :return: (B,N,N)
        '''

        Q = torch.matmul(x, self.WQ)
        K_T = torch.matmul(x, self.WK).permute(0, 2, 1)

        product = torch.matmul(Q, K_T)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.sigmoid(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)

        S.masked_fill(prune_mask, 1e-9)

        S_normalized = F.softmax(S, dim=-1)

        return S_normalized


class ASTGCN_block_GRU_transformer_prune(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_prune, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product_prune(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        prune_mask = torch.repeat_interleave(torch.sum(mask, dim=-1) == 0, num_of_vertices, dim=1)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length), prune_mask), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer7(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer7, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.time_fre_fuse = nn.Linear(2 * num_of_timesteps, self.attention_d_model)
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, attention_d_model)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        fuse_embedding = self.time_fre_fuse(torch.cat([torch.squeeze(ffted_length), torch.squeeze(x)], dim=-1))
        spatial_At = torch.mean(self.SAt(fuse_embedding), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer9(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer9, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.reduction = nn.Conv2d(self.attention_d_model, 1, kernel_size=1, stride=1)
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask= \
                                                                            (1 - mask.permute(3, 0, 2, 1).reshape(
                                                                                num_of_timesteps,
                                                                                batch_size * num_of_vertices,
                                                                                1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        # ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        # ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        # ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
        #     torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        # spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        spatial_At = torch.mean(self.SAt(torch.squeeze(self.reduction(x_TAt))), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # spatial_At[1 - spatial_At_mask] = 1

        # sigma = torch.std(spatial_At)
        # adj = torch.exp(-torch.square(spatial_At / sigma))
        # adj[adj < ] = 0.
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer10(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer10, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.reduction = nn.Conv2d(self.attention_d_model, 1, kernel_size=1, stride=1)
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)
        self.sum_adj = torch.zeros(size=[num_of_vertices, num_of_vertices]).to(DEVICE)
        self.total = 0
        self.spatial_At = torch.zeros(size=[num_of_vertices, num_of_vertices]).to(DEVICE)
        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0, update=False):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask= \
                                                                            (1 - mask.permute(3, 0, 2, 1).reshape(
                                                                                num_of_timesteps,
                                                                                batch_size * num_of_vertices,
                                                                                1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        if update == True:
            ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
            ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
            ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
                torch.float32)  # [batch, variable_num, 1, time]
            self.sum_adj = self.sum_adj + torch.sum(self.SAt(torch.squeeze(ffted_length)), dim=0)
            self.total = self.total + x.shape[0]
            self.spatial_At = self.sum_adj / self.total
        # spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # adj.append(self.SAt(torch.squeeze(ffted_length)))
        # spatial_At = torch.mean(self.SAt(torch.squeeze(self.reduction(x_TAt))), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # spatial_At[1 - spatial_At_mask] = 1

        # sigma = torch.std(spatial_At)
        # adj = torch.exp(-torch.square(spatial_At / sigma))
        # adj[adj < ] = 0.
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, self.spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer11(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer11, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_exact(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                    d_out=1, n_layers=n_layer,
                                                    num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask= \
                                                                            (1 - mask.permute(3, 0, 2, 1).reshape(
                                                                                num_of_timesteps,
                                                                                batch_size * num_of_vertices,
                                                                                1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = self.SAt(torch.squeeze(ffted_length))
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer12(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer12, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.fre_encoder = nn.Linear(in_features=num_of_timesteps, out_features=self.attention_d_model)
        self.weight_key = nn.Parameter(torch.randn(size=(self.attention_d_model, 1)))
        self.weight_query = nn.Parameter(torch.randn(size=(self.attention_d_model, 1)))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)
        # self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask= \
                                                                            (1 - mask.permute(3, 0, 2, 1).reshape(
                                                                                num_of_timesteps,
                                                                                batch_size * num_of_vertices,
                                                                                1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        input = self.fre_encoder(torch.squeeze(ffted_length))
        key = torch.matmul(input, self.weight_key)  # 32 * 228 * 1
        query = torch.matmul(input, self.weight_query)  # 32 * 228 * 1
        data = key.repeat(1, 1, num_of_vertices).view(batch_size, num_of_vertices * num_of_vertices, 1) + \
               query.repeat(1, num_of_vertices, 1)  # 32 * （228 * 228） * 1
        data = data.squeeze(2)
        data = data.view(batch_size, num_of_vertices, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=-1)
        attention = self.dropout(attention)
        spatial_At = torch.mean(attention, dim=0)
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        # spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class Spatial_Attention_layer_Dot_Product_stem(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, num_of_vertices, d_model):
        super(Spatial_Attention_layer_Dot_Product_stem, self).__init__()
        self.d_model = d_model
        self.WK = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.WQ = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)
        # nn.init.xavier_uniform_(self.WK.data, gain=1)
        # nn.init.xavier_uniform_(self.WQ.data, gain=1)
        # nn.init.normal_(self.WK, mean=0, std=1)
        # nn.init.normal_(self.WQ, mean=0, std=1)
        # nn.init.xavier_normal_(self.WK.data, gain=1)
        # nn.init.xavier_normal_(self.WQ.data, gain=1)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T)
        :return: (B,N,N)
        '''

        Q = torch.matmul(x, self.WQ)
        K_T = torch.matmul(x, self.WK).permute(0, 2, 1)

        product = torch.matmul(Q, K_T)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.sigmoid(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)

        S = self.leakyrelu(S)

        S_normalized = F.softmax(S, dim=-1)

        S_normalized = self.dropout(S_normalized)

        return S_normalized


class ASTGCN_block_GRU_transformer13(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer13, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product_stem(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask= \
                                                                            (1 - mask.permute(3, 0, 2, 1).reshape(
                                                                                num_of_timesteps,
                                                                                batch_size * num_of_vertices,
                                                                                1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)

        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_nufft_3(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_3, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)

        self.spectral_encoder = nn.Sequential(
            nn.Linear(in_features=num_of_timesteps, out_features=self.attention_d_model),
            nn.ReLU()
        )
        self.observerd_timestamp_encoder = nn.Sequential(
            nn.Linear(in_features=num_of_timesteps, out_features=self.attention_d_model),
            nn.ReLU()
        )
        self.fuse_encoder = nn.Sequential(
            nn.Linear(in_features=2 * self.attention_d_model, out_features=2 * self.attention_d_model),
            nn.ReLU()
        )
        # self.spectral_input_projection = nn.Conv2d(in_channels=3, out_channels=self.attention_d_model, kernel_size=1)
        # self.spectral_input_projection = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        # self.spectral_projection = nn.Conv2d(5, 1024, kernel_size=1)
        # self.spectral_projection2 = nn.Conv2d(1024, 1, kernel_size=1)

        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, 2 * self.attention_d_model)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.attention_d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        # fft_embedding = self.spectral_input_projection(nufft)
        # nufft = self.spectral_projection(nufft)
        # spatial_At = torch.mean(self.SAt(fft_embedding.permute(0, 3, 1, 2)), dim=0)
        # spatial_At = torch.mean(self.SAt(torch.squeeze(nufft[:, 1, :, :]).permute(0, 2, 1)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
        # fuse_embedding = torch.cat([x.permute(0, 2, 1, 3), mask, nufft.permute(0, 1, 3, 2)], dim=1)
        # fuse_embedding = self.spectral_projection(fuse_embedding)
        # fuse_embedding = self.spectral_projection2(fuse_embedding)
        # nufft = self.spectral_projection(nufft[:, 1, :, :].permute(0, 2, 1))
        step_emb = torch.squeeze(step * mask)
        step_emb = self.observerd_timestamp_encoder(step_emb)
        spectral_emb = self.spectral_encoder(nufft[:, 1, :, :].permute(0, 2, 1))
        fuse_emb = self.fuse_encoder(torch.cat([step_emb, spectral_emb], dim=-1))
        # fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(step_emb)], dim=-1)

        # fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(x)], dim=-1)
        # fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(step_emb)], dim=-1)

        # spatial_At = torch.mean(self.SAt(torch.squeeze(fuse_embedding)), dim=0)
        spatial_At = torch.mean(self.SAt(fuse_emb), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class Spatial_Attention_layer_MTGNN(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(Spatial_Attention_layer_MTGNN, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        #        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        #        mask.fill_(float('0'))
        #        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        #        mask.scatter_(1,t1,s1.fill_(1))
        #        adj = adj*mask
        return adj


class ASTGCN_block_GRU_transformer_nufft_MTGNN(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_MTGNN, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        # self.spectral_input_projection = nn.Conv2d(in_channels=3, out_channels=self.attention_d_model, kernel_size=1)
        # self.spectral_input_projection = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        # self.spectral_projection = nn.Conv2d(5, 1024, kernel_size=1)
        # self.spectral_projection2 = nn.Conv2d(1024, 1, kernel_size=1)
        self.SAt = Spatial_Attention_layer_MTGNN(nnodes=num_of_vertices, k=10, dim=64, device=DEVICE)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.attention_d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        # fft_embedding = self.spectral_input_projection(nufft)
        # nufft = self.spectral_projection(nufft)
        # spatial_At = torch.mean(self.SAt(fft_embedding.permute(0, 3, 1, 2)), dim=0)
        # spatial_At = torch.mean(self.SAt(torch.squeeze(nufft[:, 1, :, :]).permute(0, 2, 1)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
        # fuse_embedding = torch.cat([x.permute(0, 2, 1, 3), mask, nufft.permute(0, 1, 3, 2)], dim=1)
        # fuse_embedding = self.spectral_projection(fuse_embedding)
        # fuse_embedding = self.spectral_projection2(fuse_embedding)
        # nufft = self.spectral_projection(nufft[:, 1, :, :].permute(0, 2, 1))
        spatial_At = self.SAt(torch.arange(num_of_vertices).to(self.DEVICE))

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class Spatial_Attention_layer_Dot_Product_emb(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, num_of_vertices, num_of_timestamps, d_model):
        super(Spatial_Attention_layer_Dot_Product_emb, self).__init__()
        self.d_model = d_model
        self.linear = nn.Linear(2 * num_of_timestamps, d_model)
        self.node_emb = nn.Embedding(num_of_vertices, d_model)
        self.fuse_linear = nn.Linear(2 * d_model, d_model)
        self.WK = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.WQ = nn.Parameter(torch.randn(d_model, d_model).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, T)
        :return: (B,N,N)
        '''
        batch = x.shape[0]
        x_info = F.tanh(self.linear(x))
        node_emb = torch.repeat_interleave(self.node_emb.weight.unsqueeze(0), batch, dim=0)
        fuse_emb = F.tanh(self.fuse_linear(torch.cat([x_info, node_emb], dim=-1)))

        Q = torch.matmul(fuse_emb, self.WQ)
        K_T = torch.matmul(fuse_emb, self.WK).permute(0, 2, 1)

        product = torch.matmul(Q, K_T)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.sigmoid(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=-1)

        return S_normalized


class ASTGCN_block_GRU_transformer_nufft_emb(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_emb, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)

        self.SAt = Spatial_Attention_layer_Dot_Product_emb(DEVICE, num_of_vertices,
                                                           num_of_timesteps, self.attention_d_model)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.attention_d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output

        step_emb = torch.squeeze(step * mask)
        spectral_emb = nufft[:, 1, :, :].permute(0, 2, 1)
        fuse_info = torch.cat([step_emb, torch.squeeze(x)], dim=-1)
        # fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(step_emb)], dim=-1)

        # fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(x)], dim=-1)
        # fuse_embedding = torch.cat([nufft[:, 1, :, :].permute(0, 2, 1), torch.squeeze(step_emb)], dim=-1)

        # spatial_At = torch.mean(self.SAt(torch.squeeze(fuse_embedding)), dim=0)
        spatial_At = torch.mean(self.SAt(fuse_info), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer14(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer14, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps * 2)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input, src_key_padding_mask=transformer_mask)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask= \
                                                                            (1 - mask.permute(3, 0, 2, 1).reshape(
                                                                                num_of_timesteps,
                                                                                batch_size * num_of_vertices,
                                                                                1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.cat([torch.squeeze(ffted_length), torch.squeeze(mask)], dim=-1)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer15(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer15, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.attention_d_model // 4, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_mask_te(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                      d_out=1, n_layers=n_layer,
                                                      num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,src_key_padding_mask=\
        # (1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_var_length(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_var_length, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, var_length, var_last_obs_tp, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = (torch.arange(num_of_timesteps)[None, :] >= (
        var_length.reshape(batch_size * num_of_vertices).cpu()[:, None])).cuda()
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        # temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,src_key_padding_mask=\
        # (1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)).squeeze(-1).permute(1, 0))
        temporal_transformer_output = torch.where(torch.isnan(temporal_transformer_output),
                                                  torch.full_like(temporal_transformer_output, 0),
                                                  temporal_transformer_output)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer16(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer16, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=1, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop(d_in=self.attention_d_model, d_model=self.graph_node_d_model, d_out=1,
                                              n_layers=n_layer,
                                              num_nodes=num_of_vertices, support_len=1, kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(x.permute(0, 2, 1, 3))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,src_key_padding_mask=\
        #        (1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)).squeeze(-1).permute(1, 0))

        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
        #            src_key_padding_mask=(1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)))
        x_TAt = mask * temporal_observation_embedding + (1 - mask) * temporal_transformer_output
        # x_TAt = temporal_transformer_output
        # return x_TAt.permute(0, 2, 1, 3)

        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
            torch.float32)  # [batch, variable_num, 1, time]
        # ffted_phi = torch.arctan(ffted_imag / (ffted_real + 1e-8))
        # ffted_input = torch.cat([ffted_length, ffted_phi], dim=2)
        lambd = 0
        ffted_length = lambd * ffted_length + (1 - lambd) * x

        spatial_At = torch.mean(self.SAt(torch.squeeze(ffted_length)), dim=0)

        # spatial_At = 0.5 * (spatial_At + spatial_At.T)
        #        spatial_At_mask = spatial_At < 1 / 36
        #        spatial_At[spatial_At_mask] = 0
        #        spatial_At[~spatial_At_mask] = spatial_At[~spatial_At_mask] * 3
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_nufft_4(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_4, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        #        self.SAt_p = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_distribution(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                           d_out=1, n_layers=n_layer,
                                                           num_nodes=num_of_vertices, support_len=1,
                                                           kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,src_key_padding_mask=\
        #        (1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)).squeeze(-1).permute(1, 0))
        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
        #            src_key_padding_mask=(1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)))
        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output

        #        fre_fuse = torch.cat([torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1)), torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))], dim=-1)
        #        spatial_At = torch.mean(self.SAt(torch.squeeze(x)), dim=0)
        #        spatial_At_m = torch.mean(self.SAt_m(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)
        #        spatial_At_p = torch.mean(self.SAt_p(torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))), dim=0)
        spatial_At = torch.mean(self.SAt(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)
        #        spatial_At = torch.mean(self.SAt(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1)) * torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))), dim=0)
        #        adj_mask_m = torch.zeros_like(spatial_At_m).to(self.DEVICE)
        #        adj_mask_m.fill_(float('0'))
        #        s1, t1 = adj_mask_m.topk(num_of_vertices // 4, -1)
        #        adj_mask_m.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_m = spatial_At_m * adj_mask_m
        #        spatial_At_m = 0.5 * (spatial_At_m + spatial_At_m.T)
        #
        #        adj_mask_p = torch.zeros_like(spatial_At_p).to(self.DEVICE)
        #        adj_mask_p.fill_(float('0'))
        #        s1, t1 = adj_mask_p.topk(num_of_vertices // 4, -1)
        #        adj_mask_p.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_p = spatial_At_p * adj_mask_p
        #        spatial_At_p = 0.5 * (spatial_At_p + spatial_At_p.T)
        #
        #        spatial_At = 0.5 * (spatial_At_m + spatial_At_p)

        #        spatial_At = 0.5 * (spatial_At_m + spatial_At_p)
        ##        spatial_At = torch.mean(self.SAt(torch.squeeze(x)), dim=0)
        ##        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        adj_mask = torch.zeros_like(spatial_At).to(self.DEVICE)
        adj_mask.fill_(float('0'))
        s1, t1 = spatial_At.topk(num_of_vertices // 4, -1)
        adj_mask.scatter_(1, t1, s1.fill_(1))
        spatial_At = spatial_At * adj_mask
        spatial_At = 0.5 * (spatial_At + spatial_At.T)

        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_nufft_5(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_5, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)

        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)

        self.m_linear = nn.Linear(num_of_timesteps, 128)
        self.p_linear = nn.Linear(num_of_timesteps, 128)
        self.SAt = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, 128)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_distribution(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                           d_out=1, n_layers=n_layer,
                                                           num_nodes=num_of_vertices, support_len=1,
                                                           kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,src_key_padding_mask=\
        #        (1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)).squeeze(-1).permute(1, 0))
        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
        #            src_key_padding_mask=(1 - mask.permute(3, 0, 2, 1).reshape(num_of_timesteps, batch_size * num_of_vertices, 1)))
        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output

        # fre_fuse = torch.cat([torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1)), torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))], dim=-1)
        magnitudes = F.tanh(self.m_linear(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))))
        phase = F.tanh(self.p_linear(torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))))

        spatial_At = torch.mean(self.SAt(F.tanh(magnitudes * phase)), dim=0)
        #        spatial_At = torch.mean(self.SAt(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)
        spatial_At = 0.5 * (spatial_At + spatial_At.T)

        #        spatial_At_mask = spatial_At < 1 / 36
        #        spatial_At[spatial_At_mask] = 0
        #        spatial_At[~spatial_At_mask] = spatial_At[~spatial_At_mask] * 3
        # spatial_At = torch.mean(self.SAt(ffted_length), dim=0)

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


# class ASTGCN_block_GRU_transformer_nufft_tftopk(nn.Module):
#
#    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
#                 n_class, n_layer=1, kernel_size=3):
#        super(ASTGCN_block_GRU_transformer_nufft_tftopk, self).__init__()
#        self.num_of_vertices = num_of_vertices
#        self.attention_d_model = attention_d_model
#        self.graph_node_d_model = graph_node_d_model
#        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
#        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
#                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
#        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
#        self.topK = 10
#        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
#        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
#        self.SAt_time = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
#        self.SAt_frequency = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
#        #        self.SAt_p = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
#        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)
#
#        self.GCRNN = GCRNN_epsilon_early_stop_distribution(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
#                                                           d_out=1, n_layers=n_layer,
#                                                           num_nodes=num_of_vertices, support_len=1,
#                                                           kernel_size=kernel_size)
#
#        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
#        self.d_static = d_static
#        if d_static != 0:
#            self.emb = nn.Linear(d_static, num_of_vertices)
#            self.classifier = nn.Sequential(
#                nn.Linear(num_of_vertices * 2, 200),
#                nn.ReLU(),
#                nn.Linear(200, n_class)).to(DEVICE)
#        else:
#            self.classifier = nn.Sequential(
#                nn.Linear(num_of_vertices, 200),
#                nn.ReLU(),
#                nn.Linear(200, n_class)).to(DEVICE)
#        self.DEVICE = DEVICE
#
#        self.to(DEVICE)
#
#    def forward(self, x, nufft, delta_t, static=None, lengths=0):
#        '''
#        :param x: (B, N_nodes, F_in, T_in)
#        :return: (B, N_nodes, T_out)
#        '''
#        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
#        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
#        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
#                                                                                                        1).unsqueeze(1)
#        x = value
#        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
#        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
#        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
#            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
#        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
#        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
#        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
#        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
#        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
#                                                                                          num_of_vertices, dim=1)
#        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
#                                                                        src_key_padding_mask=transformer_mask)
#
#        temporal_transformer_input = temporal_transformer_input \
#            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
#        temporal_transformer_output = temporal_transformer_output \
#            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
#
#        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output
#
#        spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
#        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)
##        spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
##        spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)
#        adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
#        adj_mask_f.fill_(float('0'))
#        s1, t1 = spatial_At_f.topk(self.topK // 2, -1)
#        adj_mask_f.scatter_(1, t1, s1.fill_(1))
#        spatial_At_f = spatial_At_f * adj_mask_f
#
##        spatial_At_t = spatial_At_t *
#        adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
#        adj_mask_t.fill_(float('0'))
#        s1, t1 = spatial_At_t.topk(self.topK // 2, -1)
#        adj_mask_t.scatter_(1, t1, s1.fill_(1))
#        spatial_At_t = spatial_At_t * adj_mask_t
#
#        spatial_At = spatial_At_f + spatial_At_t
#        spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
#        spatial_At = (spatial_At + spatial_At.T)
##        spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
##        spatial_At = 0.5 * (spatial_At + spatial_At.T)
#
##        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
#        # [batches, nodes, hidden_dim, timestamps]
#        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)
#
#        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))
#
#        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
#        if self.d_static != 0:
#            emb = self.emb(static)
#            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
#        else:
#            output = self.classifier(torch.squeeze(out))
#        return output
class ASTGCN_block_GRU_transformer_nufft_tftopk(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_tftopk, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.sigma = nn.ReLU()

        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        self.SAt_frequency = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        #        self.SAt_frequency_p = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        #        self.SAt_p = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_distribution(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                           d_out=1, n_layers=n_layer,
                                                           num_nodes=num_of_vertices, support_len=1,
                                                           kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.sigma(
            self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1)))
        #        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)
        #
        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #
        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output
        #        x_TAt = temporal_transformer_input
        #        x_TAt = temporal_transformer_output
        #        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        #        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        #        # ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
        #        #     torch.float32)  # [batch, variable_num, 1, time]
        #        ffted_length = (ffted_real ** 2 + ffted_imag ** 2) ** 0.5
        #        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(ffted_length)), dim=0)

        #        spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        #        spatial_At_t = torch.mean(self.SAt_frequency(torch.squeeze(x)), dim=0)

        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)

        #        spatial_At_p = torch.mean(self.SAt_frequency_p(torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))), dim=0)
        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        #        adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        #        adj_mask_f.fill_(float('0'))
        #        s1, t1 = spatial_At_f.topk(self.topK, -1)
        #        adj_mask_f.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_f = spatial_At_f * adj_mask_f

        #        adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        #        adj_mask_t.fill_(float('0'))
        #        s1, t1 = spatial_At_t.topk(self.topK, -1)
        #        adj_mask_t.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_t = spatial_At_t * adj_mask_t

        #        adj_mask_p = torch.zeros_like(spatial_At_p).to(self.DEVICE)
        #        adj_mask_p.fill_(float('0'))
        #        s1, t1 = spatial_At_p.topk(self.topK // 3, -1)
        #        adj_mask_p.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_p = spatial_At_p * adj_mask_p * (1 - adj_mask_f) * (1 - adj_mask_t)

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)
        #
        #
        spatial_At = spatial_At_f
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        spatial_At = 0.5 * (torch.softmax(spatial_At, dim=-1) + torch.softmax(spatial_At, dim=-1).T)
        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class all_ablation(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(all_ablation, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        #        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
        #                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        #        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        #        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        #        self.SAt_time = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        #        self.SAt_frequency = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        #        self.SAt_frequency_p = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        #        self.SAt_p = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_distribution_wo_diffusion(d_in=self.attention_d_model,
                                                                        d_model=self.graph_node_d_model,
                                                                        d_out=1, n_layers=n_layer,
                                                                        num_nodes=num_of_vertices, support_len=1,
                                                                        kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
        #                                                                        src_key_padding_mask=transformer_mask)
        #
        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #        temporal_transformer_output = temporal_transformer_output \
        #            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #
        #        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output
        x_TAt = temporal_transformer_input
        #        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        #        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        #        # ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
        #        #     torch.float32)  # [batch, variable_num, 1, time]
        #        ffted_length = (ffted_real ** 2 + ffted_imag ** 2) ** 0.5
        #        spatial_At = torch.mean(self.SAt_frequency(torch.squeeze(ffted_length)), dim=0)

        #        spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        #        spatial_At_t = torch.mean(self.SAt_frequency(torch.squeeze(x)), dim=0)
        #        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)
        #        spatial_At_p = torch.mean(self.SAt_frequency_p(torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))), dim=0)
        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        #        adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        #        adj_mask_f.fill_(float('0'))
        #        s1, t1 = spatial_At_f.topk(self.topK, -1)
        #        adj_mask_f.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_f = spatial_At_f * adj_mask_f

        #        adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        #        adj_mask_t.fill_(float('0'))
        #        s1, t1 = spatial_At_t.topk(self.topK, -1)
        #        adj_mask_t.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_t = spatial_At_t * adj_mask_t

        #        adj_mask_p = torch.zeros_like(spatial_At_p).to(self.DEVICE)
        #        adj_mask_p.fill_(float('0'))
        #        s1, t1 = spatial_At_p.topk(self.topK // 3, -1)
        #        adj_mask_p.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_p = spatial_At_p * adj_mask_p * (1 - adj_mask_f) * (1 - adj_mask_t)

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)
        #
        #
        #        spatial_At = spatial_At_f
        #        spatial_At = 0.5 * (spatial_At + spatial_At.T)

        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices

        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_nufft_fttopk(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_fttopk, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        self.SAt_frequency = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        #        self.SAt_p = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, self.d_model, num_of_vertices, num_of_timesteps)

        self.GCRNN = GCRNN_epsilon_early_stop_distribution(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                                           d_out=1, n_layers=n_layer,
                                                           num_nodes=num_of_vertices, support_len=1,
                                                           kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)

        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)

        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output

        spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)

        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        adj_mask_t.fill_(float('0'))
        s1, t1 = spatial_At_t.topk(self.topK // 2, -1)
        adj_mask_t.scatter_(1, t1, s1.fill_(1))
        spatial_At_t = spatial_At_t * adj_mask_t

        adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        adj_mask_f.fill_(float('0'))
        s1, t1 = spatial_At_f.topk(self.topK // 2, -1)
        adj_mask_f.scatter_(1, t1, s1.fill_(1))
        spatial_At_f = spatial_At_f * adj_mask_f * (1 - adj_mask_t)

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)

        spatial_At = spatial_At_f + spatial_At_t
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        # spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class ASTGCN_block_GRU_transformer_nufft_tftopk_wo_diffusion(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(ASTGCN_block_GRU_transformer_nufft_tftopk_wo_diffusion, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        self.SAt_frequency = Spatial_Attention_layer_Dot_Product(DEVICE, num_of_vertices, num_of_timesteps)
        self.GCRNN = GCRNN_epsilon_early_stop_distribution_wo_diffusion(d_in=self.attention_d_model,
                                                                        d_model=self.graph_node_d_model,
                                                                        d_out=1, n_layers=n_layer,
                                                                        num_nodes=num_of_vertices, support_len=1,
                                                                        kernel_size=kernel_size)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)

        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)

        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output

        #        ffted_real = torch.fft.fft(x).real  # [batch, variable_num, 1, time]
        #        ffted_imag = torch.fft.fft(x).imag  # [batch, variable_num, 1, time]
        #        # ffted_length = torch.sqrt(ffted_real * ffted_real + ffted_imag * ffted_imag).to(
        #        #     torch.float32)  # [batch, variable_num, 1, time]
        #        ffted_length = (ffted_real ** 2 + ffted_imag ** 2) ** 0.5
        #        spatial_At = torch.mean(self.SAt_frequency(torch.squeeze(ffted_length)), dim=0)
        #
        # spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)
        #        spatial_At_p = torch.mean(self.SAt_frequency_p(torch.squeeze(nufft[:, 2, :, :].permute(0, 2, 1))), dim=0)
        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        # adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        # adj_mask_f.fill_(float('0'))
        # s1, t1 = spatial_At_f.topk(self.topK, -1)
        # adj_mask_f.scatter_(1, t1, s1.fill_(1))
        # spatial_At_f = spatial_At_f * adj_mask_f

        # adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        # adj_mask_t.fill_(float('0'))
        # s1, t1 = spatial_At_t.topk(self.topK, -1)
        # adj_mask_t.scatter_(1, t1, s1.fill_(1))
        # spatial_At_t = spatial_At_t * adj_mask_t

        #        adj_mask_p = torch.zeros_like(spatial_At_p).to(self.DEVICE)
        #        adj_mask_p.fill_(float('0'))
        #        s1, t1 = spatial_At_p.topk(self.topK // 3, -1)
        #        adj_mask_p.scatter_(1, t1, s1.fill_(1))
        #        spatial_At_p = spatial_At_p * adj_mask_p * (1 - adj_mask_f) * (1 - adj_mask_t)

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)
        #
        #
        spatial_At = spatial_At_f
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        # spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class gru_test(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3):
        super(gru_test, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.gru = nn.GRU(36, 36, 2)
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.d_static = d_static

        self.classifier = nn.Sequential(
            nn.Linear(num_of_vertices, 200),
            nn.ReLU(),
            nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        _, h = self.gru(torch.squeeze(x).permute(2, 0, 1))
        # self.gru(temporal_transformer_input)

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        output = self.classifier(torch.squeeze(h[0]))
        return output


class Variable_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, num_of_vertices, num_of_timestamps, d_model):
        super(Variable_Attention_layer, self).__init__()
        self.d_model = d_model
        self.WK = nn.Parameter(torch.randn(num_of_timestamps, d_model).to(DEVICE))
        self.WQ = nn.Parameter(torch.randn(num_of_timestamps, d_model).to(DEVICE))
        # nn.init.xavier_uniform_(self.WK.data, gain=1)
        # nn.init.xavier_uniform_(self.WQ.data, gain=1)
        # nn.init.normal_(self.WK, mean=0, std=1)
        # nn.init.normal_(self.WQ, mean=0, std=1)
        # nn.init.xavier_normal_(self.WK.data, gain=1)
        # nn.init.xavier_normal_(self.WQ.data, gain=1)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T)
        :return: (B,N,N)
        '''

        Q = torch.matmul(x, self.WQ)
        K_T = torch.matmul(x, self.WK).permute(0, 2, 1)

        product = torch.matmul(Q, K_T)  # (b,N,T)(b,T,N) -> (B, N, N)

        # S = torch.sigmoid(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)
        S = torch.relu(product) / torch.sqrt(torch.tensor(self.d_model))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=-1)

        return S_normalized


class CGU(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3, at=0, bt=0, varatt_dim=0, beta_start=1e-5, beta_end=2e-5):
        super(CGU, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.sigma = nn.ReLU()
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.softmax = nn.Softmax()
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                 varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.SAt_frequency = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                      varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.GCRNN = CGRNN(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                           d_out=1, n_layers=n_layer,
                           num_nodes=num_of_vertices, support_len=1,
                           kernel_size=kernel_size, at=at, bt=bt, beta_start=beta_start, beta_end=beta_end)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        #        temporal_observation_embedding = self.sigma(self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1)))
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)

        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)

        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output

        # spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)

        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        # adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        # adj_mask_f.fill_(float('0'))
        # s1, t1 = spatial_At_f.topk(self.topK // 2, -1)
        # adj_mask_f.scatter_(1, t1, s1.fill_(1))
        # spatial_At_f = spatial_At_f * adj_mask_f

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)
        # adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        # adj_mask_t.fill_(float('0'))
        # s1, t1 = spatial_At_t.topk(self.topK // 2, -1)
        # adj_mask_t.scatter_(1, t1, s1.fill_(1))
        # spatial_At_t = spatial_At_t * adj_mask_t * (1 - adj_mask_f)

        # spatial_At = spatial_At_f + spatial_At_t
        spatial_At = spatial_At_f
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        spatial_At = 0.5 * (torch.softmax(spatial_At, dim=-1) + torch.softmax(spatial_At, dim=-1).T)
        # spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class CGU_timevaratt(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3, at=0, bt=0, varatt_dim=0):
        super(CGU_timevaratt, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.sigma = nn.ReLU()
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.softmax = nn.Softmax()
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                 varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.SAt_frequency = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                      varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.GCRNN = CGRNN(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                           d_out=1, n_layers=n_layer,
                           num_nodes=num_of_vertices, support_len=1,
                           kernel_size=kernel_size, at=at, bt=bt)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        #        temporal_observation_embedding = self.sigma(self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1)))
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)

        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)

        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output

        spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        #        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)

        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        # adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        # adj_mask_f.fill_(float('0'))
        # s1, t1 = spatial_At_f.topk(self.topK // 2, -1)
        # adj_mask_f.scatter_(1, t1, s1.fill_(1))
        # spatial_At_f = spatial_At_f * adj_mask_f

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)
        # adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        # adj_mask_t.fill_(float('0'))
        # s1, t1 = spatial_At_t.topk(self.topK // 2, -1)
        # adj_mask_t.scatter_(1, t1, s1.fill_(1))
        # spatial_At_t = spatial_At_t * adj_mask_t * (1 - adj_mask_f)

        # spatial_At = spatial_At_f + spatial_At_t
        spatial_At = spatial_At_t
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        spatial_At = 0.5 * (torch.softmax(spatial_At, dim=-1) + torch.softmax(spatial_At, dim=-1).T)
        # spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class CGU_wovaratt(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3, at=0, bt=0, varatt_dim=0):
        super(CGU_wovaratt, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.sigma = nn.ReLU()
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.softmax = nn.Softmax()
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        #        self.SAt_time = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps, varatt_dim if varatt_dim!=0 else num_of_timesteps)
        #        self.SAt_frequency = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps, varatt_dim if varatt_dim!=0 else num_of_timesteps)
        self.GCRNN = CGRNN(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                           d_out=1, n_layers=n_layer,
                           num_nodes=num_of_vertices, support_len=1,
                           kernel_size=kernel_size, at=at, bt=bt)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        #        temporal_observation_embedding = self.sigma(self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1)))
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)

        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)

        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output
        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class CGU_wo_timeatt(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3, at=0, bt=0, varatt_dim=0):
        super(CGU_wo_timeatt, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.sigma = nn.ReLU()
        #        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
        #                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        #        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.softmax = nn.Softmax()
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                 varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.SAt_frequency = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                      varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.GCRNN = CGRNN(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                           d_out=1, n_layers=n_layer,
                           num_nodes=num_of_vertices, support_len=1,
                           kernel_size=kernel_size, at=at, bt=bt)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        #        temporal_observation_embedding = self.sigma(self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1)))
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        #        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
        #                                                                        src_key_padding_mask=transformer_mask)

        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        #        temporal_transformer_output = temporal_transformer_output \
        #            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)

        x_TAt = temporal_transformer_input

        # spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)

        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        # adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        # adj_mask_f.fill_(float('0'))
        # s1, t1 = spatial_At_f.topk(self.topK // 2, -1)
        # adj_mask_f.scatter_(1, t1, s1.fill_(1))
        # spatial_At_f = spatial_At_f * adj_mask_f

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)
        # adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        # adj_mask_t.fill_(float('0'))
        # s1, t1 = spatial_At_t.topk(self.topK // 2, -1)
        # adj_mask_t.scatter_(1, t1, s1.fill_(1))
        # spatial_At_t = spatial_At_t * adj_mask_t * (1 - adj_mask_f)

        # spatial_At = spatial_At_f + spatial_At_t
        spatial_At = spatial_At_f
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        spatial_At = 0.5 * (torch.softmax(spatial_At, dim=-1) + torch.softmax(spatial_At, dim=-1).T)
        # spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class CGU_small(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3, at=0, bt=0, varatt_dim=0):
        super(CGU_small, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.sigma = nn.ReLU()
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.softmax = nn.Softmax()
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                 varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.SAt_frequency = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                      varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.GCRNN = CGRNN(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                           d_out=1, n_layers=n_layer,
                           num_nodes=num_of_vertices, support_len=1,
                           kernel_size=kernel_size, at=at, bt=bt)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # temporal_observation_embedding = self.sigma(self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1)))
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        x_TAt = temporal_transformer_input
        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


class CGU_wo_interval(nn.Module):

    def __init__(self, DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, d_static,
                 n_class, n_layer=1, kernel_size=3, at=0, bt=0, varatt_dim=0):
        super(CGU_wo_interval, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.attention_d_model = attention_d_model
        self.graph_node_d_model = graph_node_d_model
        self.input_projection = nn.Conv2d(in_channels=2, out_channels=self.attention_d_model, kernel_size=1, stride=1)
        self.sigma = nn.ReLU()
        encoder_layers = TransformerEncoderLayer(d_model=self.attention_d_model, nhead=1,
                                                 dim_feedforward=self.attention_d_model, dropout=0.3)
        self.transformer_encoder_temporal = TransformerEncoder(encoder_layers, 1)
        self.softmax = nn.Softmax()
        self.topK = 10
        self.time_encoder = PositionalEncodingTF(self.attention_d_model, num_of_timesteps, 100)
        self.time_encoder2 = PositionalEncodingTF(self.graph_node_d_model, num_of_timesteps, 100)
        self.SAt_time = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                 varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.SAt_frequency = Variable_Attention_layer(DEVICE, num_of_vertices, num_of_timesteps,
                                                      varatt_dim if varatt_dim != 0 else num_of_timesteps)
        self.GCRNN = CGRNN_wo_interval(d_in=self.attention_d_model, d_model=self.graph_node_d_model,
                                       d_out=1, n_layers=n_layer,
                                       num_nodes=num_of_vertices, support_len=1,
                                       kernel_size=kernel_size, at=at, bt=bt)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, nufft, delta_t, static=None, lengths=0):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        value = x[:, :, :self.num_of_vertices].permute(0, 2, 1).unsqueeze(2)
        mask = x[:, :, self.num_of_vertices:2 * self.num_of_vertices].permute(0, 2, 1).unsqueeze(1)
        step = torch.repeat_interleave(x[:, :, -1].unsqueeze(-1), self.num_of_vertices, dim=-1).permute(0, 2,
                                                                                                        1).unsqueeze(1)
        x = value
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # temporal_observation_embedding = self.sigma(self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1)))
        temporal_observation_embedding = self.input_projection(torch.cat([x.permute(0, 2, 1, 3), mask], dim=1))
        temporal_transformer_input = temporal_observation_embedding.permute(3, 0, 2, 1) \
            .reshape(num_of_timesteps, batch_size * num_of_vertices, self.attention_d_model)
        transformer_mask = torch.arange(num_of_timesteps)[None, :] >= (lengths.cpu()[:, None])
        transformer_mask = torch.repeat_interleave(transformer_mask.cuda(), num_of_vertices, dim=0)
        time_encoding = self.time_encoder(step.squeeze(1)[:, 0, :].permute(1, 0))
        graph_input_time_encoding = self.time_encoder2(step.squeeze(1)[:, 0, :].permute(1, 0))
        temporal_transformer_input = temporal_transformer_input + torch.repeat_interleave(time_encoding,
                                                                                          num_of_vertices, dim=1)
        temporal_transformer_output = self.transformer_encoder_temporal(temporal_transformer_input,
                                                                        src_key_padding_mask=transformer_mask)

        temporal_transformer_input = temporal_transformer_input \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)
        temporal_transformer_output = temporal_transformer_output \
            .reshape(num_of_timesteps, batch_size, num_of_vertices, self.attention_d_model).permute(1, 3, 2, 0)

        x_TAt = mask * temporal_transformer_input + (1 - mask) * temporal_transformer_output

        # spatial_At_t = torch.mean(self.SAt_time(torch.squeeze(x)), dim=0)
        spatial_At_f = torch.mean(self.SAt_frequency(torch.squeeze(nufft[:, 1, :, :].permute(0, 2, 1))), dim=0)

        # spatial_At_t = 0.5 * (spatial_At_t + spatial_At_t.T)
        # spatial_At_f = 0.5 * (spatial_At_f + spatial_At_f.T)

        # adj_mask_f = torch.zeros_like(spatial_At_f).to(self.DEVICE)
        # adj_mask_f.fill_(float('0'))
        # s1, t1 = spatial_At_f.topk(self.topK // 2, -1)
        # adj_mask_f.scatter_(1, t1, s1.fill_(1))
        # spatial_At_f = spatial_At_f * adj_mask_f

        # spatial_At_t = spatial_At_t * (1 - adj_mask_f)
        # adj_mask_t = torch.zeros_like(spatial_At_t).to(self.DEVICE)
        # adj_mask_t.fill_(float('0'))
        # s1, t1 = spatial_At_t.topk(self.topK // 2, -1)
        # adj_mask_t.scatter_(1, t1, s1.fill_(1))
        # spatial_At_t = spatial_At_t * adj_mask_t * (1 - adj_mask_f)

        # spatial_At = spatial_At_f + spatial_At_t
        spatial_At = spatial_At_f
        spatial_At = 0.5 * (spatial_At + spatial_At.T)
        spatial_At = 0.5 * (torch.softmax(spatial_At, dim=-1) + torch.softmax(spatial_At, dim=-1).T)
        # spatial_At[torch.where(spatial_At > 0.04)] = spatial_At[torch.where(spatial_At > 0.04)] * 0.5
        #        spatial_At = torch.ones(size=[num_of_vertices, num_of_vertices]).to(self.DEVICE) / num_of_vertices
        # [batches, nodes, hidden_dim, timestamps]
        spatial_gcn = self.GCRNN(x_TAt, spatial_At, delta_t, lengths, graph_input_time_encoding, mask)

        out = torch.squeeze(self.final_conv(spatial_gcn.unsqueeze(-1)))

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        if self.d_static != 0:
            emb = self.emb(static)
            output = self.classifier(torch.cat([torch.squeeze(out), emb], dim=-1))
        else:
            output = self.classifier(torch.squeeze(out))
        return output


def make_model(DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps, n_layer, kernel_size,
               d_static, n_class, ablation):
    if ablation == 'full':
        #        model = gru_test(DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps,
        #                                         d_static, n_class, n_layer, kernel_size)
        #        model = all_ablation(DEVICE, attention_d_model, graph_node_d_model, num_of_vertices, num_of_timesteps,
        #                                         d_static, n_class, n_layer, kernel_size)
        model = ASTGCN_block_GRU_transformer_nufft_tftopk(DEVICE, attention_d_model, graph_node_d_model,
                                                          num_of_vertices, num_of_timesteps,
                                                          d_static, n_class, n_layer, kernel_size)
    elif ablation == 'wo_temporal_attention':
        model = ASTGCN_block_GRU_transformer_wo_temporal_attention(DEVICE, attention_d_model, graph_node_d_model,
                                                                   num_of_vertices, num_of_timesteps,
                                                                   d_static, n_class, n_layer, kernel_size)
    elif ablation == 'wo_frequency_variable_attention':
        model = ASTGCN_block_GRU_transformer_wo_frequency_variable_attention(DEVICE, attention_d_model,
                                                                             graph_node_d_model, num_of_vertices,
                                                                             num_of_timesteps,
                                                                             d_static, n_class, n_layer, kernel_size)
    elif ablation == 'wo_variable_attention':
        model = ASTGCN_block_GRU_transformer_wo_variable_attention(DEVICE, attention_d_model, graph_node_d_model,
                                                                   num_of_vertices, num_of_timesteps,
                                                                   d_static, n_class, n_layer, kernel_size)
    elif ablation == 'wo_time_interval_modeling':
        model = ASTGCN_block_GRU_transformer_nufft_tftopk_wo_diffusion(DEVICE, attention_d_model, graph_node_d_model,
                                                                       num_of_vertices, num_of_timesteps,
                                                                       d_static, n_class, n_layer, kernel_size)
    return model