# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from graph_unit import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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