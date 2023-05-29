import torch
from torch import nn

epsilon = 1e-8

class SpatialConvOrderK(nn.Module):
    """
    Spatial convolution of order K with possibly different diffusion matrices (useful for directed graphs)

    Efficient implementation inspired from graph-wavenet codebase
    """

    def __init__(self, c_in, c_out, support_len=1, order=2, include_self=True):
        super(SpatialConvOrderK, self).__init__()
        self.include_self = include_self
        # c_in = (order + 1) * c_in
        c_in = (order * support_len + 1) * c_in
        # c_in = (order * support_len + (1 if include_self else 0)) * c_in
        self.support_len = support_len
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.order = order

    @staticmethod
    def compute_support(adj, device=None):
        if device is not None:
            adj = adj.to(device)
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + epsilon)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + epsilon)
        support = [adj_fwd, adj_bwd]
        return support

    @staticmethod
    def compute_support_orderK(adj, k, include_self=False, device=None):
        if isinstance(adj, (list, tuple)):
            support = adj
        else:
            support = SpatialConvOrderK.compute_support(adj, device)
        supp_k = []
        for a in support:
            ak = a
            for i in range(k - 1):
                ak = torch.matmul(ak, a.T)
                if not include_self:
                    ak.fill_diagonal_(0.)
                supp_k.append(ak)
        return support + supp_k

    def forward(self, x, support):
        # [batch, features, nodes, steps]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
            for i in range(1, self.support_len):
                support.append(support[0])
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2

        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        return out

class epsilon_distribution(nn.Module):
    def __init__(self, d_model, num_variables, device='cuda:0'):
        super().__init__()
        self.num_variables = num_variables
        self.device = device
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=d_model, dropout=0.3)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        transformer_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=d_model, dropout=0.3)
        self.transformer2 = nn.TransformerEncoder(transformer_layer2, num_layers=1)
        # self.gate = SpatialConvOrderK(c_in=d_model, c_out=d_model, support_len=1, order=3)
        self.embed_layer = nn.Embedding(
            num_embeddings=num_variables, embedding_dim=d_model
        )
        self.linear_mu = nn.Linear(d_model, d_model)
        self.linear_sigma = nn.Linear(d_model, d_model)

    def forward(self, h, mask, mask_last_tp, time_encoding, adj):
        num_variable, batch, hidden_dim = h.shape
        feature_embed = self.embed_layer(
            torch.arange(self.num_variables).to(self.device)
        )  # (K,emb)
        output_1 = h
        transformer_input_2 = output_1 + torch.repeat_interleave(
            feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask_last_tp.permute(2, 0, 1)
        transformer_output = self.transformer2(transformer_input_2)
        return self.linear_mu(transformer_output), self.linear_sigma(transformer_output)

class CGRNN(nn.Module):
    def __init__(self, d_in, d_model, d_out, n_layers, support_len, num_nodes, kernel_size=2, at=0, bt=0, beta_start=1e-5, beta_end=2e-5):
        super(CGRNN, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(CGRNN_cell(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                             num_units=self.d_model, support_len=self.support_len, order=self.ks, at=at,
                                             bt=bt, beta_start=beta_start, beta_end=beta_end))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj, mask, mask_last_tp, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj, mask, mask_last_tp, time_encoding, step)

        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(int(torch.max(lengths))):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
            # return output[-1]
            # else:
            delta_t = delta_t_tensor[:, step].unsqueeze(-1)
            # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
            x_input = x[..., step]
            mask_input_last_tp = mask[..., step - 1] if step > 0 else mask[..., 0]
            mask_input = mask[..., step] if step > 0 else mask[..., 0]
            time_encoding_input = time_encoding[step - 1] if step > 0 else time_encoding[0]
            out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input,
                                      step)
            output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
            # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output

class CGRNN_cell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh', at=0, bt=0, beta_start=1e-5, beta_end=2e-5):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(CGRNN_cell, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units
        # beta_start = 1e-4
        # beta_end = 2e-4
        print(beta_start, beta_end)
        self.dt = 100
        self.at = at
        self.bt = bt
        if at == 0:
            # print(beta_end, ' ', beta_start)
            #        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, 1000).cuda()
            betas = torch.linspace(beta_end ** 0.5, beta_start ** 0.5, self.dt + 1).cuda()
            #            betas = torch.linspace(beta_end, beta_start, self.dt+1).cuda()
            # print(betas.shape[0])
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            # sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.epsilon = epsilon_distribution(d_model=num_units, num_variables=num_nodes)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        # self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        # self.residual = nn.Conv2d(d_in, num_units, kernel_size=1, stride=1)

    def forward(self, x, delta_t, h, adj, mask, mask_last_tp, time_encoding, step):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape

        if step != 0:
            epsilon_mu, epsilon_sigma = self.epsilon(h.permute(2, 0, 1), mask, mask_last_tp, time_encoding, adj)
            epsilon_mu = epsilon_mu.permute(1, 2, 0)
            epsilon_sigma = epsilon_sigma.permute(1, 2, 0)
            noise = epsilon_mu + torch.randn_like(epsilon_sigma) * epsilon_sigma
            if self.at != 0:
                bt = self.bt * delta_t.unsqueeze(-1)
                at = 1 - bt ** 2
                h = at * h + bt * noise
            #                h = self.at * h + self.bt * noise
            else:
                diffusion_step = torch.squeeze(delta_t // (1 / self.dt)).long()
                sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[diffusion_step]
                sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[diffusion_step]
                h = sqrt_alphas_cumprod.view(-1, 1, 1) * h + sqrt_one_minus_alphas_cumprod.view(-1, 1, 1) * noise

        #		  sqrt_one_minus_alphas_cumprod
        x_gates = torch.cat([x, h], dim=1)
        # r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        # x_c = x
        x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # x_residual = self.residual(x.unsqueeze(-1))
        #        return h + mask * c
        return u * h + (1. - u) * c

class CGRNN_cell_wo_interval(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh', at=0, bt=0):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(CGRNN_cell_wo_interval, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        # self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        # self.residual = nn.Conv2d(d_in, num_units, kernel_size=1, stride=1)

    def forward(self, x, delta_t, h, adj, mask, mask_last_tp, time_encoding, step):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape
        x_gates = torch.cat([x, h], dim=1)
        # r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        # x_c = x
        x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # x_residual = self.residual(x.unsqueeze(-1))
        #        return h + mask * c
        return u * h + (1. - u) * c

class CGRNN_wo_interval(nn.Module):
    def __init__(self, d_in, d_model, d_out, n_layers, support_len, num_nodes, kernel_size=2, at=0, bt=0):
        super(CGRNN_wo_interval, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(
                CGRNN_cell_wo_interval(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                       num_units=self.d_model, support_len=self.support_len, order=self.ks, at=at,
                                       bt=bt))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj, mask, mask_last_tp, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj, mask, mask_last_tp, time_encoding, step)

        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(int(torch.max(lengths))):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
            # return output[-1]
            # else:
            delta_t = delta_t_tensor[:, step].unsqueeze(-1)
            # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
            x_input = x[..., step]
            mask_input_last_tp = mask[..., step - 1] if step > 0 else mask[..., 0]
            mask_input = mask[..., step] if step > 0 else mask[..., 0]
            time_encoding_input = time_encoding[step - 1] if step > 0 else time_encoding[0]
            out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input,
                                      step)
            output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
            # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output