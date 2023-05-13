import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from spatial_conv import *

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        # self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        # h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 2

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                # block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                #Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        # self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                # block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                #Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        # x:[batch, channel, h, w], time:[batch]
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            # x = block2(x, t)
            # x = attn(x)
            # h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        # x = self.mid_block2(x, t)

        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim = 1)
            # x = block2(x, t)
            # x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class GCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)

        self.decay_weight = nn.Parameter(torch.randn(1, self.num_nodes))
        self.decay_bias = nn.Parameter(torch.randn(1, self.num_nodes))

        self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)

    def forward(self, x, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update



        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        return u * h + (1. - u) * c



class GCGRUCell_time(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_time, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units

        encoder_layers = TransformerEncoderLayer(num_units, 1, 64, 0.1)
        self.epsilon = TransformerEncoder(encoder_layers, 2)

        # self.epsilon = SpatialConvOrderK(c_in=num_units, c_out=num_units, support_len=support_len,
        #                                      order=order)
        # self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
        #                                      order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        # self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)

    def forward(self, x, delta_t, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape

        epsilon = self.epsilon(h.permute(2, 0, 1)).permute(1, 2, 0)
        # epsilon = self.epsilon(h, adj)

        h = torch.sqrt(1 - delta_t).unsqueeze(-1) * h - torch.sqrt(delta_t).unsqueeze(-1) * epsilon
        x_gates = torch.cat([x, h], dim=1)
        # r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        # x_c = x
        x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # return u * h + (1. - u) * c
        return u * h + (1. - u) * c



class GCGRUCell_time_mask(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_time_mask, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units
        self.decay_weight = nn.Parameter(torch.randn(1, self.num_nodes))
        self.decay_bias = nn.Parameter(torch.randn(1, self.num_nodes))
        self.zeros = torch.zeros([1, self.num_nodes]).to("cuda:0")
        self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)

    def forward(self, x, delta_t, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape
        gamma_h = torch.exp(-torch.max(self.zeros, (delta_t * self.decay_weight + self.decay_bias)))
        gamma_h = torch.repeat_interleave(gamma_h, repeats=self.num_units, dim=0).reshape(B, self.num_units, num_nodes)

        h_update = h[:B]
        h_update = gamma_h * h_update

        x_gates = torch.cat([x, h_update], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h_update], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # out_h = torch.cat([(u * h_update + (1. - u) * c), h[B:]], dim=0)
        return u * h_update + (1. - u) * c

class GCRNN_time(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_time, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_time(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))
        self.output_layer = nn.Conv2d(self.d_model, self.d_out, kernel_size=1)

    def init_hidden_states(self, x):
        return [torch.randn(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj)
        return out, h

    def forward(self, x, adj, lengths, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        output = []
        for step in range(steps):
            if step == 0:
                delta_t = torch.zeros([batch, 1]).to(x.device)
            else:
                delta_t = (x[:, 2, 0, step] - x[:, 2, 0, step - 1]).unsqueeze(-1)

            input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)

            out, h = self.single_pass(input, delta_t, h, adj)
            output.append(out)
        return torch.stack(output).permute(1, 3, 2, 0)

class GCRNN_time_origin(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_time_origin, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_time(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))
        self.output_layer = nn.Conv2d(self.d_model, self.d_out, kernel_size=1)

    def init_hidden_states(self, x):
        return [torch.randn(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj)
        return out, h

    def forward(self, x, adj, lengths, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        output = []
        for step in range(steps):
            if step == 0:
                delta_t = torch.zeros([batch, 1]).to(x.device)
            else:
                delta_t = (x[:, 2, 0, step] - x[:, 2, 0, step - 1]).unsqueeze(-1)

            # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)

            out, h = self.single_pass(x[..., step], delta_t, h, adj)
            output.append(out)
        return torch.stack(output).permute(1, 3, 2, 0)
        # return self.output_layer(out[..., None])

class GCRNN_time_origin_mask(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_time_origin_mask, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_time_mask(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))
        self.output_layer = nn.Conv2d(self.d_model, self.d_out, kernel_size=1)

    def init_hidden_states(self, x):
        return [torch.randn(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj):
        B = x.shape[0]
        if B == 0:
            return h[-1], h
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = layer(out, delta_t, h[l], adj)
            h[l] = torch.cat([out, h[l][B:]], dim=0)
        return h[-1], h

    def forward(self, x, adj, lengths, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        x = x[torch.argsort(lengths, descending=True)]
        output = []
        for step in range(steps):
            mini_batch_size = torch.sum(lengths >= step)
            if mini_batch_size == 0:
                output.append(out)
            else:
                if step == 0:
                    delta_t = torch.zeros([mini_batch_size, 1]).to(x.device)
                else:
                    delta_t = (x[:mini_batch_size, 2, 0, step] - x[:mini_batch_size, 2, 0, step - 1]).unsqueeze(-1)

                # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)

                out, h = self.single_pass(x[:mini_batch_size, :, :, step], delta_t, h, adj)
                output.append(out)

        return torch.stack(output).permute(1, 3, 2, 0)
        # return self.output_layer(out[..., None])

class GCRNN_Unet(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_Unet, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_time_Unet(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))
        self.output_layer = nn.Conv2d(self.d_model, self.d_out, kernel_size=1)

    def init_hidden_states(self, x):
        return [torch.randn(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj)
        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        output = []
        for step in range(steps):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
            # else:
            delta_t = delta_t_tensor[:, step].unsqueeze(-1)
            input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
            out, h = self.single_pass(input, delta_t, h, adj)
            output.append(out)
        return torch.stack(output).permute(1, 3, 2, 0)

class GCRNN_transformer(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_transformer, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_time(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))
        self.output_layer = nn.Conv2d(self.d_model, self.d_out, kernel_size=1)

    def init_hidden_states(self, x):
        return [torch.randn(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj)
        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        output = []
        for step in range(steps):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
                # return output[-1]
            # else:
            delta_t = delta_t_tensor[:, step].unsqueeze(-1)
            # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
            input = x[..., step]
            out, h = self.single_pass(input, delta_t, h, adj)
            output.append(out)
        return torch.stack(output).permute(1, 3, 2, 0)
        # return output[-1]
        
class epsilon(nn.Module):
    def __init__(self, d_model, num_variables, device='cuda:0'):
        super().__init__()
        self.num_variables = num_variables
        self.device = device
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=d_model, dropout=0.3)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        transformer_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=d_model, dropout=0.3)
        self.transformer2 = nn.TransformerEncoder(transformer_layer2, num_layers=1)
        #self.gate = SpatialConvOrderK(c_in=d_model, c_out=d_model, support_len=1, order=3)
        self.embed_layer = nn.Embedding(
            num_embeddings=num_variables, embedding_dim=d_model
        )
    def forward(self, h, mask, mask_last_tp, time_encoding, adj):
        num_variable, batch, hidden_dim = h.shape
        feature_embed = self.embed_layer(
            torch.arange(self.num_variables).to(self.device)
        )  # (K,emb)
#        transformer_input = h
#        transformer_input = h + torch.repeat_interleave(
#            feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask.permute(2, 0, 1)
        # output = self.transformer3(h)
        # transformer_input = output + torch.repeat_interleave(time_encoding.unsqueeze(0), num_variable, dim=0) * mask.permute(2, 0, 1)
#        output_1 = self.transformer(transformer_input)
        #output_1 = self.activation(self.transformer(h))
        # transformer_input = h
        # transformer_input = h + torch.repeat_interleave(
        #     feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask.permute(2, 0, 1)
#        output_1 = h

#        transformer_input_2 = output_1
        transformer_input_2 = h + torch.repeat_interleave(
            feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask_last_tp.permute(2, 0, 1)
        transformer_output = self.transformer2(transformer_input_2)
#        transformer_output = transformer_output + self.time_encoding_conv(time_encoding.unsqueeze(1).unsqueeze(-1))\
#            .squeeze(-1).permute(1, 0, 2)
        # transformer_output = self.transformer2(transformer_input)
        # output = self.gate(transformer_output.permute(1, 2, 0), adj).permute(2, 0, 1)
        # transformer_output = self.transformer_layer2(transformer_input)
        return F.tanh(transformer_output)
class epsilon_2channel(nn.Module):
    def __init__(self, d_model, num_variables, device='cuda:0'):
        super().__init__()
        self.num_variables = num_variables
        self.device = device
        self.linear1 = nn.Linear(d_model, d_model // 2)
        self.linear2 = nn.Linear(d_model, d_model // 2)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model // 2, nhead=1, dim_feedforward=d_model, dropout=0.3)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        transformer_layer2 = nn.TransformerEncoderLayer(d_model=d_model // 2, nhead=1, dim_feedforward=d_model, dropout=0.3)
        self.linear3 = nn.Linear(d_model, d_model)
        self.transformer2 = nn.TransformerEncoder(transformer_layer2, num_layers=1)
        #self.gate = SpatialConvOrderK(c_in=d_model, c_out=d_model, support_len=1, order=3)
        self.embed_layer = nn.Embedding(
            num_embeddings=num_variables, embedding_dim=d_model // 2
        )
        self.activation = nn.ReLU()

    def forward(self, h, mask, mask_last_tp, time_encoding, adj):
        num_variable, batch, hidden_dim = h.shape
        feature_embed = self.embed_layer(
            torch.arange(self.num_variables).to(self.device)
        )  # (K,emb)
        h1 = self.linear1(h)
        h2 = self.linear1(h)
        transformer_input1 = h1 + torch.repeat_interleave(
            feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask_last_tp.permute(2, 0, 1)
        transformer_ouput1 = self.transformer(transformer_input1)
        transformer_input2 = h2 + torch.repeat_interleave(
            feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask.permute(2, 0, 1)
        transformer_ouput2 = self.transformer(transformer_input2)
        return self.linear3(torch.cat([transformer_ouput1, transformer_ouput2], dim=-1))
#class epsilon(nn.Module):
#    def __init__(self, d_model, num_variables, device='cuda:0'):
#        super().__init__()
#        self.num_variables = num_variables
#        self.device = device
#        self.conv = nn.Conv2d(1, num_variables, 1)
#        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=d_model, dropout=0.3)
#        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
#        transformer_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=d_model, dropout=0.3)
#        self.transformer2 = nn.TransformerEncoder(transformer_layer2, num_layers=1)
#        #self.gate = SpatialConvOrderK(c_in=d_model, c_out=d_model, support_len=1, order=3)
#        self.embed_layer = nn.Embedding(
#            num_embeddings=num_variables, embedding_dim=d_model
#        )
#        self.activation = nn.ReLU()
#    def forward(self, h, mask, time_encoding, adj):
#        num_variable, batch, hidden_dim = h.shape
#        feature_embed = self.embed_layer(
#            torch.arange(self.num_variables).to(self.device)
#        )  # (K,emb)
#        time_encoding = self.conv(time_encoding.unsqueeze(1).unsqueeze(-1))
#        time_encoding = torch.squeeze(time_encoding.permute(1, 0, 2, 3))
#        # transformer_input = h + torch.repeat_interleave(time_encoding.unsqueeze(0), num_variable, dim=0)
#        # output_1 = self.transformer(transformer_input)
#        output_1 = self.transformer(h + time_encoding)
#        # transformer_input = h
#        # transformer_input = h + torch.repeat_interleave(
#        #     feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask.permute(2, 0, 1)
#        # output_1 = h
#        transformer_input_2 = output_1 + torch.repeat_interleave(
#            feature_embed.unsqueeze(0), batch, dim=0).permute(1, 0, 2) * mask.permute(2, 0, 1)
#        transformer_output = self.transformer2(transformer_input_2)
#        # transformer_output = self.transformer2(transformer_input)
#        # output = self.gate(transformer_output.permute(1, 2, 0), adj).permute(2, 0, 1)
#        # transformer_output = self.transformer_layer2(transformer_input)
#        return transformer_output
class GCGRUCell_epsilon(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_epsilon, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units

        self.epsilon = epsilon(d_model=num_units, num_variables=num_nodes)
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
            epsilon = self.epsilon(h.permute(2, 0, 1), mask, mask_last_tp, time_encoding, adj).permute(1, 2, 0)
            # epsilon = self.epsilon(h, adj)

            h = torch.sqrt(1 - delta_t).unsqueeze(-1) * h - torch.sqrt(delta_t).unsqueeze(-1) * epsilon
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


class GCGRUCell_epsilon_exact(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_epsilon_exact, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units

        self.epsilon = epsilon(d_model=num_units, num_variables=num_nodes)
        self.update_gate = SpatialConvOrderK_exact(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        # self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK_exact(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        # self.residual = nn.Conv2d(d_in, num_units, kernel_size=1, stride=1)
    def forward(self, x, delta_t, h, adj, mask, time_encoding, step):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape

        if step != 0:
            epsilon = self.epsilon(h.permute(2, 0, 1), mask, time_encoding, adj).permute(1, 2, 0)
            # epsilon = self.epsilon(h, adj)

            h = torch.sqrt(1 - delta_t).unsqueeze(-1) * h - torch.sqrt(delta_t).unsqueeze(-1) * epsilon
        x_gates = torch.cat([x, h], dim=1)
        # r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        # x_c = x
        x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # x_residual = self.residual(x.unsqueeze(-1))
        # return u * h + (1. - u) * c
        return u * h + (1. - u) * c * mask

class GCRNN_epsilon_early_stop_exact(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop_exact, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon_exact(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj, mask, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj, mask, time_encoding, step)
        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(steps):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
                # return output[-1]
            # else:
            if step < torch.max(lengths):
                delta_t = delta_t_tensor[:, step].unsqueeze(-1)
                # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
                x_input = x[..., step]
                mask_input = mask[..., step - 1] if step > 0 else mask[..., 0]
                time_encoding_input = time_encoding[step - 1] if step > 0 else time_encoding[0]
                out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, time_encoding_input, step)
                output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output

class GCGRUCell_wo_epsilon(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_wo_epsilon, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units

        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
    def forward(self, x, delta_t, h, adj, mask, time_encoding, step):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape
        x_gates = torch.cat([x, h], dim=1)
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        return u * h + (1. - u) * c

class GCGRUCell_epsilon_wo_variable_attention(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_epsilon_wo_variable_attention, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units

        self.epsilon = epsilon(d_model=num_units, num_variables=num_nodes)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        # self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        # self.residual = nn.Conv2d(d_in, num_units, kernel_size=1, stride=1)
    def forward(self, x, delta_t, h, adj, mask, time_encoding, step):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape

        if step != 0:
            epsilon = self.epsilon(h.permute(2, 0, 1), mask, time_encoding, adj).permute(1, 2, 0)
            # epsilon = self.epsilon(h, adj)

            h = torch.sqrt(1 - delta_t).unsqueeze(-1) * h - torch.sqrt(delta_t).unsqueeze(-1) * epsilon
            #h = torch.sqrt(1 - delta_t).unsqueeze(-1) * h + torch.sqrt(delta_t).unsqueeze(-1) * epsilon
        x_gates = torch.cat([x, h], dim=1)
        # r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        # x_c = x
        x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # x_residual = self.residual(x.unsqueeze(-1))
        # return u * h + (1. - u) * c
        return u * h + (1. - u) * c

class GCRNN_epsilon_early_stop(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

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
           out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input, step)
           output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
           # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output
class GCGRUCell_epsilon_density(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_epsilon_density, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units

        self.epsilon = epsilon(d_model=num_units, num_variables=num_nodes)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        # self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        # self.residual = nn.Conv2d(d_in, num_units, kernel_size=1, stride=1)
    def forward(self, x, density, delta_t, h, adj, mask, time_encoding, step):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape

        if step != 0:
            epsilon = self.epsilon(h.permute(2, 0, 1), mask, time_encoding, adj).permute(1, 2, 0)
            # epsilon = self.epsilon(h, adj)

            h = torch.sqrt(1 - delta_t).unsqueeze(-1) * h - torch.sqrt(delta_t).unsqueeze(-1) * epsilon
        x_gates = torch.cat([density.unsqueeze(1) * x, h], dim=1)
        # r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        # x_c = x
        x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # x_residual = self.residual(x.unsqueeze(-1))
        # return u * h + (1. - u) * c
        return u * h + (1. - u) * c

class GCRNN_epsilon_early_stop_density(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop_density, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon_density(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, density, delta_t, h, adj, mask, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, density, delta_t, h[l], adj, mask, time_encoding, step)
        return out, h

    def forward(self, x, density, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(steps):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
                # return output[-1]
            # else:
            if step < torch.max(lengths):
                delta_t = delta_t_tensor[:, step].unsqueeze(-1)
                # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
                x_input = x[..., step]
                mask_input = mask[..., step - 1] if step > 0 else mask[..., 0]
                time_encoding_input = time_encoding[step - 1] if step > 0 else time_encoding[0]
                out, h = self.single_pass(x_input, density, delta_t, h, adj, mask_input, time_encoding_input, step)
                output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output
class GCRNN_wo_epsilon_early_stop(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_wo_epsilon_early_stop, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_wo_epsilon(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj, mask, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj, mask, time_encoding, step)
        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(steps):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
                # return output[-1]
            # else:
            if step < torch.max(lengths):
                delta_t = delta_t_tensor[:, step].unsqueeze(-1)
                # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
                x_input = x[..., step]
                mask_input = mask[..., step - 1] if step > 0 else mask[..., 0]
                time_encoding_input = time_encoding[step - 1] if step > 0 else time_encoding[0]
                out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, time_encoding_input, step)
                output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output

class GCRNN_epsilon_early_stop_wo_variable_attention(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop_wo_variable_attention, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj, mask, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj, mask, time_encoding, step)
        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(steps):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
                # return output[-1]
            # else:
            if step < torch.max(lengths):
                delta_t = delta_t_tensor[:, step].unsqueeze(-1)
                # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
                x_input = x[..., step]
                mask_input = mask[..., step - 1] if step > 0 else mask[..., 0]
                time_encoding_input = time_encoding[step - 1] if step > 0 else time_encoding[0]
                out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, time_encoding_input, step)
                output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output
class GCRNN_epsilon_early_stop_mask_te(nn.Module):
    def __init__(self, d_in, d_model, d_out, n_layers, support_len, num_nodes, kernel_size=2):
        super(GCRNN_epsilon_early_stop_mask_te, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        self.input_projection = nn.Conv2d(d_in + d_in // 4 + 1, d_in, kernel_size=1)
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, delta_t, h, adj, mask, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, delta_t, h[l], adj, mask, time_encoding, step)
        return out, h

    def forward(self, x, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(steps):
            # if step > torch.max(lengths):
            #     output.append(torch.zeros_like(h[0]))
                # return output[-1]
            # else:
            if step < torch.max(lengths):
                delta_t = delta_t_tensor[:, step].unsqueeze(-1)
                # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
                x_input = x[..., step]
                mask_input = mask[..., step]
                mask_input_last = mask[..., step - 1] if step > 0 else mask[..., 0]
                time_encoding_input = time_encoding[step]
                time_encoding_input_last = time_encoding[step - 1] if step > 0 else time_encoding[0]
                x_input = self.input_projection(torch.cat([x_input, mask_input, torch.repeat_interleave(time_encoding_input.unsqueeze(-1), nodes, dim=-1)],
                                    dim=1).unsqueeze(-1)).squeeze(-1)
                out, h = self.single_pass(x_input, delta_t, h, adj, mask_input_last, time_encoding_input_last, step)
                output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output
class GCGRUCell_epsilon_var_length(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_epsilon_var_length, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units

        self.epsilon = epsilon(d_model=num_units, num_variables=num_nodes)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        # self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        # self.residual = nn.Conv2d(d_in, num_units, kernel_size=1, stride=1)
    def forward(self, x, var_need_update, delta_t, h, adj, mask, time_encoding, step):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        B, input_dim, num_nodes = x.shape

        if step != 0:
            epsilon = self.epsilon(h.permute(2, 0, 1), mask, time_encoding, adj).permute(1, 2, 0)
            # epsilon = self.epsilon(h, adj)

            h = torch.sqrt(1 - delta_t).unsqueeze(-1) * h - torch.sqrt(delta_t).unsqueeze(-1) * epsilon
        cur_h = h[:, :, var_need_update]
        cur_x = x[:, :, var_need_update]
        cur_adj = adj[var_need_update]
        cur_adj = cur_adj[:, var_need_update]
        x_gates = torch.cat([cur_x, cur_h], dim=1)
        # r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, cur_adj))
        # x_c = x
        # x_c = torch.cat([x, h], dim=1)
        c = self.c_gate(x_gates, cur_adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        # x_residual = self.residual(x.unsqueeze(-1))
#        return h + mask * c
        update_h = u * cur_h + (1. - u) * c
        h[:, :, var_need_update] = update_h
        return h

class GCRNN_epsilon_early_stop_var_length(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop_var_length, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon_var_length(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, var_need_update, delta_t, h, adj, mask, time_encoding, step):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, var_need_update, delta_t, h[l], adj, mask, time_encoding, step)
        return out, h

    def forward(self, x, var_length, var_last_obs_tp, adj, delta_t_tensor, lengths, time_encoding, mask, h=None):
        # x:[batch, features, nodes, steps]
        batch, features, nodes, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        # output = []
        output = torch.zeros_like(h[0])
        for step in range(0, int(torch.max(lengths))):
            var_need_update = torch.where(var_last_obs_tp >= step)[0]
            if var_need_update.shape[0] == 0:
                return output
            delta_t = delta_t_tensor[:, step].unsqueeze(-1)
            # input = torch.cat([x[..., step], torch.repeat_interleave(delta_t, self.num_nodes, dim=-1).unsqueeze(1)], dim=1)
            x_input = x[:, :, :, step]
            mask_input = mask[:, :, :, step - 1] if step > 0 else mask[:, :, :, 0]
            time_encoding_input = time_encoding[step - 1] if step > 0 else time_encoding[0]
            out, h = self.single_pass(x_input, var_need_update, delta_t, h, adj, mask_input, time_encoding_input, step)
            output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
        return output
class epsilon_distribution(nn.Module):
    def __init__(self, d_model, num_variables, device='cuda:0'):
        super().__init__()
        self.num_variables = num_variables
        self.device = device
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=d_model, dropout=0.3)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        transformer_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=d_model, dropout=0.3)
        self.transformer2 = nn.TransformerEncoder(transformer_layer2, num_layers=1)
        #self.gate = SpatialConvOrderK(c_in=d_model, c_out=d_model, support_len=1, order=3)
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
class GCGRUCell_epsilon_distribution(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_epsilon_distribution, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units
        beta_start = 0.0001
        beta_end = 0.0002
        self.dt = 100
        print(beta_end, ' ', beta_start)
#        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, 1000).cuda()
        betas = torch.linspace(beta_end ** 0.5, beta_start ** 0.5, self.dt+1).cuda()
        print(betas.shape[0])
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
            # epsilon = self.epsilon(h, adj)
            epsilon_mu = epsilon_mu.permute(1, 2, 0)
            epsilon_sigma = epsilon_sigma.permute(1, 2, 0)
#            diffusion_step = torch.squeeze(delta_t // (1 / self.dt)).long()
#            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[diffusion_step]
#            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[diffusion_step]
            noise = epsilon_mu + torch.randn_like(epsilon_sigma) * epsilon_sigma
#            h = sqrt_alphas_cumprod.view(-1, 1, 1) * h + sqrt_one_minus_alphas_cumprod.view(-1, 1, 1) * noise
#		  sqrt_one_minus_alphas_cumprod
            h = (1 - 1e-6) * h + 1e-3 * noise
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
class GCRNN_epsilon_early_stop_distribution(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop_distribution, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon_distribution(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

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
            out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input, step)
            output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output

class GCRNN_epsilon_early_stop_distribution_sumoutput(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop_distribution_sumoutput, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon_distribution(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

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
        sum_output = torch.zeros_like(h[0])
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
            out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input, step)
            sum_output = sum_output * out
            output[torch.where(step == (lengths - 1))] = sum_output[torch.where(step == (lengths - 1))]
#            output[torch.where(step == (lengths - 1))] = sum_output[torch.where(step == (lengths - 1))] / lengths[torch.where(step == (lengths - 1))].view(-1, 1, 1)
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output
class GCGRUCell_epsilon_distribution_wo_diffusion(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, num_nodes=36, order=3, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell_epsilon_distribution_wo_diffusion, self).__init__()
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

class GCRNN_epsilon_early_stop_distribution_wo_diffusion(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 num_nodes,
                 kernel_size=2):
        super(GCRNN_epsilon_early_stop_distribution_wo_diffusion, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.num_nodes = num_nodes
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell_epsilon_distribution_wo_diffusion(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))

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
            out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input, step)
            output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output
class CGRNN(nn.Module):
    def __init__(self, d_in, d_model, d_out, n_layers, support_len, num_nodes, kernel_size=2, at=0, bt=0):
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
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks, at=at, bt=bt))

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
            out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input, step)
            output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output

class CGRNN_cell(nn.Module):
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
        super(CGRNN_cell, self).__init__()
        self.num_nodes = num_nodes
        self.activation_fn = getattr(torch, activation)
        self.num_units = num_units
        beta_start = 1e-5
        beta_end = 2e-5
        print(beta_start, beta_end)
        self.dt = 100
        self.at = at
        self.bt = bt
        if at == 0:
            # print(beta_end, ' ', beta_start)
    #        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, 1000).cuda()
            betas = torch.linspace(beta_end ** 0.5, beta_start ** 0.5, self.dt+1).cuda()
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
            self.rnn_cells.append(CGRNN_cell_wo_interval(d_in=self.d_in if i == 0 else self.d_model, num_nodes=num_nodes,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks, at=at, bt=bt))

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
            out, h = self.single_pass(x_input, delta_t, h, adj, mask_input, mask_input_last_tp, time_encoding_input, step)
            output[torch.where(step == (lengths - 1))] = out[torch.where(step == (lengths - 1))]
                # output.append(out + torch.squeeze(self.residual(x_input.unsqueeze(-1))))
        # return torch.stack(output).permute(1, 3, 2, 0)
        return output


        