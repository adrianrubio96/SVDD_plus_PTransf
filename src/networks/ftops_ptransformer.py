import math
import random
import warnings
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v

def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)

def pairwise_lv_fts(xi, xj, idi, idj, sm_int_matrix, eps=1e-8, for_onnx=False):
    xij = xi + xj
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    deltaij = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndeltaij = torch.log(deltaij.clamp(min=eps))

    ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
    lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
    lnm2 = torch.log(to_m2(xij, eps=eps))

    id_int = sm_int_matrix[idi, idj].unsqueeze(1).float()

    return torch.cat([lndeltaij, lnm2, id_int], dim=1)
    # return torch.cat([lndeltaij, lnz], dim=1)
    # return torch.cat([lnm2, lnz], dim=1)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim, bias=False),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    # input_dim is fixed to 3
    def __init__(self, dims, normalize_input=True, activation='gelu', eps=1e-8, for_onnx=False):
        super().__init__()

        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, eps=eps, for_onnx=for_onnx)

        # SM interaction matrix        -  j  jb e- e+ m- m+ g
        #sm_int_matrix = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],  # -
        #                              [0, 1, 1, 0, 0, 0, 0, 1],  # j
        #                              [0, 1, 1, 0, 0, 0, 0, 1],  # jb
        #                              [0, 0, 0, 0, 1, 0, 0, 1],  # e-
        #                              [0, 0, 0, 1, 0, 0, 0, 1],  # e+
        #                              [0, 0, 0, 0, 0, 0, 1, 1],  # m-
        #                              [0, 0, 0, 0, 0, 1, 0, 1],  # m+
        #                              [0, 1, 1, 1, 1, 1, 1, 0]]) # g

        # SM int. matrix for DarkM v11 -  j  jb l g
        sm_int_matrix = torch.tensor([[0, 0, 0, 0, 0],  # -
                                      [0, 1, 1, 0, 1],  # j
                                      [0, 1, 1, 1, 1],  # jb
                                      [0, 0, 1, 1, 1],  # l
                                      [0, 1, 1, 1, 0]]) # g

        self.register_buffer("sm_int_matrix", sm_int_matrix)
        
        input_dim = 3
        module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
        for dim in dims:
            module_list.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

        self.out_dim = dims[-1]

    def forward(self, x, ids):
        # x: (batch, v_dim, seq_len)
        with torch.no_grad():
            batch_size, _, seq_len = x.size()
            if not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, device=x.device)
                x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                ids = ids.unsqueeze(-1).repeat(1, 1, seq_len)
                xi  = x[ :, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                xj  = x[ :, :, j, i]
                idi = ids[:, i, j]
                idj = ids[:, j, i]
                x = self.pairwise_lv_fts(xi=xi, xj=xj, idi=idi, idj=idj, sm_int_matrix=self.sm_int_matrix)
            else:
                # NOT UPDATED FOR NEW PAIRWISE INTERACTIONS
                x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2)).view(batch_size, -1, seq_len * seq_len)

        elements = self.embed(x)  # (batch, embed_dim, num_elements)
        if not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=x.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(batch_size, -1, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True, 
                 add_bias_attn=False, seq_len=-1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
            bias = False
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim, bias=False)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim, bias=False)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

        self.add_bias_attn = add_bias_attn
        self.attn_bias = None
        if add_bias_attn:
            assert(seq_len > 0)
            # self.attn_bias = nn.Parameter(torch.zeros(seq_len, seq_len).float(), requires_grad=True)
            self.attn_bias = nn.Parameter(torch.rand(seq_len, seq_len)*2.-1., requires_grad=True)

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            assert(not ((self.attn_bias is not None) and (attn_mask is not None)))
            if attn_mask is not None:
                x = self.attn(x, x, x, key_padding_mask=padding_mask,
                              attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)
            elif self.attn_bias is not None:
                x = self.attn(x, x, x, key_padding_mask=padding_mask,
                              attn_mask=self.attn_bias.float())[0]  # (seq_len, batch, embed_dim)  
            else:
                x = self.attn(x, x, x, key_padding_mask=padding_mask)[0]  # (seq_len, batch, embed_dim)                 
  

        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x

class ParticleTransformer(BaseNet):

    def __init__(self, **kwargs) -> None:

        super().__init__()

        # Copy kwargs 
        kwargs = dict(kwargs)

        net_name = kwargs['net_name']
        input_dim = kwargs['input_dim']
        aux_dim = kwargs['aux_dim']
        embed_dims = kwargs['embed_dims']
        pair_embed_dims = kwargs['pair_embed_dims']
        num_heads = kwargs['num_heads']
        num_layers = kwargs['num_layers']
        num_cls_layers = kwargs['num_cls_layers']
        block_params = None if kwargs['block_params']=='None' else kwargs['block_params']
        cls_block_params = kwargs['cls_block_params']
        fc_params = kwargs['fc_params']
        aux_fc_params = kwargs['aux_fc_params']
        activation = kwargs['activation']
        add_bias_attn = kwargs['add_bias_attn']
        seq_len = kwargs['seq_len']

        #Used internally
        self.rep_dim = kwargs['training']['rep_dim']
        self.trim = kwargs['trim']
        self.for_inference = kwargs['for_inference']
        self.use_amp = kwargs['use_amp']
        self._counter = 0

        default_cfg = dict(embed_dim=embed_dims[-1], num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True,
                           add_bias_attn=add_bias_attn, seq_len=seq_len)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.embed = Embed(input_dim, embed_dims, activation=activation)
        self.pair_embed = PairEmbed(pair_embed_dims + [cfg_block['num_heads']], for_onnx=self.for_inference) if pair_embed_dims is not None else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dims[-1])

        if aux_dim is not None:
            aux_fcs = []
            in_dim = aux_dim
            for out_dim, drop_rate in aux_fc_params:
                aux_fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            aux_fcs.append(nn.Linear(in_dim, embed_dims[-1], bias=False))
            self.aux_fc = nn.Sequential(*aux_fcs)
        else:
            self.aux_fc = None

        if fc_params is not None:
            fcs = []
            in_dim = embed_dims[-1]
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, self.rep_dim, bias=False))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)
        #self.cls_token = nn.init.xavier_uniform_(self.cls_token, gain=1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, ids=None, mask=None, aux=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            if mask is None:
                mask = torch.ones_like(x[:, :1])
            mask = mask.bool()

            if self.trim and not self.for_inference:
                if self._counter < 5:
                    #print(self._counter)
                    self._counter += 1
                else:
                    self.training = False
                    if self.training:
                        raise ValueError("Should never get here")
                        q = min(1, random.uniform(0.9, 1.02))
                        maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                        rand = torch.rand_like(mask.type_as(x))
                        rand.masked_fill_(~mask, -1)
                        perm = rand.argsort(dim=-1, descending=True)
                        mask = torch.gather(mask, -1, perm)
                        x = torch.gather(x, -1, perm.expand_as(x))
                        if v is not None and ids is not None and self.pair_embed is not None:
                            v = torch.gather(v, -1, perm.expand_as(v))
                    else:
                        maxlen = mask.sum(dim=-1).max()
                    maxlen = max(maxlen, 1)
                    if maxlen < mask.size(-1):
                        mask = mask[:, :, :maxlen]
                        x = x[:, :, :maxlen]
                        if v is not None and ids is not None and self.pair_embed is not None:
                            v = v[:, :, :maxlen]
                            ids = ids[:, :maxlen]
                
                #print("mask", mask.shape)
            padding_mask = ~mask.squeeze(1)  # (N, P)
            #print("padding_mask", padding_mask.shape)

        with torch.cuda.amp.autocast(enabled=self.use_amp):  
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            #print("embeded x", x.shape)
            attn_mask = None
            if v is not None and ids is not None and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, ids).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
                #print("block x", x.shape)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
                #print("cls_tokens", cls_tokens.shape)

            x_cls = self.norm(cls_tokens).squeeze(0)
            #print("x_cls", x_cls.shape)

            if self.aux_fc is not None and aux is not None:
                x_cls += self.aux_fc(aux)

            # fc
            if self.fc is None:
                return x_cls
            #print("fc", self.fc)
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
                
            return output



class FTOPS_Mlp(BaseNet):

    def __init__(self, **kwargs):
        super().__init__()

        self.num_features = kwargs['num_features'] 
        self.rep_dim = kwargs['rep_dim']
        
                
        #self.fc1 = nn.Linear(self.num_features, 8, bias=False)
        #self.fc2 = nn.Linear(8, self.rep_dim, bias=False)

        
        self.fc1 = nn.Linear(self.num_features, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, self.rep_dim, bias=False)
        
        """
        self.fc1 = nn.Linear(self.num_features, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, 8, bias=False)
        self.fc4 = nn.Linear(8, self.rep_dim, bias=False)
        self.do1 = nn.Dropout(0.1)  # 20% Probability
         """

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        #x = self.do1(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        #x = self.do1(x)
        x = self.fc3(x)        
        #x = self.do1(x)
        #x = self.fc4(x)
        return x

class FTOPS_Mlp_Autoencoder(BaseNet):

    def __init__(self, **kwargs):
        super().__init__()

        self.num_features = kwargs['num_features']
        self.rep_dim = kwargs['rep_dim']


        #encoder
        self.fc1 = nn.Linear(self.num_features, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, self.rep_dim, bias=False)
       
        #decoder
        self.fc4 = nn.Linear(self.rep_dim, 16, bias=False)
        self.fc5 = nn.Linear(16, 32, bias=False)
        self.fc6 = nn.Linear(32, self.num_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        x = F.leaky_relu(x)
        x = self.fc6(x)
 
        return x



