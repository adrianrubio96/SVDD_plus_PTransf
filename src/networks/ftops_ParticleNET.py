import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

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

def to_ptrapphim(x, return_mass=True, eps=1e-8):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)

def pairwise_lv_fts_simple(xi, xj, eps=1e-8):
    xij = xi + xj
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None).split((1, 1, 1), dim=1)
    deltaij = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndeltaij = torch.log(deltaij.clamp(min=eps))

    ptmin = torch.minimum(pti, ptj)
    lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
    lnm2 = torch.log(to_m2(xij, eps=eps))

    return torch.cat([lndeltaij, lnm2], dim=1)
    # return torch.cat([lndeltaij, lnz], dim=1)
    # return torch.cat([lnm2, lnz], dim=1)
    
class PairEmbed(nn.Module):
    def __init__(self, dims, normalize_input=True, eps=1e-8):
        super().__init__()

        self.pairwise_lv_fts = partial(pairwise_lv_fts_simple, eps=eps)

        input_dim = 2
        module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
        for dim in dims:
            module_list.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

        self.out_dim = dims[-1]

    def forward(self, x):
        # x: (batch, v_dim, seq_len)
        with torch.no_grad():
            batch_size, _, seq_len = x.size()
            i, j = torch.tril_indices(seq_len, seq_len, device=x.device)
            x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
            xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
            xj = x[:, :, j, i]
            x = self.pairwise_lv_fts(xi, xj)

        elements = self.embed(x)  # (batch, embed_dim, num_elements)
        y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=x.device)
        y[:, :, i, j] = elements
        y[:, :, j, i] = elements
        
        return y


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, in_feat, out_feats, batch_norm=True, activation=True, pair_embed_dim=None, attention_dims=None):
        super(EdgeConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)

        self.pair_conv = nn.Conv2d(pair_embed_dim, in_feat, kernel_size=1) if pair_embed_dim is not None else None

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

        if attention_dims is not None:
            module_list = []
            in_dim = out_feats[-1]
            for idx, out_dim in enumerate(attention_dims):
                module_list.extend([nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),
                                   ])
                in_dim = out_dim
            module_list.extend([nn.Conv2d(in_dim, 1, kernel_size=1)])

            self.attention = nn.Sequential(*module_list)
        else:
            self.attention = None


    def forward(self, features, pair_embed=None):
        # Set up message indices
        message_indices = torch.arange(features.shape[-1], device=features.device)[None, :].repeat(features.shape[-1], 1)
        message_indices[message_indices >= torch.arange(features.shape[-1], device=features.device)[:, None]] += 1
        message_indices = message_indices[:,:-1][None, None, :, :].expand(features.shape[0], features.shape[1], -1, -1)
        
        # Set up message
        x_features = features.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]-1)
        x_neighbors = torch.gather(features.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]).transpose(-1, -2), -1, message_indices).contiguous()

        if pair_embed is not None:
            x_pair = self.pair_conv(pair_embed)
            x_pair = torch.gather(x_pair, -1, message_indices)
            x_neighbors += x_pair

        # Not sure if this contiguous is needed
        x = torch.cat((x_features, x_neighbors - x_features), 1).contiguous()

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        if self.attention is not None:
            w = torch.nn.functional.softmax(self.attention(x), dim=-1).permute(0,2,3,1)
            fts = torch.matmul(x.permute(0,2,1,3), w).squeeze().permute(0,2,1)
        else:
            fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):
#class ParticleNet(BaseNet):

    def __init__(self, **kwargs):
        
        #super(ParticleNet, self).__init__(**kwargs)
        super().__init__()
        input_dim = kwargs['input_dim']
        
        conv_params = kwargs['conv_params']
        fc_params = kwargs['fc_params']
        aux_dim = None if kwargs['aux_dim'] == 'None' else kwargs['aux_dim']
        aux_fc_params = kwargs['aux_fc_params']
        pair_embed_dims = kwargs['pair_embed_dims']
        attention_dims = kwargs['attention_dims']
        for_segmentation=False

        # Used internally
        self.rep_dim = kwargs['training']['rep_dim']
        self.use_fts_bn = True
        self.use_fusion = True
        self.use_counts = True
        
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dim)

        self.edge_convs = nn.ModuleList()
        for idx, channels in enumerate(conv_params):
            in_feat = input_dim if idx == 0 else conv_params[idx - 1][-1]
            pair_embed_dim = pair_embed_dims[-1] if pair_embed_dims is not None else None
            self.edge_convs.append(EdgeConvBlock(in_feat=in_feat, out_feats=channels, pair_embed_dim=pair_embed_dim, attention_dims=attention_dims))

        if self.use_fusion:
            in_chn = sum(x[-1] for x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels, bias=False), nn.ReLU(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(fc_params[-1][0], self.rep_dim, bias=False))
        self.fc = nn.Sequential(*fcs)

        self.pair_embed = PairEmbed(pair_embed_dims) if pair_embed_dims is not None else None

        if aux_dim is not None:
            aux_fcs = []
            in_dim = aux_dim
            for out_dim, drop_rate in aux_fc_params:
                aux_fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            out_dim = out_chn if self.use_fusion else conv_params[-1][1][-1]
            aux_fcs.append(nn.Linear(in_dim, out_dim, bias=False))
            self.aux_fc = nn.Sequential(*aux_fcs)
        else:
            self.aux_fc = None

    def forward(self, features, vectors=None, mask=None, aux=None):
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        vectors *= mask
        features *= mask
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.pair_embed is not None:
            pair_embedding = self.pair_embed(vectors)

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            if self.pair_embed is not None:
                fts = conv(fts, pair_embed=pair_embedding) * mask
            else:
                fts = conv(fts) * mask
            
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask
        
        if self.use_counts:
            x = fts.sum(dim=-1) / counts  # divide by the real counts
        else:
            x = fts.mean(dim=-1)

        if self.aux_fc is not None and aux is not None:
            x += self.aux_fc(aux)

        output = self.fc(x)

        return output