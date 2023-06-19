from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F




class Attentinal_GNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.Computedesc=HardNet128()
        self.kenc = KeypointEncoder(128, [32, 64, 128])
        self.gnn = AttentionalGNN(128, ['self'] * 1)
        self.final_proj = nn.Conv1d( 128, 128, kernel_size=1, bias=True)
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)


    def forward(self, x, key, score, image):

        x=x.reshape(x.shape[1], x.shape[2],x.shape[3])
        inputs = x.view(x.shape[0], 1, 128, 128)
        desc0= self.Computedesc(inputs)
        score=score.reshape(score.shape[1], score.shape[2])
        kpts0= key.reshape(key.shape[1], key.shape[2])
        desc0 = desc0.transpose(0, 1)

        kpts0 = torch.reshape(kpts0, (1, -1, 2))

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, image)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, score)

        # Multi-layer Transformer network.
        desc0 = self.gnn(desc0)

        # Final MLP projection.
        mdesc0 = self.final_proj(desc0)
        return mdesc0

        #-----------------------------------------------------------------------------------------------------------------------

class HardNet128(nn.Module):
        """HardNet model definition
        """

        def __init__(self):
            super(HardNet128, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16, affine=False),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16, affine=False),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32, affine=False),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, affine=False),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=False),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=False),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv2d(128, 128, kernel_size=8, bias=False),
                nn.BatchNorm2d(128, affine=False),
            )

            self.features.apply(weights_init)
            return

        def input_norm(self, x):
            flat = x.view(x.size(0), -1)

            mp = torch.mean(flat, dim=1)
            sp = torch.std(flat, dim=1) + 1e-7
            return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
                -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

        def forward(self, input):
            x_features = self.features(self.input_norm(input))
            x = x_features.view(x_features.size(0), -1)
            return (x)

def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.6)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image):
    """ Normalize keypoints locations based on image image_shape"""

    height, width = image.shape[1:]
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):

        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]


        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names


    def forward(self, desc0):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            src0 = desc0
            delta0= layer(desc0, src0)
            desc0= (desc0 + delta0)
        return desc0


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x
