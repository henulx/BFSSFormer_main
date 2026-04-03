import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from thop import profile
import torch.nn.init as init
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from torchvision.transforms import Compose, Normalize, ToTensor



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Dyt(nn.Module):
    def __init__(self,dim,num_tokens=10):
        super().__init__()
        self.init_a = nn.Parameter(torch.empty(64, 1, (num_tokens+1)))
        self.Alpha = nn.Parameter(torch.ones(64,64,1))
        self.Beta = nn.Parameter(torch.zeros(64,(num_tokens+1),64))
        self.Gamma = nn.Parameter(torch.ones(64,(num_tokens+1),64))

    def forward(self, x):
        x = torch.tanh(torch.einsum('bij,bjk->bik',self.Alpha, x))
        y = torch.einsum('bij,bjk->bik',self.Gamma, x)
        return y + self.Beta

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DecayPos1d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen):
        '''
        slen: (c)
        recurrent is not implemented
        '''
        mask_c = self.generate_1d_decay(slen)
        retention_rel_pos = mask_c

        return retention_rel_pos

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

        self.realPos = DecayPos1d(64, heads, 2, 4)

    def differential_attention(self, scores):##差分计算
        diff_scores = scores[:, 1:] - scores[:, :-1]
        pld = (0,0,0,0,1,0)#torch.Size([64, 8, 5, 5])
        diff_scores = torch.nn.functional.pad(diff_scores, pld, 'constant',0)
        return diff_scores

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3,dim=-1，一般是最后一维。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots1 = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale ##torch.Size([64, 8, 5, 5])
        dots2 = self.differential_attention(dots1) * self.scale
        dots = dots1 - dots2
        #mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        S = dots[:,0] ##torch.Size([64, 5, 5])
        # S = torch.tanh_(S) ##torch.Size([64, 5, 5])
        m = nn.Mish()
        S = m(S)
        # m = nn.Softplus()
        # S = m(S)
        # m = nn.Softshrink()
        # S = m(S)
        # m = nn.Tanhshrink()
        # S = m(S)
        # m = nn.ReLU()
        # S = m(S)
        # m = nn.SiLU()
        # S = m(S)
        # S[...,0] = 0
        # S = (1-torch.eye(n))*S
        S2 = torch.roll(S, 1, -2)
        # S[..., 0, :] = 0
        F1 = torch.cumsum(S2,dim=-2)
        # dots = m(dots)
        F2 = F1[:,None]
        dots = dots - F2
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        # realPos = self.realPos(24 / self.heads)
        # attn = self.add_3d_from_4d_torch(attn,realPos)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,num_channel):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),#64 11 64
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
                # Residual(Dyt(Attention(dim, heads=heads, dropout=dropout))),
                # Residual(Dyt(MLP_Block(dim, mlp_dim, dropout=dropout))),
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0)) #原版
            # self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        last_output = []
        nl = 0
        for attention, mlp in self.layers:
            last_output.append(x)
            if nl > 1:
                x = self.skipcat[nl - 2](torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(
                    3)#跳接
            x = attention(x, mask=mask)  # go to attention
            # x = attention(x)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

# NUM_CLASS = 9
NUM_CLASS = 16
# NUM_CLASS = 15
# NUM_CLASS = 13
# NUM_CLASS = 22

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=10, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1,num_channel=1):
    # def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=3, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1, num_channel=1):
        super(SSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.scale = dim ** -1/2

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3d_features5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(5, 5, 5)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features5 = nn.Sequential(
            nn.Conv2d(in_channels=208, out_channels=64, kernel_size=(5, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.unmix_encoder = nn.Sequential(
            nn.Conv2d(64, 15, kernel_size=(3, 3), stride=1, padding=1),
            # 步幅：卷积核经过输入特征图的采样间隔，希望减小输入参数的数目，减少计算量
            # 填充：填充：在输入特征图的每一边添加一定数目的行列，使得输出的特征图的长、宽 = 输入的特征图的长、宽
            nn.BatchNorm2d(15, affine=True),
            # 在卷积层之后和激活函数之前，BatchNorm2d的主要作用是通过减少内部协变量偏移来加速网络的训练，并提高模型的泛化能力。
            nn.ReLU(),

            nn.Conv2d(15, 7, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(7, affine=True),
            nn.ReLU(),

            nn.Conv2d(7, num_classes, kernel_size=(3, 3), stride=1, padding=1),
            nn.Softmax(dim=1)
        )  # 编码器，包含3个1*1的卷积

        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 3*3大小卷积核，

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout,num_channel)#dim=64,depth=1,heads=8

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):

        y = x
        z = x

        z = self.conv3d_features5(z)
        z = rearrange(z, 'b c h w y -> b (c h) w y')
        z = self.conv2d_features5(z)
        z = rearrange(z, 'b c h w -> b (h w) c')

        y = self.conv3d_features5(y)
        y = rearrange(y, 'b c h w y -> b (c h) w y')
        y = self.conv2d_features5(y)
        y = rearrange(y,'b c h w -> b (h w) c')


        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')

        FusedFeatures = torch.concat((x,z,y), dim=1)

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa) * self.scale
        # A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokenFFs = self.cls_token.expand(FusedFeatures.shape[0],-1,-1)
        FusedFeatures = torch.cat((cls_tokenFFs, T), dim=1)
        FusedFeatures += self.pos_embedding
        FusedFeatures = self.dropout(FusedFeatures)
        FusedFeatures = self.transformer(FusedFeatures, mask)  # main game # torch.Size([64, 5, 64])
        FusedFeatures = self.to_cls_token(FusedFeatures[:, 0])

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game # torch.Size([64, 5, 64])
        x = self.to_cls_token(x[:, 0])


        FusedFeatures = self.nn1(FusedFeatures)
        # x = self.nn1(x)


        return FusedFeatures
        # return x

# def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
#     preprocessing = Compose([
#         ToTensor(),
#         Normalize(mean=mean, std=std)
#     ])
#     return preprocessing(img.copy()).unsqueeze(0)

if __name__ == '__main__':
    model1 = Attention(dim=64)
    model1.eval()
    print(model1)
    model = SSFTTnet()
    model.eval()
    print(model)
    input = torch.randn(60, 1, 30, 19, 19)
    flops, params = profile(model,(input,))
    print('flops: ', str(flops/1024**3) + 'G', 'params: ', str(params/1024**2) + 'M')
    y = model(input)
    print(y.size())

