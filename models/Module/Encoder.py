from torch import nn

from .SubModule.SelfAttention import Attention
from .SubModule.MLP import MLP


class EncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 mlp_ratio,
                 qkv_bias=False,
                 dropout=0.,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, n_heads, qkv_bias,
                              attn_dropout, proj_dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = norm_layer(dim)
        mlp_hid_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hid_dim, dim, dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
