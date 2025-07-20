from torch import nn


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 n_heads=8,
                 qkv_bias=False,
                 attn_dropout=0.,
                 proj_dropout=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -.5


        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        b, n, c = x.shape       # batch 数，序列长度，token的embedding尺寸
        qkv = (self.qkv(x)
               .reshape(b, n, 3, self.n_heads, c//self.n_heads)
               .permute(2, 0, 3, 1, 4)
               )
        q,k,v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1,2).reshape(b,n,c)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

