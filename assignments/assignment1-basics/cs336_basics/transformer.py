import torch
import torch.nn as nn
from einops import rearrange, einsum
# y = Wx
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        sigma = torch.sqrt(torch.tensor(2.0 / (input_dim + output_dim)))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x):
        return einsum(self.weight, x, 'o i, ... i -> ... o' )
        # return torch.einsum('oi, ...i -> ...o', self.weight, x)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return torch.einsum('...d, d -> ...d', x/rms, self.scale).to(in_type)
    

class SwiGLU(nn.Module): # W2((W1x * sigmoid(W1x)) * W3x)
    def __init__(self, d_model: int, d_ff: int):
        super(SwiGLU, self).__init__()
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model))
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff))
        self.w3 = nn.Parameter(torch.randn(d_ff, d_model))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = torch.einsum('oi, ...i -> ...o', self.w1, x)
        silu = x1 * self.sigmoid(x1)
        x3 = torch.einsum('oi, ...i -> ...o', self.w3, x)
        x = torch.einsum('...i, ...i -> ...i', silu, x3)
        return torch.einsum('oi, ...i -> ...o', self.w2, x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        angles = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))  # (d_k/2,)
        # angles = torch.arange(max_seq_len).unsqueeze(1) * angles.unsqueeze(0)  # (seq_len, d_k/2)
        angles = torch.einsum('s, d -> sd', torch.arange(max_seq_len, device=device).float(), angles)  # (seq_len, d_k/2)
        cos_matrix = torch.cos(angles)
        sin_matrix = torch.sin(angles)
        self.register_buffer('cos_matrix', cos_matrix, persistent=False)
        self.register_buffer('sin_matrix', sin_matrix, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_matrix[token_positions]  # (seq_len, d_k/2)
        sin = self.sin_matrix[token_positions]  # (seq_len, d_k/2)
        y = torch.zeros_like(x)
        x_0 = x[..., 0::2]
        x_1 = x[..., 1::2]
        y[..., 0::2] = x_0 * cos - x_1 * sin
        y[..., 1::2] = x_0 * sin + x_1 * cos
        return y


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_val = torch.max(x, dim=i, keepdim=True)[0]
    x = x - max_val
    return torch.exp(x) / torch.sum(torch.exp(x), dim=i, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask): #False means need to mask
    d_k = Q.shape[-1]
    scores = einsum(Q, K, '... n_len d_k, ... m_len d_k -> ... n_len m_len')
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None: # 0 * -inf = nan
        scores = scores.masked_fill(mask == False, float('-inf'))
    attn = softmax(scores, -1)
    return einsum(attn, V, '... n_len m_len, ... m_len d_v -> ... n_len d_v')


class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, position_emb = False, theta=None, max_seq_len=None, token_positions=None):
        super(multihead_self_attention, self).__init__()
        self.theta = theta
        self.token_positions = token_positions
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.position_emb = position_emb
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.Q_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.K_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.V_weight =  nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.O_weight = nn.Parameter(torch.randn(d_model, num_heads * self.d_v))
    
    def forward(self, x):
        query = rearrange(self.Q_weight, '(num_heads d_k) d_in -> num_heads d_k d_in', num_heads = self.num_heads, d_k = self.d_k)
        key = rearrange(self.K_weight, '(num_heads d_k) d_in -> num_heads d_k d_in', num_heads = self.num_heads, d_k = self.d_k)
        value = rearrange(self.V_weight, '(num_heads d_v) d_in -> num_heads d_v d_in', num_heads = self.num_heads, d_v = self.d_v)

        query = einsum(query, x, 'num_heads d_k d_in, ... seq_len d_in -> ... num_heads seq_len d_k')
        key = einsum(key, x, 'num_heads d_k d_in, ... seq_len d_in -> ... num_heads seq_len d_k')
        value = einsum(value, x, 'num_heads d_v d_in, ... seq_len d_in -> ... num_heads seq_len d_v')

        if self.position_emb:
            if self.token_positions is None:
                self.token_positions = torch.arange(x.shape[-2], device=x.device)
            rotary_emb = RotaryPositionalEmbedding(self.theta, int(self.d_k), self.max_seq_len)
            query = rotary_emb(query, self.token_positions)
            key = rotary_emb(key, self.token_positions)
        mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2]), device=x.device)).bool()
        attn_output = scaled_dot_product_attention(query, key, value, mask)
        attn_output = rearrange(attn_output, '... num_heads seq_len d_v -> ... seq_len (num_heads d_v)')
        return einsum(self.O_weight, attn_output, 'd_model d_v, ... seq_len d_v -> ... seq_len d_model')



class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, position_emb = False, theta=None, max_seq_len=None):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.mhsa = multihead_self_attention(d_model, num_heads, position_emb, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super(Transformer, self).__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, position_emb=True, theta=rope_theta, max_seq_len=context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.linear(x)
        # prob = softmax(x, -1)
        return x
        # return prob