import math
import torch
import torch.nn.functional as F

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model, device):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        # Precompute positional encodings for max_len tokens
        pe = torch.zeros(max_len, d_model, device=device)
        for pos in range(max_len):
            for i in range(d_model):
                angle_rate = pos / (10000 ** ((2 * (i // 2)) / d_model))
                if i % 2 == 0:
                    pe[pos, i] = math.sin(angle_rate)
                else:
                    pe[pos, i] = math.cos(angle_rate)
        self.register_buffer("pe", pe)
        self.max_len = max_len

    def forward(self, x):
        # x shape: [B, T, d_model]
        T = x.size(1)
        if T > self.max_len:
            # Generate additional positional encodings for T - self.max_len tokens
            pe_extra = torch.zeros(T - self.max_len, self.d_model, device=self.device)
            for pos in range(self.max_len, T):
                for i in range(self.d_model):
                    angle_rate = pos / (10000 ** ((2 * (i // 2)) / self.d_model))
                    if i % 2 == 0:
                        pe_extra[pos - self.max_len, i] = math.sin(angle_rate)
                    else:
                        pe_extra[pos - self.max_len, i] = math.cos(angle_rate)
            pe_full = torch.cat([self.pe, pe_extra], dim=0)
        else:
            pe_full = self.pe
        return x + pe_full[:T, :]

class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, d_model):
        super(MaskedSelfAttention, self).__init__()
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
    def forward(self, x):
        # x: [B, T, d_model]
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        T = x.size(1)
        # Create a causal mask (upper-triangular matrix with -inf above the diagonal)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        attn_scores = attn_scores + mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V)
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MaskedSelfAttention(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        # Apply masked self-attention and add residual connection
        x = self.norm1(x + self.self_attn(x))
        # Apply feedforward network and add residual connection
        x = self.norm2(x + self.ff(x))
        return x

class MyDecoder(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, d_model, num_layers, d_ff, device):
        super(MyDecoder, self).__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(max_seq_length, d_model, device)
        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, d_ff) for _ in range(num_layers)])
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
    def forward(self, x):
        # x: [B, T] token IDs
        x = self.embed(x)  # [B, T, d_model]
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.fc_out(x)  # [B, T, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
