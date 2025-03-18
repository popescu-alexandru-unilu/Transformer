# encoder.py
import math
import torch
import torch.nn.functional as F

class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionBlock, self).__init__()
        self.W_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_K = torch.nn.Linear(embed_dim, embed_dim)
        self.W_V = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs):
        # inputs: [B, seq_len, embed_dim]
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V)
        return out

class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocab_size, max_sequence_length=32, embed_dim=64, num_layers=4, device='cpu'):
        super(TransformerEncoder, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim
        self.emb = torch.nn.Embedding(vocab_size, embed_dim)

        # Create positional encoding matrix
        pos_encoding_matrix = torch.zeros((max_sequence_length, embed_dim), device=device)
        for pos in range(max_sequence_length):
            for i in range(embed_dim):
                angle_rate = pos / (10000 ** (2 * (i // 2) / embed_dim))
                if i % 2 == 0:
                    pos_encoding_matrix[pos, i] = math.sin(angle_rate)
                else:
                    pos_encoding_matrix[pos, i] = math.cos(angle_rate)
        self.register_buffer("pos_encoding", pos_encoding_matrix)

        # Stack of self-attention layers
        self.layers = torch.nn.ModuleList([SelfAttentionBlock(embed_dim) for _ in range(num_layers)])

    def forward(self, inputs):
        # inputs shape: [B, seq_len]
        x = self.emb(inputs)  # [B, seq_len, embed_dim]
        x = x + self.pos_encoding[:inputs.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return x  # encoder representations
