# decoder.py
import math
import torch
import torch.nn.functional as F

class MaskedSelfAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MaskedSelfAttentionBlock, self).__init__()
        self.W_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_K = torch.nn.Linear(embed_dim, embed_dim)
        self.W_V = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs):
        # inputs: [B, seq_len, embed_dim]
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        # Create a causal mask (upper triangular) to block future tokens
        seq_len = inputs.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=inputs.device) * float('-inf'), diagonal=1)
        attn_scores = attn_scores + causal_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V)
        return out

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionBlock, self).__init__()
        self.W_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_K = torch.nn.Linear(embed_dim, embed_dim)
        self.W_V = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, decoder_inputs, encoder_outputs):
        # decoder_inputs: [B, tgt_seq_len, embed_dim]
        # encoder_outputs: [B, src_seq_len, embed_dim]
        Q = self.W_Q(decoder_inputs)
        K = self.W_K(encoder_outputs)
        V = self.W_V(encoder_outputs)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V)
        return out

class TransformerDecoder(torch.nn.Module):
    def __init__(self, vocab_size, max_sequence_length=32, embed_dim=64, num_layers=4, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim
        self.emb = torch.nn.Embedding(vocab_size, embed_dim)
        
        # Create positional encoding matrix for decoder
        pos_encoding_matrix = torch.zeros((max_sequence_length, embed_dim), device=device)
        for pos in range(max_sequence_length):
            for i in range(embed_dim):
                angle_rate = pos / (10000 ** (2 * (i // 2) / embed_dim))
                if i % 2 == 0:
                    pos_encoding_matrix[pos, i] = math.sin(angle_rate)
                else:
                    pos_encoding_matrix[pos, i] = math.cos(angle_rate)
        self.register_buffer("pos_encoding", pos_encoding_matrix)
        
        # Masked self-attention layers for decoder
        self.self_attn_layers = torch.nn.ModuleList([MaskedSelfAttentionBlock(embed_dim) for _ in range(num_layers)])
        # Cross-attention layers to attend over encoder outputs
        self.cross_attn_layers = torch.nn.ModuleList([CrossAttentionBlock(embed_dim) for _ in range(num_layers)])
        self.output_layer = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, target_inputs, encoder_outputs):
        # If target_inputs is 1D (i.e. a single token sequence), unsqueeze to add batch dimension.
        if target_inputs.dim() == 1:
            target_inputs = target_inputs.unsqueeze(1)
        # target_inputs: [B, tgt_seq_len]
        x = self.emb(target_inputs)
        # Add positional encoding; note that we slice based on the sequence length.
        x = x + self.pos_encoding[:x.size(1), :]
        
        # Apply masked self-attention layers (with causal mask)
        for self_attn in self.self_attn_layers:
            x = self_attn(x)
        
        # Apply cross-attention layers over encoder outputs
        for cross_attn in self.cross_attn_layers:
            x = cross_attn(x, encoder_outputs)
        
        logits = self.output_layer(x)
        probs = F.log_softmax(logits, dim=-1)
        return probs
