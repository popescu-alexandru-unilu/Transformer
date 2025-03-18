import os
import math
import random
import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm 

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to cleaned text8 corpus
text8_path = './data/cleaned_text8.txt'

# Train SentencePiece model if not already trained
model_prefix = './models/spm_vocab_text8_32k'

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
vocab_size = sp.get_piece_size()
print("SentencePiece vocab size:", vocab_size)

# Read the entire corpus
with open(text8_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

# Encode the entire corpus into token IDs
token_ids = sp.encode(corpus, out_type=int)

# Create training examples for masked language modeling
seq_length = 32              # Sequence length for each example
mask_token = sp.piece_to_id('[MASK]')  # Reserve the '?' as the mask token (ensure it's in your vocab)

training_examples = []
# Slide over the tokenized corpus in chunks of seq_length
for i in range(0, len(token_ids) - seq_length, seq_length):
    sequence = token_ids[i:i+seq_length]
    # Randomly choose a position in the sequence to mask
    mask_pos = random.randint(0, seq_length - 1)
    target = sequence[mask_pos]
    # Replace token with the mask token
    sequence[mask_pos] = mask_token
    training_examples.append((sequence, mask_pos, target))

print(f"Created {len(training_examples)} training examples.")

# --- Define a simple BERT-style model ---

class Magic(torch.nn.Module):
    def __init__(self):
        super(Magic, self).__init__()
        self.W_Q = torch.nn.Linear(9, 9)
        self.W_K = torch.nn.Linear(9, 9)
        self.W_V = torch.nn.Linear(9, 9)

    def forward(self, inputs):
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V)
        return out

class BERT(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BERT, self).__init__()
        self.max_sequence_length = seq_length
        self.embedding_dim = 9
        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim)

        # Create a positional encoding matrix
        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim), device=device)
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                if dimension % 2 == 0:
                    pos_encoding_matrix[position, dimension] = math.sin(angle_rate)
                else:
                    pos_encoding_matrix[position, dimension] = math.cos(angle_rate)
        self.register_buffer("pos_encoding", pos_encoding_matrix)

        # Create a series of self-attention layers
        self.magics = torch.nn.ModuleList([Magic() for _ in range(4)])
        self.vocab_out = torch.nn.Linear(self.embedding_dim, vocab_size)

    def forward(self, inputs):
        # Get word embeddings and add positional encodings
        embs = self.emb(inputs)
        pos_encodings = self.pos_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings
        # Pass through self-attention layers
        for magic in self.magics:
            embs = magic(embs)
        # Project to vocabulary logits and compute log-probabilities
        logits = self.vocab_out(embs)
        probs = F.log_softmax(logits, dim=-1)
        return probs

# Instantiate the model and optimizer
model = BERT(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training loop ---
num_epochs = 1  # For demonstration, we use 1 epoch; adjust as needed
for epoch in range(num_epochs):
    random.shuffle(training_examples)
    total_loss = 0.0
    progress_bar = tqdm(training_examples, desc=f"Epoch {epoch + 1}", unit="batch")
    for seq, mask_pos, target in progress_bar:
        optimizer.zero_grad()
        inp = torch.tensor(seq, device=device).unsqueeze(0)  # Shape: [1, seq_length]
        out = model(inp)
        # Get the output at the masked position
        pred = out[:, mask_pos, :]
        target_tensor = torch.tensor([target], device=device)
        loss = F.nll_loss(pred, target_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(training_examples)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# Save the trained model
save_path = f"../models/bert_encoder_text8-{loss}-{epoch}.pth"
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print("Model saved at", save_path)
