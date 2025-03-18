import os
import math
import random
import pickle
import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# File paths
training_examples_file = './data/training_examples_cleaned_text8.pkl'
model_prefix = './models/spm_vocab_text8_32k'

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
vocab_size = sp.get_piece_size()
print("SentencePiece vocab size:", vocab_size)

# Load training examples from file
with open(training_examples_file, 'rb') as f:
    training_examples = pickle.load(f)
print(f"Loaded {len(training_examples)} training examples.")

# --- Define the model (same as before) ---
class Magic(torch.nn.Module):
    def __init__(self):
        super(Magic, self).__init__()
        self.W_Q = torch.nn.Linear(64, 64)
        self.W_K = torch.nn.Linear(64, 64)
        self.W_V = torch.nn.Linear(64, 64)

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
        self.max_sequence_length = 32
        self.embedding_dim = 64
        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim)

        # Create positional encoding matrix
        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim), device=device)
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                if dimension % 2 == 0:
                    pos_encoding_matrix[position, dimension] = math.sin(angle_rate)
                else:
                    pos_encoding_matrix[position, dimension] = math.cos(angle_rate)
        self.register_buffer("pos_encoding", pos_encoding_matrix)

        self.magics = torch.nn.ModuleList([Magic() for _ in range(4)])
        self.vocab_out = torch.nn.Linear(self.embedding_dim, vocab_size)

    def forward(self, inputs):
        embs = self.emb(inputs)
        pos_encodings = self.pos_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings
        for magic in self.magics:
            embs = magic(embs)
        logits = self.vocab_out(embs)
        probs = F.log_softmax(logits, dim=-1)
        return probs

# Instantiate model and optimizer
model = BERT(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Batch Training Loop with tqdm ---
num_epochs = 5      # Adjust as needed
batch_size = 128    # Adjust batch size as needed

for epoch in range(num_epochs):
    random.shuffle(training_examples)
    total_loss = 0.0
    # Create batches by slicing the training examples list
    num_batches = (len(training_examples) + batch_size - 1) // batch_size
    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}", unit="batch")

    for batch_idx in progress_bar:
        # Create the batch
        batch = training_examples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        # Unzip the batch into sequences, mask positions, and targets
        batch_sequences = [ex[0] for ex in batch]      # list of lists, each of length 32
        batch_mask_pos = [ex[1] for ex in batch]         # list of indices
        batch_targets = [ex[2] for ex in batch]          # list of target token IDs

        # Convert lists to tensors
        inp = torch.tensor(batch_sequences, device=device)  # Shape: [B, 32]
        out = model(inp)  # Shape: [B, 32, vocab_size]

        # For each example in the batch, select the prediction at its mask position.
        # Create a tensor of mask positions (B,) then expand to gather predictions.
        mask_pos_tensor = torch.tensor(batch_mask_pos, device=device).unsqueeze(1)  # Shape: [B, 1]
        # Gather predictions at mask positions for each batch item.
        # First, expand mask_pos_tensor to shape [B, 1, vocab_size]
        gathered = out.gather(dim=1, index=mask_pos_tensor.unsqueeze(2).expand(-1, 1, out.size(2)))  # [B, 1, vocab_size]
        pred = gathered.squeeze(1)  # Shape: [B, vocab_size]

        # Create target tensor
        target_tensor = torch.tensor(batch_targets, device=device)  # Shape: [B]

        loss = F.nll_loss(pred, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)  # Multiply by batch size for sum

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(training_examples)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# Save the trained model
save_path = f"./models/bert_encoder_text8-{avg_loss:.4f}-{epoch}.pth"
torch.save(model.state_dict(), save_path)
print("Model saved at", save_path)
