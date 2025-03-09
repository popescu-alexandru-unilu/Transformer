import os
import torch
import torch.nn.functional as F
import math


print("Is CUDA available:", torch.cuda.is_available())

# Display the CUDA version PyTorch was built with
print("PyTorch CUDA version:", torch.version.cuda)

# Display the cuDNN version PyTorch is using
print("cuDNN version:", torch.backends.cudnn.version())

# Ensure model runs on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Vocabulary
vocab = {"?": 0, "A": 1, "B": 2, "C": 3, "D": 4}

# Training data
train = [
    (["A", "?", "B"], "A"),
    (["C", "B", "?"], "B"),
    (["B", "C", "?"], "C"),
    (["A", "?", "A"], "A"),
    (["B", "A", "?"], "A"),
    (["C", "?", "B"], "C"),
    (["A", "B", "?", "A"], "B"),
    (["C", "A", "C", "?"], "A"),
    (["B", "?", "B", "A"], "A"),
    (["A", "B", "C", "?"], "D"),
    (["?", "B", "C", "D"], "A"),
    (["A", "?", "C", "D"], "B"),
    (["A", "A", "?", "B", "B"], "A"),
    (["C", "C", "B", "?", "B"], "C"),
    (["B", "A", "B", "?", "A"], "B")
]

# Set manual seed for reproducibility
torch.manual_seed(29)

# Self-attention module
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

# BERT-style model
class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.max_sequence_length = 128
        self.embedding_dim = 9
        self.emb = torch.nn.Embedding(5, 9)

        # Positional encoding matrix on GPU
        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim), device=device)
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                if dimension % 2 == 0:
                    pos_encoding_matrix[position, dimension] = math.sin(angle_rate)
                else:
                    pos_encoding_matrix[position, dimension] = math.cos(angle_rate)

        self.register_buffer("pos_encoding", pos_encoding_matrix)  # Store on GPU
        self.magics = torch.nn.ModuleList([Magic() for _ in range(4)])
        self.vocab = torch.nn.Linear(9, 5)

    def forward(self, inputs):
        embs = self.emb(inputs)  # Get word embeddings
        pos_encodings = self.pos_encoding[:inputs.size(1), :]  # Get relevant positional encodings
        embs = embs + pos_encodings  # Add positional encodings
        for magic in self.magics:
            embs = magic(embs)  # Apply self-attention layers
        logits = self.vocab(embs)  # Linear projection to vocab size
        probs = F.log_softmax(logits, dim=-1)
        return probs

# Move model to GPU
B = BERT().to(device)

# Optimizer
optimizer = torch.optim.Adam(B.parameters(), lr=0.0001)

# Training loop
stop_training = False
for epoch in range(10000):
    correct_count = 0
    for input_sequence, target in train:
        optimizer.zero_grad()

        # Convert input sequence to tensor & move to GPU
        ips = torch.tensor([vocab[w] for w in input_sequence], device=device).unsqueeze(0)
        out = B(ips)

        idx = input_sequence.index('?')
        prd = out[:, idx, :]

        # Convert target token to tensor & move to GPU
        tgt = torch.tensor([vocab[target]], device=device)
        loss = F.nll_loss(prd, tgt)
        loss.backward()
        optimizer.step()

        # Get predicted token
        predicted_token_id = prd.argmax(dim=-1).item()
        predicted_token = list(vocab.keys())[list(vocab.values()).index(predicted_token_id)]
        print(f"Predicted token for {input_sequence} (target: {target}): {predicted_token}")
        
        # Stop condition
        if predicted_token == target:
            correct_count += 1
        else:
            correct_count = 0  

        if correct_count == 4:
            stop_training = True
            print("Model correctly predicted all targets in this epoch. Stopping training.")
            break

        print("Loss:", loss.item())
        
    if stop_training:
        print(f"Training completed successfully after {epoch + 1} epochs.")

        # **Save the model (outside the loop)**
        save_path = r"C:\Users\bobia\trained_model.pth"

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model on GPU
        torch.save(B.state_dict(), save_path)
        print(f"Model saved successfully at '{save_path}'.")
        break
