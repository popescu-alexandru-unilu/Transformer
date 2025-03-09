import os
import torch
import torch.nn.functional as F
import math

# (Re)define your model architecture and vocabulary (must match training)
vocab = {"?": 0, "A": 1, "B": 2, "C": 3, "D": 4}

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
    def __init__(self):
        super(BERT, self).__init__()
        self.max_sequence_length = 128
        self.embedding_dim = 9
        self.emb = torch.nn.Embedding(5, 9)

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
        self.vocab = torch.nn.Linear(9, 5)

    def forward(self, inputs):
        embs = self.emb(inputs)
        pos_encodings = self.pos_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings
        for magic in self.magics:
            embs = magic(embs)
        logits = self.vocab(embs)
        probs = F.log_softmax(logits, dim=-1)
        return probs

# Ensure the correct device is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate and load the trained model
B = BERT().to(device)
save_path = "../models/trained_model1.2576768398284912_131.pth"  # Update this path if necessary
B.load_state_dict(torch.load(save_path, map_location=device))
B.eval()  # Set to evaluation mode

# Define a new sequence to evaluate
# new_sequence = ["C", "C", "B", "?", "B"]  # C
new_sequence = ["D", "A", "C", "?", "C"]  # A
# new_sequence = ["A", "?", "B", "C"]  # A

# Prepare input tensor for the new sequence
new_ips = torch.tensor([vocab[w] for w in new_sequence], device=device).unsqueeze(0)
print("new_ips contains: ",new_ips)

# Evaluate the model without tracking gradients
with torch.no_grad():
    out = B(new_ips)

# Get the prediction for the missing token ("?")
missing_idx = new_sequence.index('?')
prd = out[:, missing_idx, :]
predicted_token_id = prd.argmax(dim=-1).item()
predicted_token = list(vocab.keys())[list(vocab.values()).index(predicted_token_id)]
print(f"Predicted token for {new_sequence}: {predicted_token}")
