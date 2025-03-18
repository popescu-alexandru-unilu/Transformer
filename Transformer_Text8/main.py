# main.py
import os
import math
import random
import pickle
import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm

from encoder import TransformerEncoder
from decoder import TransformerDecoder

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

# Load training examples (assumed to be tuples: (source_sequence, target_sequence))
with open(training_examples_file, 'rb') as f:
    training_examples = pickle.load(f)
print(f"Loaded {len(training_examples)} training examples.")

# Instantiate the encoder and decoder modules
encoder = TransformerEncoder(vocab_size, max_sequence_length=32, embed_dim=64, num_layers=4, device=device).to(device)
decoder = TransformerDecoder(vocab_size, max_sequence_length=32, embed_dim=64, num_layers=4, device=device).to(device)

# Setup optimizer for both encoder and decoder
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

# Training parameters
num_epochs = 5
batch_size = 128

for epoch in range(num_epochs):
    random.shuffle(training_examples)
    total_loss = 0.0
    num_batches = (len(training_examples) + batch_size - 1) // batch_size
    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}", unit="batch")

    for batch_idx in progress_bar:
        batch = training_examples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        # Each example is assumed to be a tuple: (source_sequence, target_sequence)
        batch_source = [ex[0] for ex in batch]
        batch_target = [ex[1] for ex in batch]

        src = torch.tensor(batch_source, device=device)  # Shape: [B, src_seq_length]
        tgt = torch.tensor(batch_target, device=device)  # Shape: [B, tgt_seq_length]
        
        encoder_outputs = encoder(src)
        output_probs = decoder(tgt, encoder_outputs)
        # Compute loss across all tokens in target sequence
        loss = F.nll_loss(output_probs.view(-1, vocab_size), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(training_examples)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# Save the transformer model (encoder and decoder)
save_path = f"./models/transformer_model-{avg_loss:.4f}-{epoch}.pth"
torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, save_path)
print("Model saved at", save_path)
