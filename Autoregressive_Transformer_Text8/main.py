import os
import math
import random
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

from decoder import MyDecoder
import sentencepiece as spm

# Load the model
sp = spm.SentencePieceProcessor(model_file='./models/spm_vocab_text8_32k.model')


# -------------------------
# Hyperparameters & Paths
# -------------------------
training_examples_file = './data/gpt_training_examples_single_seq.pkl'
model_save_path = './models/GPT_only_decoding_model.pth'
seq_length = 32              # Each example is a sequence of 32 tokens
d_model = 128
num_layers = 8
d_ff = 256
batch_size = 64
num_epochs = 5               # Adjust as needed
learning_rate = 1e-4
loss_threshold = 0.5         # Save model when average loss is below this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set your vocab size (should match your SentencePiece vocab)
vocab_size = 32000  # Adjust to your actual vocabulary size

# Instantiate the custom decoder (GPT-style but built from scratch)
model = MyDecoder(vocab_size=vocab_size,
                  max_seq_length=seq_length,
                  d_model=d_model,
                  num_layers=num_layers,
                  d_ff=d_ff,
                  device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# Load Training Data
# -------------------------
with open(training_examples_file, 'rb') as f:
    training_examples = pickle.load(f)
print(f"Loaded {len(training_examples)} training examples.")

# -------------------------
# Helper: Autoregressive Text Generation
# -------------------------
def generate_text(model, init_sequence, gen_steps=10):
    """
    Given an initial sequence (list of token IDs), generate additional tokens.
    """
    model.eval()
    generated = init_sequence.copy()
    with torch.no_grad():
        for _ in range(gen_steps):
            inp = torch.tensor([generated], device=device)  # [1, current_length]
            output = model(inp)  # [1, current_length, vocab_size]
            # Get next token from last time step
            next_token = output[0, -1].argmax(dim=-1).item()
            generated.append(next_token)
            # Optionally, you can break if an EOS token is predicted.
    model.train()
    return generated

# -------------------------
# Training Loop
# -------------------------
for epoch in range(1, num_epochs + 1):
    random.shuffle(training_examples)
    total_loss = 0.0
    num_batches = (len(training_examples) + batch_size - 1) // batch_size
    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch}", unit="batch")

    for batch_idx in progress_bar:
        batch_examples = training_examples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        # Convert list of sequences to tensor of shape [B, seq_length]
        batch_tensor = torch.tensor(batch_examples, device=device)
        
        # For autoregressive training, split into input (first seq_length-1 tokens)
        # and target (last seq_length-1 tokens), i.e., next-token prediction.
        inputs = batch_tensor[:, :-1]    # shape: [B, seq_length-1]
        targets = batch_tensor[:, 1:]    # shape: [B, seq_length-1]

        optimizer.zero_grad()
        # Forward pass: output shape [B, seq_length-1, vocab_size]
        output = model(inputs)
        loss = F.nll_loss(output.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())

        # Occasional inference: every 1000 batches, generate text from the first example in batch.
        if batch_idx % 1000 == 0:
            example = inputs[0].tolist()
            generated = generate_text(model, example, gen_steps=10)
            print("\n[Inference] Input:", example)
            print("[Inference] Generated:", generated)
            input_text = sp.decode(example)
            generated_text = sp.decode(generated)
            print("[Inference] Input (text):", input_text)
            print("[Inference] Generated (text):", generated_text)
    

    avg_loss = total_loss / len(training_examples)
    print(f"\nEpoch {epoch} completed. Average Loss: {avg_loss:.4f}")

    # Save model if average loss is below threshold
    if avg_loss < loss_threshold:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path} because average loss {avg_loss:.4f} < {loss_threshold}")
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved at {model_save_path} after training completion.")
