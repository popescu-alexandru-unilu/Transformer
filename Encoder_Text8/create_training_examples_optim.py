import os
import random
import pickle
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# Paths
text8_path = './data/cleaned_text8.txt'
model_prefix = './models/spm_vocab_text8_32k'
output_file = './data/training_examples_cleaned_text8.pkl'

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')

# Read the entire corpus
with open(text8_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

# Encode the corpus into token IDs and convert to a NumPy array for efficient slicing
token_ids = np.array(sp.encode(corpus, out_type=int))

# Define parameters
seq_length = 32
mask_token = sp.piece_to_id('[MASK]')  # Ensure '[MASK]' is reserved in your vocab

# Determine the number of training examples (using non-overlapping segments)
n_examples = (len(token_ids) - seq_length) // seq_length

# Pre-generate random mask positions for each example using NumPy
mask_positions = np.random.randint(0, seq_length, size=n_examples)

training_examples = []

# Use tqdm to track progress over examples
for j in tqdm(range(n_examples), desc="Generating training examples", unit="batch"):
    # Compute start and end indices for the current segment
    start_idx = j * seq_length
    end_idx = start_idx + seq_length
    
    # Copy the segment so that modifying it doesn't affect the original array
    sequence = token_ids[start_idx:end_idx].copy()
    
    # Get the mask position for this example and the original target token
    mask_pos = int(mask_positions[j])
    target = int(sequence[mask_pos])
    
    # Replace the token at mask_pos with the mask token
    sequence[mask_pos] = mask_token
    
    # Append the tuple; convert the sequence back to a list (or keep as NumPy array if preferred)
    training_examples.append((sequence.tolist(), mask_pos, target))

# Save the training examples to a file using pickle
with open(output_file, 'wb') as f:
    pickle.dump(training_examples, f)

print(f"Created {len(training_examples)} training examples and saved them to {output_file}")
print(f"First 10 training examples: {training_examples[:10]}")
