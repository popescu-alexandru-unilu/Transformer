import os
import random
import pickle
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

# Encode the corpus into token IDs
token_ids = sp.encode(corpus, out_type=int)

# Define parameters
seq_length = 32
mask_token = sp.piece_to_id('[MASK]')  # Ensure '[MASK]' is reserved in your vocab

training_examples = []
# Generate training examples by sliding over the token IDs in fixed chunks

for i in tqdm(range(0, len(token_ids) - seq_length, seq_length)):
    sequence = token_ids[i:i+seq_length]
    mask_pos = random.randint(0, seq_length - 1)  # Randomly choose a token to mask
    target = sequence[mask_pos]
    sequence[mask_pos] = mask_token  # Replace with mask token
    training_examples.append((sequence, mask_pos, target))

# Save the training examples to a file
with open(output_file, 'wb') as f:
    pickle.dump(training_examples, f)

print(f"Created {len(training_examples)} training examples and saved them to {output_file}")
print(f"First training examples: {training_examples[:10]} ")
