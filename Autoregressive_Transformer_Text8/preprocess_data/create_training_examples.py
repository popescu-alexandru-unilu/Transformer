import os
import pickle
import sentencepiece as spm
from tqdm import tqdm

# Paths
text8_path = '../data/cleaned_text8.txt'
model_prefix = '../models/spm_vocab_text8_32k'
output_file = '../data/gpt_training_examples_single_seq.pkl'

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')

# Read the entire corpus
with open(text8_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

# Encode the corpus into token IDs
token_ids = sp.encode(corpus, out_type=int)

# Define parameters
seq_length = 32  # Each training example will be a sequence of 32 tokens

training_examples = []
# Generate training examples by sliding a window over the token IDs
for i in tqdm(range(0, len(token_ids) - seq_length, seq_length)):
    # Get a window of tokens (length seq_length)
    sequence = token_ids[i:i + seq_length]
    # Store the full sequence as one training example
    training_examples.append(sequence)

# Save the training examples to a file
with open(output_file, 'wb') as f:
    pickle.dump(training_examples, f)

print(f"Created {len(training_examples)} training examples and saved them to {output_file}")
print("First training examples:", training_examples[:10])
