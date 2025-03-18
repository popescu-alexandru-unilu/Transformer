import os
import pickle
import sentencepiece as spm

# --- File Paths ---
training_examples_file = './data/training_examples_cleaned_text8.pkl'
model_prefix = './models/spm_vocab_text8_32k'  # Path to your trained SentencePiece model

# --- Load SentencePiece model ---
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')

# --- Load training examples from the pickle file ---
with open(training_examples_file, 'rb') as f:
    training_examples = pickle.load(f)

# --- Display dataset details ---
num_examples = len(training_examples)
print(f"✅ Loaded {num_examples} training examples!")

# --- Explore a few samples ---
print("\n📌 Sample Training Examples:")
for i in range(5):  # Print 5 random samples
    sequence, mask_pos, target = training_examples[i]
    
    # Decode token IDs to words
    original_tokens = sp.decode(sequence)  # Decoded sentence with masked token
    target_word = sp.id_to_piece(target)   # Original word that was masked
    
    print(f"\n📝 Example {i+1}:")
    print(f"Tokenized (IDs): {sequence}")
    print(f"Masked Position: {mask_pos}")
    print(f"Masked Word (Target): {target_word}")
    print(f"Decoded Sentence (with [MASK]): {original_tokens}")

print("\n🔍 Exploration Complete!")
