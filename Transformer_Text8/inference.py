# inference.py
import os
import random 
import torch
import pickle
import sentencepiece as spm
import torch.nn.functional as F
from encoder import TransformerEncoder
from decoder import TransformerDecoder

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# File paths -- update these paths as needed!
model_prefix = './models/spm_vocab_text8_32k'
checkpoint_path = './models/transformer_model-0.0899-4.pth'  # Replace with your actual checkpoint file
training_examples_file = './data/training_examples_cleaned_text8.pkl'

# Load the SentencePiece model
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
vocab_size = sp.get_piece_size()
print("SentencePiece vocab size:", vocab_size)

# Instantiate the encoder and decoder
encoder = TransformerEncoder(vocab_size, max_sequence_length=32, embed_dim=64, num_layers=4, device=device).to(device)
decoder = TransformerDecoder(vocab_size, max_sequence_length=32, embed_dim=64, num_layers=4, device=device).to(device)

# Load the saved checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()
print("Model loaded and set to eval mode.")

# Load one training example to test inference (or construct your own input)
with open(training_examples_file, 'rb') as f:
    training_examples = pickle.load(f)

# Assume each training example is a tuple like: (source_sequence, target_sequence, target_token)
# We'll use the source sequence containing a [MASK] token.
example = random.choice(training_examples)
input_sequence = example[0]  # a list of token IDs of length 32
mask_token = sp.piece_to_id('[MASK]')

# Display the original masked sentence
print("Original (masked):", sp.decode(input_sequence))

# --- Inference process ---

# 1. Prepare the source sequence as a tensor: shape [1, seq_length]
src = torch.tensor([input_sequence], device=device)  # [1, 32]

# 2. Run the encoder on the input sequence
encoder_outputs = encoder(src)  # shape: [1, 32, embed_dim]

# 3. Prepare the decoder input.
# Since our training expected a single token target (the missing token),
# we supply the [MASK] token as the initial target.
tgt = torch.tensor([mask_token], device=device)  # shape: [1] (will be unsqueezed to [1,1] inside decoder)

# 4. Run the decoder to get the prediction.
output_probs = decoder(tgt, encoder_outputs)  # shape: [1, 1, vocab_size]
predicted_token = output_probs.argmax(dim=-1).item()  # Get the predicted token ID

print("Predicted token:", sp.id_to_piece(predicted_token))

# 5. Replace the [MASK] token in the input sequence with the prediction and decode the result.
filled_sequence = input_sequence.copy()
if mask_token in filled_sequence:
    mask_index = filled_sequence.index(mask_token)
    filled_sequence[mask_index] = predicted_token

print("Filled sentence:", sp.decode(filled_sequence))

