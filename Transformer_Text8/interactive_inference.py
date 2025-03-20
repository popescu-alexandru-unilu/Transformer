# interactive_inference.py
import torch
import sentencepiece as spm
from encoder import TransformerEncoder
from decoder import TransformerDecoder

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# File paths and parameters
model_prefix = './models/spm_vocab_text8_32k'
checkpoint_path = './models/transformer_model-0.0899-4.pth'  # Replace with your actual checkpoint file
max_seq_length = 32

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
vocab_size = sp.get_piece_size()
print("SentencePiece vocab size:", vocab_size)

# Instantiate the encoder and decoder
encoder = TransformerEncoder(vocab_size, max_sequence_length=max_seq_length, embed_dim=64, num_layers=4, device=device).to(device)
decoder = TransformerDecoder(vocab_size, max_sequence_length=max_seq_length, embed_dim=64, num_layers=4, device=device).to(device)

# Load the saved checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()
print("Model loaded and set to eval mode.")

# Request user input
print("Enter a sentence with the [MASK] token (max 32 tokens):")
user_input = input("Input sentence: ")

# Tokenize the input text
tokens = sp.encode(user_input, out_type=int)

# Truncate or pad to match max_seq_length
if len(tokens) > max_seq_length:
    tokens = tokens[:max_seq_length]
elif len(tokens) < max_seq_length:
    # Here we use 0 as the pad token (adjust if your tokenizer uses a different pad token)
    tokens = tokens + [0] * (max_seq_length - len(tokens))

print("Tokenized input:", tokens)

# Ensure the input contains the [MASK] token
mask_token = sp.piece_to_id('[MASK]')
if mask_token not in tokens:
    print("Error: Your input must contain a [MASK] token.")
    exit(1)

# Prepare tensors for the encoder (batch size = 1)
src = torch.tensor([tokens], device=device)  # Shape: [1, max_seq_length]

# For the decoder, we supply the [MASK] token as the initial input
tgt = torch.tensor([mask_token], device=device)  # Shape: [1]

# Run the encoder
encoder_outputs = encoder(src)  # Shape: [1, max_seq_length, embed_dim]

# Run the decoder to predict the missing token
output_probs = decoder(tgt, encoder_outputs)  # Shape: [1, 1, vocab_size]
predicted_token = output_probs.argmax(dim=-1).item()
print("Predicted token:", sp.id_to_piece(predicted_token))

# Replace the [MASK] token with the predicted token
filled_tokens = tokens.copy()
mask_index = filled_tokens.index(mask_token)
filled_tokens[mask_index] = predicted_token

# Decode the filled sequence back to text
completed_sentence = sp.decode(filled_tokens)
print("Completed sentence:", completed_sentence)
